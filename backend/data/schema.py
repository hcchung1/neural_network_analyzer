"""Derived feature, token, oracle, and binary-record schema contracts."""

import argparse
import ast
import json
import torch
from torch.utils.data import (
    Dataset,
    IterableDataset,
    Sampler,
    get_worker_info,
)
from typing import List, Dict, Tuple, Any, Optional, Sequence, Set
from collections import OrderedDict
import numpy as np
import pandas as pd
import os
import math

# import mahjong_pybind  # Use this if you have a C++ binding for tile
# conversion
from loguru import logger
import struct
import mmap
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from dataclasses import dataclass
from config import *

LEGACY_SEQ_LEN = 1


def _load_cached_max_seq_len(default: int = 136) -> int:
    """Read max_seq_len from cache/weight.txt when available."""
    transformer_dir = os.path.dirname(os.path.dirname(__file__))
    cache_dir = os.path.join(transformer_dir, "cache")
    weight_path = os.path.join(cache_dir, "weight.txt")
    try:
        with open(weight_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                if parts[0] == "max_seq_len":
                    value = int(parts[1])
                    if value <= 0:
                        return default
                    return value
    except FileNotFoundError:
        logger.debug("weight.txt not found; using default max_seq_len")
    except (OSError, ValueError) as exc:
        logger.debug(f"Failed to parse max_seq_len from weight.txt: {exc}")
    return default


NEW_FORMAT_MAX_SEQ_LEN = _load_cached_max_seq_len()
MAX_SEQ_LEN_FALLBACK = (
    LEGACY_SEQ_LEN if INPUT_FORMAT_IS_OLD_SINGLE else NEW_FORMAT_MAX_SEQ_LEN
)  # Actual input sequence length: 1 for old single-snapshot format, from cache for others
MAX_ACTION_TILES = 4
BINARY_RECORD_SIZE_BASE = 1349
BINARY_RECORD_SIZE_WITH_SHANTEN = BINARY_RECORD_SIZE_BASE + OPP_SHANTEN_BYTES
BINARY_RECORD_SIZE_WITH_WAITS = BINARY_RECORD_SIZE_BASE + OPP_WAIT_BYTES
BINARY_RECORD_SIZE_WITH_ORACLE = BINARY_RECORD_SIZE_BASE + OPP_HAND_BYTES
BINARY_RECORD_SIZE_WITH_SHANTEN_AND_WAITS = (
    BINARY_RECORD_SIZE_BASE + OPP_SHANTEN_BYTES + OPP_WAIT_BYTES
)
BINARY_RECORD_SIZE_WITH_WAITS_AND_ORACLE = (
    BINARY_RECORD_SIZE_BASE + OPP_WAIT_BYTES + OPP_HAND_BYTES
)
BINARY_RECORD_SIZE_WITH_SHANTEN_AND_ORACLE = (
    BINARY_RECORD_SIZE_BASE + OPP_SHANTEN_BYTES + OPP_HAND_BYTES
)
BINARY_RECORD_SIZE_WITH_SHANTEN_AND_WAITS_AND_ORACLE = (
    BINARY_RECORD_SIZE_BASE
    + OPP_SHANTEN_BYTES
    + OPP_WAIT_BYTES
    + OPP_HAND_BYTES
)
BINARY_RECORD_SIZE = (
    BINARY_RECORD_SIZE_BASE
    + (OPP_HAND_BYTES if ORACLE_GUIDING else 0)
    + (OPP_WAIT_BYTES if ORACLE_GUIDING_WAITS else 0)
    + (OPP_SHANTEN_BYTES if ORACLE_GUIDING_SHANTEN else 0)
)


def build_length_conditioned_sample_weights(
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    smoothing: float = LENGTH_CONDITIONAL_REWEIGHT_SMOOTHING,
    clip_min: float = LENGTH_CONDITIONAL_REWEIGHT_CLIP_MIN,
    clip_max: float = LENGTH_CONDITIONAL_REWEIGHT_CLIP_MAX,
    normalize_mean: bool = LENGTH_CONDITIONAL_REWEIGHT_NORMALIZE_MEAN,
) -> torch.Tensor:
    """Compute inverse P(label | seq_length) sample weights for one batch.

    Args:
        labels: Binary labels of shape (B,).
        attention_mask: Optional mask of shape (B, seq_len). If missing,
            fallback length MAX_SEQ_LEN_FALLBACK is used for all samples.
    Returns:
        Tensor of shape (B,) on the same device as `labels`.
    """
    if labels.dim() == 0:
        labels_flat = labels.view(1)
    else:
        labels_flat = labels.view(-1)
    device = labels_flat.device

    if labels_flat.dtype.is_floating_point:
        labels_bin = (labels_flat > 0.5).to(torch.int64)
    else:
        labels_bin = labels_flat.to(torch.int64).clamp(min=0, max=1)

    if attention_mask is None:
        seq_lens = torch.full(
            (labels_bin.numel(),),
            int(MAX_SEQ_LEN_FALLBACK),
            dtype=torch.int64,
            device=device,
        )
    else:
        mask = attention_mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        seq_lens = (mask > 0).sum(dim=1).to(device=device, dtype=torch.int64)
        if seq_lens.numel() != labels_bin.numel():
            fallback_len = int(mask.size(-1)) if mask.dim() >= 2 else int(
                MAX_SEQ_LEN_FALLBACK
            )
            seq_lens = torch.full(
                (labels_bin.numel(),),
                fallback_len,
                dtype=torch.int64,
                device=device,
            )
    seq_lens = seq_lens.clamp(min=1)

    weights = torch.ones(labels_bin.numel(), dtype=torch.float32, device=device)
    if labels_bin.numel() == 0:
        return weights

    alpha = max(float(smoothing), 0.0)
    counts = torch.bincount(seq_lens).to(torch.float32)
    pos_counts = torch.bincount(
        seq_lens,
        weights=labels_bin.to(torch.float32),
        minlength=counts.numel(),
    )
    neg_counts = counts - pos_counts
    denom = counts + 2.0 * alpha
    p_pos = ((pos_counts + alpha) / denom.clamp_min(1e-12)).clamp_min(1e-6)
    p_neg = ((neg_counts + alpha) / denom.clamp_min(1e-12)).clamp_min(1e-6)
    weights = torch.where(
        labels_bin > 0,
        p_pos[seq_lens].reciprocal(),
        p_neg[seq_lens].reciprocal(),
    )

    if clip_min <= clip_max:
        weights = torch.clamp(weights, min=float(clip_min), max=float(clip_max))
    if normalize_mean:
        mean_weight = weights.mean()
        safe_mean = torch.where(
            torch.isfinite(mean_weight) & (mean_weight > 0),
            mean_weight,
            torch.ones_like(mean_weight),
        )
        weights = weights / safe_mean
    return weights


_IMPORTANCE_SAMPLING_TURN_BIN_SIZE = 10


def _normalize_importance_sampling_mode(mode: str) -> str:
    """Normalize importance-sampling mode labels."""
    return str(mode).strip().lower().replace("-", "_").replace(" ", "_")


def _bucketize_turn_numbers(
    turns: np.ndarray, mode: str
) -> np.ndarray:
    """Convert turn numbers into bucket ids for importance sampling."""
    normalized_mode = _normalize_importance_sampling_mode(mode)
    if normalized_mode not in IMPORTANCE_SAMPLING_MODE_CHOICES:
        raise ValueError(
            "Unsupported importance-sampling mode "
            f"'{mode}'. Valid options: "
            f"{sorted(IMPORTANCE_SAMPLING_MODE_CHOICES)}."
        )

    bucket_ids = np.asarray(turns, dtype=np.int32)
    if bucket_ids.ndim != 1:
        raise ValueError(
            f"turns must be 1D for importance sampling. Got {bucket_ids.shape}."
        )
    if np.any(bucket_ids < 0):
        bucket_ids = bucket_ids.copy()
        bucket_ids[bucket_ids < 0] = 0

    if normalized_mode == "turn":
        return bucket_ids

    return (
        bucket_ids // _IMPORTANCE_SAMPLING_TURN_BIN_SIZE
    ) * _IMPORTANCE_SAMPLING_TURN_BIN_SIZE


def build_importance_sampling_weights(
    labels: np.ndarray,
    turns: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build bucket-balanced sampling weights for training-time resampling."""
    labels_np = np.asarray(labels)
    turns_np = np.asarray(turns)
    if labels_np.ndim != 1 or turns_np.ndim != 1:
        raise ValueError(
            "labels and turns must be 1D arrays for importance sampling. "
            f"Got labels={labels_np.shape}, turns={turns_np.shape}."
        )
    if labels_np.size != turns_np.size:
        raise ValueError(
            "labels and turns must have identical lengths for importance "
            f"sampling. Got {labels_np.size} and {turns_np.size}."
        )

    normalized_mode = _normalize_importance_sampling_mode(mode)
    if labels_np.size == 0:
        return np.empty(0, dtype=np.float32), {
            "bucket_count": 0,
            "degenerate_bucket_count": 0,
            "max_turn": 0,
            "min_turn": 0,
        }

    labels_bin = (labels_np > 0).astype(np.uint8, copy=False)
    bucket_ids = _bucketize_turn_numbers(turns_np, normalized_mode)
    unique_buckets, inverse = np.unique(bucket_ids, return_inverse=True)
    bucket_counts = np.bincount(
        inverse, minlength=unique_buckets.size
    ).astype(np.int64, copy=False)
    pos_counts = np.bincount(
        inverse,
        weights=labels_bin.astype(np.float32, copy=False),
        minlength=unique_buckets.size,
    ).astype(np.int64, copy=False)
    neg_counts = bucket_counts - pos_counts

    bucket_weights = np.divide(
        np.ones(bucket_counts.size, dtype=np.float32),
        bucket_counts.astype(np.float32, copy=False),
        out=np.ones(bucket_counts.size, dtype=np.float32),
        where=bucket_counts > 0,
    )
    weights = bucket_weights[inverse]

    balanced_bucket_mask = (pos_counts > 0) & (neg_counts > 0)
    if np.any(balanced_bucket_mask):
        pos_bucket_weights = np.zeros(bucket_counts.size, dtype=np.float32)
        neg_bucket_weights = np.zeros(bucket_counts.size, dtype=np.float32)
        pos_bucket_weights[balanced_bucket_mask] = (
            0.5 / pos_counts[balanced_bucket_mask].astype(np.float32)
        )
        neg_bucket_weights[balanced_bucket_mask] = (
            0.5 / neg_counts[balanced_bucket_mask].astype(np.float32)
        )

        balanced_samples = balanced_bucket_mask[inverse]
        balanced_inverse = inverse[balanced_samples]
        balanced_labels = labels_bin[balanced_samples]
        weights[balanced_samples] = np.where(
            balanced_labels > 0,
            pos_bucket_weights[balanced_inverse],
            neg_bucket_weights[balanced_inverse],
        )

    stats = {
        "bucket_count": int(unique_buckets.size),
        "degenerate_bucket_count": int(
            np.count_nonzero(~balanced_bucket_mask)
        ),
        "max_turn": int(bucket_ids.max()),
        "min_turn": int(bucket_ids.min()),
    }
    return weights.astype(np.float32, copy=False), stats


def build_importance_sampling_loss_correction(
    sampling_weights: np.ndarray,
) -> np.ndarray:
    """Build p(i)/q(i) correction weights for importance-resampled training.

    The sampler draws index i with q(i) = w_i / sum_j w_j. The original
    empirical objective is uniform over samples, p(i)=1/N. Therefore:
        p(i)/q(i) = (sum_j w_j) / (N * w_i)
    """
    weights = np.asarray(sampling_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(
            "sampling_weights must be 1D for IS loss correction. "
            f"Got shape={weights.shape}."
        )
    if weights.size == 0:
        return np.empty(0, dtype=np.float32)
    if np.any(~np.isfinite(weights)):
        raise ValueError("sampling_weights contains non-finite values.")
    if np.any(weights <= 0):
        raise ValueError(
            "sampling_weights must be strictly positive for IS correction."
        )

    total = float(weights.sum())
    if total <= 0:
        raise ValueError("sampling_weights sum must be > 0 for IS correction.")
    count = float(weights.size)
    correction = (total / count) / weights
    return correction.astype(np.float32, copy=False)


ACTION_TYPE_TO_ID = {
    0: 0,  # Chi left
    1: 1,  # Chi middle
    2: 2,  # Chi right
    3: 3,  # Pon
    4: 4,  # Open kan
    5: 5,  # Closed kan
    6: 6,  # Added kan
    7: 7,  # Flip dora
    8: 8,  # Discard (tedashi)
    9: 9,  # Discard (tsumogiri)
}
ACTION_TYPE_LABELS = {
    0: "chi_left",
    1: "chi_middle",
    2: "chi_right",
    3: "pon",
    4: "kan_open",
    5: "kan_closed",
    6: "kan_added",
    7: "flip_dora",
    8: "discard_tedashi",
    9: "discard_tsumogiri",
}
ACTION_TYPE_COUNT = len(ACTION_TYPE_LABELS)
DISCARD_ACTION_IDS = {ACTION_TYPE_TO_ID[8], ACTION_TYPE_TO_ID[9]}
# ACTION_FEATURE_DIM = 1 + 1 + 1 + MAX_ACTION_TILES + DORA_FEATURE_SLOTS
# # pred_id, player_id, action_type category, tiles, dora

FEAT_COLS = 34
NEW_ADD_INIT_MY_HAND_ROWS = len(MY_HAND_ROWS) if INPUT_FORMAT_IS_NEW_ADD else 0
NEW_ADD_INIT_ORACLE_OFFSET_ROWS = (
    STATIC_FEATURE_ROWS + NEW_ADD_INIT_MY_HAND_ROWS
)
F_INIT_ROWS = NEW_ADD_INIT_ORACLE_OFFSET_ROWS + ORACLE_FEATURE_ROWS

F_ACT_ROWS_BASE = 40
F_ACT_ROWS = F_ACT_ROWS_BASE + (
    TIME_SECTION_ROWS if TIME_SECTION_IN_ACTION else 0
)


NEW_ADD_DISCARD_HISTORY_STEPS = 6
NEW_ADD_DISCARD_HISTORY_DIM = 4 * NEW_ADD_DISCARD_HISTORY_STEPS * FEAT_COLS
NEW_ADD_DISCARD_SET_DIM = 4 * FEAT_COLS
NEW_ADD_MELD_SET_DIM = 4 * FEAT_COLS
NEW_ADD_REMAINING_TILES_DIM = FEAT_COLS
NEW_ADD_INIT_OLD_SINGLE_RECENT_DIM = (
    TOKEN_FEATURE_DIM
    if INPUT_FORMAT_IS_NEW_ADD and USE_NEW_ADD_INIT_OLD_SINGLE_RECENT
    else 0
)
NEW_ADD_INIT_EXTRA_DIM = (
    (
        NEW_ADD_DISCARD_HISTORY_DIM
        + NEW_ADD_DISCARD_SET_DIM
        + NEW_ADD_MELD_SET_DIM
        + NEW_ADD_REMAINING_TILES_DIM
    )
    if USE_NEW_ADD_INIT_EXTRA
    else 0
) + NEW_ADD_INIT_OLD_SINGLE_RECENT_DIM

F_INIT_BASE = F_INIT_ROWS * FEAT_COLS
F_INIT = F_INIT_BASE + NEW_ADD_INIT_EXTRA_DIM
F_ACT = F_ACT_ROWS * FEAT_COLS  # 1360
F_PAD = max(F_INIT, F_ACT)  # 1360

NEW_ADD_PLAYER_ROWS = 4
NEW_ADD_ACTION_ROWS = ACTION_TYPE_COUNT
NEW_ADD_TILE_ROWS = FEAT_COLS
NEW_ADD_RIICHI_FLAG_ROWS = 1
NEW_ADD_DORA_FLAG_ROWS = 1
NEW_ADD_DANGER_FLAG_ROWS = 1
NEW_ADD_ONE_CHANCE_FLAG_ROWS = 1
NEW_ADD_NO_CHANCE_FLAG_ROWS = 1
NEW_ADD_ATTACK_EMBED_ROWS = 1
NEW_ADD_DEFENSE_EMBED_ROWS = 1
NEW_ADD_STEP_VISIBLE_DISCARD_DIM = (
    4 * FEAT_COLS if NEW_ADD_INCLUDE_VISIBLE_STATE_PER_ACTION else 0
)
NEW_ADD_STEP_VISIBLE_MELD_DIM = (
    4 * FEAT_COLS if NEW_ADD_INCLUDE_VISIBLE_STATE_PER_ACTION else 0
)
NEW_ADD_STEP_VISIBLE_DIM = (
    NEW_ADD_STEP_VISIBLE_DISCARD_DIM + NEW_ADD_STEP_VISIBLE_MELD_DIM
)
NEW_ADD_ORACLE_BLOCK_DIM = ORACLE_FEATURE_ROWS * FEAT_COLS
NEW_ADD_FINAL_ROWS = 1 if USE_NEW_ADD_FINAL_TOKEN_ONE else 0
NEW_ADD_FINAL_DIM = NEW_ADD_FINAL_ROWS * FEAT_COLS
NEW_ADD_FINAL_CONTEXT_DISCARD_DIM = 4 * FEAT_COLS
NEW_ADD_FINAL_CONTEXT_MELD_DIM = 4 * FEAT_COLS
NEW_ADD_FINAL_CONTEXT_DORA_DIM = DORA_FEATURE_SLOTS * FEAT_COLS
NEW_ADD_FINAL_CONTEXT_HAND_DIM = 4 * FEAT_COLS
NEW_ADD_FINAL_CONTEXT_DIM = (
    (
        NEW_ADD_FINAL_CONTEXT_DISCARD_DIM
        + NEW_ADD_FINAL_CONTEXT_MELD_DIM
        + NEW_ADD_FINAL_CONTEXT_DORA_DIM
        + NEW_ADD_FINAL_CONTEXT_HAND_DIM
    )
    if USE_NEW_ADD_FINAL_TOKEN_CONTEXT
    else 0
)


def compute_physical_seq_len_for_token_placement(
    raw_max_seq_len: Optional[int] = None,
) -> int:
    """Return the actual padded tensor length for the active token mode."""

    if INPUT_FORMAT_IS_OLD_SINGLE:
        return int(LEGACY_SEQ_LEN)

    base_len = int(
        NEW_FORMAT_MAX_SEQ_LEN if raw_max_seq_len is None else raw_max_seq_len
    )
    base_len = max(1, base_len)
    if TOKEN_PLACEMENT_MODE != "last_n_actions":
        return base_len

    custom_len = int(TOKEN_PLACEMENT_LAST_ACTION_TENSOR_LENGTH)
    if custom_len > 0:
        return max(1, custom_len)

    return max(1, 1 + int(TOKEN_PLACEMENT_LAST_ACTION_COUNT))


MAX_SEQ_LEN_FALLBACK = compute_physical_seq_len_for_token_placement(
    MAX_SEQ_LEN_FALLBACK
)
NEW_ADD_OG_TENPAI_FLAG_ROWS = 2 if ORACLE_GUIDING_STEP_TENPAI else 0
NEW_ADD_OG_NEW_TENPAI_FLAG_ROWS = 1 if ORACLE_GUIDING_STEP_NEW_TENPAI else 0
NEW_ADD_OG_FLAG_DIM = (
    NEW_ADD_OG_TENPAI_FLAG_ROWS + NEW_ADD_OG_NEW_TENPAI_FLAG_ROWS
)

NEW_ADD_ORACLE_OFFSET = (
    NEW_ADD_PLAYER_ROWS
    + NEW_ADD_ACTION_ROWS
    + NEW_ADD_TILE_ROWS
    + (TIME_SECTION_ROWS if TIME_SECTION_IN_ACTION else 0)
    + NEW_ADD_RIICHI_FLAG_ROWS
    + NEW_ADD_DORA_FLAG_ROWS
    + NEW_ADD_DANGER_FLAG_ROWS
    + NEW_ADD_ONE_CHANCE_FLAG_ROWS
    + NEW_ADD_NO_CHANCE_FLAG_ROWS
    + NEW_ADD_ATTACK_EMBED_ROWS
    + NEW_ADD_DEFENSE_EMBED_ROWS
    + NEW_ADD_STEP_VISIBLE_DIM
)
NEW_ADD_FINAL_OFFSET = NEW_ADD_ORACLE_OFFSET + NEW_ADD_ORACLE_BLOCK_DIM
NEW_ADD_FINAL_CONTEXT_OFFSET = NEW_ADD_FINAL_OFFSET + NEW_ADD_FINAL_DIM

F_ACT_ADD_BASE_DIM = (
    NEW_ADD_PLAYER_ROWS
    + NEW_ADD_ACTION_ROWS
    + NEW_ADD_TILE_ROWS
    + (TIME_SECTION_ROWS if TIME_SECTION_IN_ACTION else 0)
    + NEW_ADD_RIICHI_FLAG_ROWS
    + NEW_ADD_DORA_FLAG_ROWS
    + NEW_ADD_DANGER_FLAG_ROWS
    + NEW_ADD_ONE_CHANCE_FLAG_ROWS
    + NEW_ADD_NO_CHANCE_FLAG_ROWS
    + NEW_ADD_ATTACK_EMBED_ROWS
    + NEW_ADD_DEFENSE_EMBED_ROWS
    + NEW_ADD_STEP_VISIBLE_DIM
    + NEW_ADD_ORACLE_BLOCK_DIM
    + NEW_ADD_FINAL_DIM
    + NEW_ADD_FINAL_CONTEXT_DIM
)
F_ACT_ADD_DIM = F_ACT_ADD_BASE_DIM + NEW_ADD_OG_FLAG_DIM
NEW_ADD_OG_TENPAI_OFFSET = F_ACT_ADD_BASE_DIM
NEW_ADD_OG_NEW_TENPAI_OFFSET = F_ACT_ADD_BASE_DIM + NEW_ADD_OG_TENPAI_FLAG_ROWS
F_PAD_ADD = max(F_INIT, F_ACT_ADD_DIM)
if INPUT_FORMAT_IS_OLD:
    ACTION_FEATURE_DIM = TOKEN_FEATURE_DIM
elif INPUT_FORMAT_IS_NEW_MULTIPLY:
    ACTION_FEATURE_DIM = F_PAD
else:
    ACTION_FEATURE_DIM = F_PAD_ADD

# Store the wide New_Add init-only tail once in a sidecar instead of padding
# every action token to F_INIT. The model reconstructs the full init token at
# the forward boundary while action tokens keep only the compact prefix.
USE_NEW_ADD_INIT_SIDECAR = INPUT_FORMAT_IS_NEW_ADD
F_INIT_COMPACT = max(F_INIT_BASE, F_ACT_ADD_DIM) if USE_NEW_ADD_INIT_SIDECAR else F_INIT
NEW_ADD_INIT_SIDECAR_DIM = (
    F_INIT - F_INIT_COMPACT if USE_NEW_ADD_INIT_SIDECAR else 0
)
PACKED_ACTION_FEATURE_DIM = (
    F_INIT_COMPACT if USE_NEW_ADD_INIT_SIDECAR else ACTION_FEATURE_DIM
)
if NEW_ADD_INIT_SIDECAR_DIM < 0:
    raise ValueError(
        "NEW_ADD_INIT_SIDECAR_DIM < 0; F_INIT_COMPACT exceeds F_INIT. "
        f"F_INIT_COMPACT={F_INIT_COMPACT}, F_INIT={F_INIT}."
    )
if PACKED_ACTION_FEATURE_DIM < F_ACT_ADD_DIM:
    raise ValueError(
        "PACKED_ACTION_FEATURE_DIM must be >= F_ACT_ADD_DIM. "
        f"Got {PACKED_ACTION_FEATURE_DIM} vs {F_ACT_ADD_DIM}."
    )

# PACKED_FEATURE_DIM = F_INIT + F_ACT  # 850 + 1224 = 2074

# 對齊主程式用來設定 model input_shape 的常數名稱
# （保留第一個 scalar 欄位作為 pred_id）
# ACTION_FEATURE_DIM = 1 + PACKED_FEATURE_DIM

_RECORD_SIZE_CANDIDATES = tuple(
    size
    for size in (
        BINARY_RECORD_SIZE_WITH_SHANTEN_AND_WAITS_AND_ORACLE,
        BINARY_RECORD_SIZE_WITH_WAITS_AND_ORACLE,
        BINARY_RECORD_SIZE_WITH_SHANTEN_AND_WAITS,
        BINARY_RECORD_SIZE_WITH_SHANTEN_AND_ORACLE,
        BINARY_RECORD_SIZE_WITH_WAITS,
        BINARY_RECORD_SIZE_WITH_ORACLE,
        BINARY_RECORD_SIZE_WITH_SHANTEN,
        BINARY_RECORD_SIZE_BASE,
    )
    if size > 0
)
# bytes of slack allowed when inferring record sizes
_RECORD_SIZE_ALIGNMENT_TOLERANCE = 128


def detect_record_size_from_length(
    length: int, *, tolerance: int = _RECORD_SIZE_ALIGNMENT_TOLERANCE
) -> Optional[int]:
    """Return the most plausible record size given the buffer length.

    Accepts small trailing padding (<= tolerance) to accommodate per-file metadata.
    """

    best_candidate: Optional[int] = None
    smallest_remainder: Optional[int] = None

    for candidate in _RECORD_SIZE_CANDIDATES:
        if candidate <= 0 or length < candidate:
            continue
        remainder = length % candidate
        if remainder == 0:
            return candidate
        if tolerance > 0 and remainder <= tolerance:
            if (
                best_candidate is None
                or smallest_remainder is None
                or remainder < smallest_remainder
            ):
                best_candidate = candidate
                smallest_remainder = remainder

    return best_candidate


def resolve_record_size_from_buffer(raw: bytes) -> Tuple[int, int]:
    """Infer (record_size, num_records) from a raw binary buffer."""

    size = detect_record_size_from_length(len(raw))
    if size is None or size == 0:
        return 0, 0

    num_records = len(raw) // size
    if num_records == 0:
        return 0, 0

    remainder = len(raw) - num_records * size
    if remainder > 0:
        logger.debug(
            f"Truncating {remainder} trailing bytes from buffer of {len(raw)} to align with record size {size}."
        )

    return size, num_records


def oracle_should_reveal(
    epoch_index: int, step_index: int, total_steps: int
) -> bool:
    """Decide whether oracle features remain visible for the given training step."""

    if not ORACLE_GUIDING or ORACLE_FEATURE_ROWS == 0:
        return False
    if epoch_index < ORACLE_GUIDING_WARMUP_EPOCHS:
        return True
    if total_steps <= 0:
        return False
    visible_steps = max(
        1, int(math.ceil(total_steps * ORACLE_GUIDING_PARTIAL_RATIO))
    )
    return step_index < visible_steps


def apply_oracle_feature_visibility(
    batch_tensor: torch.Tensor,
    reveal: bool,
    *,
    mask_step_flags: bool = False,
) -> None:
    """Mask oracle-only feature slices when visibility is disabled.

    - Oracle leaked rows are zeroed when hidden.
    - Oracle waits rows are filled with 0.5 when hidden.
    - Oracle shanten rows are filled with -1 when hidden.
    - Optional step OG flags (New_Add tail dims) are filled with 0.5 when hidden.
    """

    if reveal:
        return
    if batch_tensor is None or batch_tensor.dim() < 3:
        return
    feature_dim = batch_tensor.size(-1)
    zero_ranges: List[Tuple[int, int]] = []

    if ORACLE_GUIDING and ORACLE_FEATURE_ROWS > 0:
        if INPUT_FORMAT_IS_NEW_ADD:
            init_start = NEW_ADD_INIT_ORACLE_OFFSET_ROWS * FEAT_COLS
            init_end = init_start + (ORACLE_FEATURE_ROWS * FEAT_COLS)
            if feature_dim > init_start and init_end > init_start:
                zero_ranges.append((init_start, min(init_end, feature_dim)))

            if (
                NEW_ADD_ORACLE_BLOCK_DIM > 0
                and feature_dim > NEW_ADD_ORACLE_OFFSET
            ):
                add_start = NEW_ADD_ORACLE_OFFSET
                if feature_dim > add_start:
                    add_end = min(
                        add_start + NEW_ADD_ORACLE_BLOCK_DIM, feature_dim
                    )
                    zero_ranges.append((add_start, add_end))
        else:
            if ORACLE_FEATURE_FLAT_DIM > 0:
                start = ORACLE_FEATURE_FLAT_OFFSET
                end = start + ORACLE_FEATURE_FLAT_DIM
                if feature_dim > start:
                    zero_ranges.append((start, min(end, feature_dim)))

    for start, end in zero_ranges:
        if start >= end or feature_dim <= start:
            continue
        batch_tensor[..., start:end] = 0.0

    if ORACLE_WAIT_FEATURE_ROWS > 0 and ORACLE_WAIT_FEATURE_FLAT_DIM > 0:
        if INPUT_FORMAT_IS_NEW_ADD:
            init_wait_start = (
                NEW_ADD_INIT_ORACLE_OFFSET_ROWS + OPP_HAND_FEATURE_ROWS
            ) * FEAT_COLS
            if feature_dim > init_wait_start:
                init_wait_end = min(
                    init_wait_start + ORACLE_WAIT_FEATURE_FLAT_DIM, feature_dim
                )
                if init_wait_end > init_wait_start:
                    batch_tensor[..., init_wait_start:init_wait_end] = 0.5

            add_wait_start = NEW_ADD_ORACLE_OFFSET + (
                OPP_HAND_FEATURE_ROWS * FEAT_COLS
            )
            if feature_dim > add_wait_start:
                add_wait_end = min(
                    add_wait_start + ORACLE_WAIT_FEATURE_FLAT_DIM, feature_dim
                )
                if add_wait_end > add_wait_start:
                    batch_tensor[..., add_wait_start:add_wait_end] = 0.5
        else:
            wait_start = ORACLE_WAIT_FEATURE_FLAT_OFFSET
            if feature_dim > wait_start:
                wait_end = min(
                    wait_start + ORACLE_WAIT_FEATURE_FLAT_DIM, feature_dim
                )
                if wait_end > wait_start:
                    batch_tensor[..., wait_start:wait_end] = 0.5

    if ORACLE_SHANTEN_FEATURE_ROWS > 0 and ORACLE_SHANTEN_FEATURE_FLAT_DIM > 0:
        shanten_start = ORACLE_SHANTEN_FEATURE_FLAT_OFFSET
        if feature_dim > shanten_start:
            shanten_end = min(
                shanten_start + ORACLE_SHANTEN_FEATURE_FLAT_DIM, feature_dim
            )
            if shanten_end > shanten_start:
                batch_tensor[..., shanten_start:shanten_end] = -1.0

    if mask_step_flags and INPUT_FORMAT_IS_NEW_ADD:
        if (
            NEW_ADD_OG_TENPAI_FLAG_ROWS > 0
            and NEW_ADD_OG_TENPAI_OFFSET < feature_dim
        ):
            batch_tensor[
                ...,
                NEW_ADD_OG_TENPAI_OFFSET : NEW_ADD_OG_TENPAI_OFFSET
                + NEW_ADD_OG_TENPAI_FLAG_ROWS,
            ] = 0.5
        if (
            NEW_ADD_OG_NEW_TENPAI_FLAG_ROWS > 0
            and NEW_ADD_OG_NEW_TENPAI_OFFSET < feature_dim
        ):
            batch_tensor[
                ...,
                NEW_ADD_OG_NEW_TENPAI_OFFSET : NEW_ADD_OG_NEW_TENPAI_OFFSET
                + NEW_ADD_OG_NEW_TENPAI_FLAG_ROWS,
            ] = 0.5


@dataclass(frozen=True)
class FeatureSchema:
    """Resolved input dimensions for one concrete training run."""

    raw_max_seq_len: int
    physical_seq_len: int
    action_feature_dim: int
    packed_action_feature_dim: int
    init_sidecar_dim: int
    init_feature_dim: int
    action_token_dim: int


def resolve_runtime_schema(raw_max_seq_len: int) -> FeatureSchema:
    raw = max(1, int(raw_max_seq_len))
    return FeatureSchema(
        raw_max_seq_len=raw,
        physical_seq_len=compute_physical_seq_len_for_token_placement(raw),
        action_feature_dim=int(ACTION_FEATURE_DIM),
        packed_action_feature_dim=int(PACKED_ACTION_FEATURE_DIM),
        init_sidecar_dim=int(NEW_ADD_INIT_SIDECAR_DIM),
        init_feature_dim=int(F_INIT),
        action_token_dim=int(F_ACT_ADD_DIM),
    )


DEFAULT_FEATURE_SCHEMA = resolve_runtime_schema(NEW_FORMAT_MAX_SEQ_LEN)

__all__ = [name for name in globals() if not name.startswith("__")]
