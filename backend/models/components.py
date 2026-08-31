"""Shared normalization, attention, encoder, and classifier components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Sequence, Any, Dict, List, Iterator
from tqdm import tqdm
import pandas as pd
import numpy as np
from loguru import logger
from torch.amp import GradScaler, autocast
import urllib.parse
import math
import json
import re
import os
from array import array
from collections import OrderedDict
from utils.mahjong_utils import (
    FEAT_COLS,
    F_INIT,
    F_INIT_COMPACT,
    F_ACT_ROWS,
    F_ACT_ADD_DIM,
    NEW_ADD_FINAL_DIM,
    NEW_ADD_FINAL_CONTEXT_DIM,
    NEW_ADD_INIT_SIDECAR_DIM,
    PACKED_ACTION_FEATURE_DIM,
    NEW_ADD_ATTACK_EMBED_ROWS,
    NEW_ADD_DEFENSE_EMBED_ROWS,
    MAX_SEQ_LEN_FALLBACK,
    ACTION_FEATURE_DIM,
    TOTAL_FEATURE_ROWS,
    BITSET_OFFSET,
    MY_DISCARD_ROWS,
    MY_TSUMOGIRI_ROWS,
    OPP_DISCARD_ROWS,
    OPP_TSUMOGIRI_ROWS,
    oracle_should_reveal,
    apply_oracle_feature_visibility,
    ORACLE_GUIDING_EVAL_FORCE_REVEAL,
    USE_TOKEN_TYPE_EMBEDDING,
    TOKEN_TYPE_EMBED_DIM,
    USE_SEAT_ACTION_PHASE_EMBEDDING,
    SEAT_EMBED_DIM,
    ACTION_EMBED_DIM,
    PHASE_EMBED_DIM,
    USE_ROPE,
    ROPE_BASE,
    USE_ABS_POS_ENCODING,
    USE_DEBIASED_QK_ATTENTION,
    DEBIASED_QK_USE_ABS,
    DEBIASED_QK_MOMENTUM,
    DEBIASED_QK_EPS,
    USE_GRAD_CLIP,
    GRAD_CLIP_NORM,
    GRAD_FINITE_CHECK_EVERY_N_STEPS,
    ATTENTION_FINITE_CHECK,
    USE_CLS_TOKEN,
    USE_LEARNABLE_POOL,
    MULTI_POOL_QUERIES,
    MULTI_POOL_DROPOUT,
    HEAD_MODE,
    OLD_SINGLE_ENCODER_ABLATION,
    INPUT_FORMAT_IS_OLD_SINGLE,
    INPUT_FORMAT_IS_NEW_ADD,
    INPUT_FORMAT_IS_NEW_MULTIPLY,
    RANDOM_FEATURE_DISTRIBUTION,
    RANDOM_FEATURE_MASK_OLD_SINGLE_DISCARDS_BY_TURN,
    RANDOM_FEATURE_PRESERVE_PADDING,
    RANDOM_FEATURE_REPLACEMENT_SCOPE,
    RANDOM_FEATURE_SCALE,
    RANDOM_FEATURE_SEED,
    PHASE_ID_TO_NAME,
    PHASE_ID_EARLY,
    PHASE_ID_MID,
    PHASE_ID_LATE,
    TURN_MID_START,
    TURN_LATE_START,
    NEW_ADD_PLAYER_ROWS,
    NEW_ADD_ACTION_ROWS,
    NEW_ADD_TILE_ROWS,
    NEW_ADD_RIICHI_FLAG_ROWS,
    NEW_ADD_DORA_FLAG_ROWS,
    NEW_ADD_DANGER_FLAG_ROWS,
    NEW_ADD_ONE_CHANCE_FLAG_ROWS,
    NEW_ADD_NO_CHANCE_FLAG_ROWS,
    TIME_SECTION_ROWS,
    TIME_SECTION_IN_ACTION,
    ACTION_TYPE_COUNT,
    SPLIT_CV_LAMBDA_AT,
    SPLIT_CV_LAMBDA_DF,
    SPLIT_CV_LAMBDA_V,
    SPLIT_KEEP_MERGE_MODE,
    USE_SIMPLE_LINEAR_HEAD,
    TWO_LOGITS_MODE,
    SOFTMAX_LOSS,
    VOG_KL_WEIGHT,
    BITSET_OFFSET,
    BITSET_ROWS_TOTAL,
    MY_DISCARD_ROWS,
    OPP_DISCARD_ROWS,
    EARLY_TURN_POSITIVE_RATE_BOOST,
    EARLY_TURN_BOOST_MAX_TURN,
    EARLY_TURN_MIN_POSITIVE_RATE,
    EARLY_TURN_MIN_TENPAI_RATIO_IN_POSITIVES,
    EARLY_TURN_BOOST_THRESHOLD_STEP,
    STRICT_REPRODUCIBLE,
    USE_LENGTH_CONDITIONAL_REWEIGHT,
    USE_LENGTH_CORR_PENALTY,
    LENGTH_CORR_PENALTY_WEIGHT,
    LENGTH_CORR_PENALTY_EPS,
    USE_LENGTH_GRL,
    LENGTH_GRL_WEIGHT,
    LENGTH_GRL_LAMBDA,
    LENGTH_PREDICTOR_BUCKET_SIZE,
    LENGTH_PREDICTOR_HIDDEN_DIM,
    LENGTH_PREDICTOR_DROPOUT,
    build_length_conditioned_sample_weights,
    # MTP Pre-training
    MTP_PRETRAINING,
    MTP_NUM_TILE_CLASSES,
    MTP_LOSS_WEIGHT,
    MTP_USE_AUX_CLASSIFICATION,
    MTP_AUX_CLASSIFICATION_WEIGHT,
    LENGTH_PERTURBATION,
    LENGTH_PERTURBATION_PROB,
    LENGTH_PERTURB_RANDOM_CROP,
    LENGTH_PERTURB_RANDOM_CROP_PROB,
    LENGTH_PERTURB_MIN_ACTION_KEEP,
    LENGTH_PERTURB_TOKEN_DROPOUT,
    LENGTH_PERTURB_TOKEN_DROPOUT_PROB,
    LENGTH_PERTURB_TOKEN_DROPOUT_RATE,
    LENGTH_PERTURB_RANDOM_PADDING,
    LENGTH_PERTURB_RANDOM_PADDING_PROB,
    PREDICT_SHANTEN,
    SHANTEN_NUM_CLASSES,
    INFO_LINE_CACHE_SIZE,
    PREDICTION_STREAM_CHUNK_SIZE,
)


def _to_two_logits(output: torch.Tensor, mode: str) -> torch.Tensor:
    if output.dim() == 2 and output.size(1) == 2:
        return output
    if output.dim() == 1:
        o = output
    elif output.dim() == 2 and output.size(1) == 1:
        o = output.squeeze(1)
    else:
        o = output.view(output.size(0), -1).mean(dim=1)
    if mode == "zero_pos":
        return torch.stack([torch.zeros_like(o), o], dim=1)
    return torch.stack([-o, o], dim=1)


def _estimate_turn_numbers_from_inputs(
    inputs: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Estimate turn number from discard occupancy for each sample in batch."""
    if inputs is None:
        return None

    if inputs.dim() == 3:
        flat = inputs[:, 0, :]
    elif inputs.dim() == 2:
        flat = inputs
    else:
        return None

    expected_dim = TOTAL_FEATURE_ROWS * FEAT_COLS
    if flat.size(-1) != expected_dim:
        return None

    batch_size = flat.size(0)
    state = flat.view(batch_size, TOTAL_FEATURE_ROWS, FEAT_COLS)
    bitset = state[:, BITSET_OFFSET : BITSET_OFFSET + BITSET_ROWS_TOTAL, :]

    discard_ranges = [MY_DISCARD_ROWS, *OPP_DISCARD_ROWS]
    turn_counts = []
    for rng in discard_ranges:
        seat_rows = bitset[:, rng.start : rng.stop, :]
        seat_discards = seat_rows.abs().sum(dim=-1).gt(0).sum(dim=1)
        turn_counts.append(seat_discards)

    return torch.stack(turn_counts, dim=1).sum(dim=1)


def _find_early_turn_threshold(
    probabilities: torch.Tensor,
    base_threshold: float,
    turn_numbers: Optional[torch.Tensor],
    targets: Optional[torch.Tensor] = None,
) -> float:
    """Find a lower threshold for early turns that satisfies configured ratios."""
    if (
        not EARLY_TURN_POSITIVE_RATE_BOOST
        or turn_numbers is None
        or probabilities is None
    ):
        return float(base_threshold)

    early_mask = turn_numbers < int(EARLY_TURN_BOOST_MAX_TURN)
    if int(early_mask.sum().item()) == 0:
        return float(base_threshold)

    probs_early = probabilities[early_mask].float()
    targets_early = None
    if targets is not None:
        targets_early = targets[early_mask].float()

    step = float(max(1e-6, EARLY_TURN_BOOST_THRESHOLD_STEP))
    for th in np.arange(float(base_threshold), -1e-9, -step):
        pred_early = probs_early > float(th)
        pred_pos_rate = pred_early.float().mean().item()
        if pred_pos_rate < float(EARLY_TURN_MIN_POSITIVE_RATE):
            continue

        if targets_early is not None:
            pred_pos_cnt = float(pred_early.float().sum().item())
            tenpai_in_pos = float(
                ((pred_early == 1) & (targets_early > 0.5)).float().sum().item()
            )
            tenpai_ratio = (
                tenpai_in_pos / pred_pos_cnt if pred_pos_cnt > 0 else 0.0
            )
            if tenpai_ratio < float(EARLY_TURN_MIN_TENPAI_RATIO_IN_POSITIVES):
                continue

        return float(th)

    return float(base_threshold)


def _predict_with_early_turn_boost(
    probabilities: torch.Tensor,
    base_threshold: float,
    turn_numbers: Optional[torch.Tensor],
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply optional early-turn threshold lowering and return binary predictions."""
    effective_th = _find_early_turn_threshold(
        probabilities=probabilities,
        base_threshold=base_threshold,
        turn_numbers=turn_numbers,
        targets=targets,
    )
    if (
        not EARLY_TURN_POSITIVE_RATE_BOOST
        or turn_numbers is None
        or effective_th >= float(base_threshold)
    ):
        return (probabilities > float(base_threshold)).long()

    pred = (probabilities > float(base_threshold)).long()
    early_mask = turn_numbers < int(EARLY_TURN_BOOST_MAX_TURN)
    pred[early_mask] = (probabilities[early_mask] > effective_th).long()
    return pred


def _reduce_weighted_loss(
    per_sample_loss: torch.Tensor, sample_weights: Optional[torch.Tensor]
) -> torch.Tensor:
    """Apply optional sample-wise weights to unreduced loss values."""
    loss_flat = per_sample_loss.view(-1)
    if sample_weights is None:
        return loss_flat.mean()
    weights = sample_weights.view(-1).to(
        device=loss_flat.device, dtype=loss_flat.dtype
    )
    if weights.numel() != loss_flat.numel():
        return loss_flat.mean()
    return (loss_flat * weights).mean()


def _compute_length_corr_penalty(
    score: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    eps: float = LENGTH_CORR_PENALTY_EPS,
) -> torch.Tensor:
    """Penalize shortcut learning on sequence length via corr(score, seq_len)^2."""
    if score is None or attention_mask is None:
        device = (
            score.device
            if isinstance(score, torch.Tensor)
            else (
                attention_mask.device
                if isinstance(attention_mask, torch.Tensor)
                else torch.device("cpu")
            )
        )
        return torch.zeros((), device=device, dtype=torch.float32)

    if attention_mask.dim() < 2:
        return torch.zeros((), device=score.device, dtype=torch.float32)

    score_flat = score.view(-1).float()
    seq_lens = (attention_mask > 0).sum(dim=1).float().view(-1)
    if score_flat.numel() != seq_lens.numel():
        return torch.zeros((), device=score.device, dtype=torch.float32)
    if score_flat.numel() < 2:
        return torch.zeros((), device=score.device, dtype=torch.float32)

    seq_var = seq_lens.var(unbiased=False)
    if seq_var.detach().item() <= float(eps):
        return torch.zeros((), device=score.device, dtype=torch.float32)

    score_centered = score_flat - score_flat.mean()
    len_centered = seq_lens - seq_lens.mean()
    cov = (score_centered * len_centered).mean()
    score_var = (score_centered * score_centered).mean()
    len_var = (len_centered * len_centered).mean()
    denom = torch.sqrt(score_var * len_var + float(eps))
    corr = cov / denom
    corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))
    return torch.clamp(corr * corr, min=0.0, max=1.0).to(dtype=torch.float32)


def _generate_random_feature_batch(
    shape: torch.Size,
    batch_seed: Optional[int],
    distribution: str,
    scale: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate deterministic random features directly on the target device."""
    if generator is None:
        if batch_seed is None:
            raise ValueError("batch_seed is required when generator is not provided.")
        generator = _make_random_feature_generator(device, int(batch_seed))
    if distribution == "uniform":
        random_data = torch.rand(
            shape,
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        random_data = (random_data * 2.0 - 1.0) * float(scale)
    else:
        random_data = torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        random_data = random_data * float(scale)
    return random_data.to(dtype=dtype)


def _make_random_feature_generator(
    device: torch.device,
    seed: int,
) -> torch.Generator:
    """Create an isolated RNG on the same device as the model input."""
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed))
    return generator


def _get_new_add_action_feature_dim(model: nn.Module) -> int:
    """Return the exact New_Add width consumed by the action projection."""
    model_impl = model.module if hasattr(model, "module") else model
    token_mode = getattr(model_impl, "token_mode", None)
    if token_mode is not None and token_mode != "add":
        raise ValueError(
            "new_add_action_tokens requires a New_Add model, but model "
            f"token_mode={token_mode!r}."
        )

    action_dim = getattr(model_impl, "_act_feature_dim", None)
    if action_dim is None:
        projection = getattr(model_impl, "proj_act", None)
        action_dim = getattr(projection, "in_features", None)
    if action_dim is None:
        prior_encoder = getattr(model_impl, "prior_encoder", None)
        projection = getattr(prior_encoder, "proj_act", None)
        action_dim = getattr(projection, "in_features", None)
    if action_dim is None:
        action_dim = F_ACT_ADD_DIM
    return int(action_dim)


def _new_add_action_token_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    """Select logical action tokens for every supported placement mode."""
    if attn_mask.dim() != 2:
        raise ValueError(
            "new_add_action_tokens requires attention_mask with shape (B, T); "
            f"got {tuple(attn_mask.shape)}."
        )
    valid_mask = attn_mask.to(dtype=torch.bool)
    action_mask = valid_mask.clone()
    if action_mask.size(1) > 0:
        action_mask[:, 0] = False

    reserved_tail_tokens = int(NEW_ADD_FINAL_CONTEXT_DIM > 0) + int(
        NEW_ADD_FINAL_DIM > 0
    )
    if reserved_tail_tokens > 0:
        rank_from_end = torch.flip(
            torch.cumsum(
                torch.flip(valid_mask.to(dtype=torch.int32), dims=(1,)),
                dim=1,
            ),
            dims=(1,),
        )
        action_mask &= rank_from_end > reserved_tail_tokens
    return action_mask


def _replace_new_add_action_tokens(
    model: nn.Module,
    data: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    *,
    distribution: str,
    scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Replace only valid New_Add action features, in place, on data.device."""
    if not INPUT_FORMAT_IS_NEW_ADD:
        raise ValueError(
            "new_add_action_tokens requires USE_OLD_INPUT_FORMAT='New_Add'."
        )
    if data.dim() != 3:
        raise ValueError(
            "new_add_action_tokens requires input with shape (B, T, F); "
            f"got {tuple(data.shape)}."
        )
    if attn_mask is None:
        raise ValueError(
            "new_add_action_tokens requires attention_mask so padding and "
            "optional final/context tokens can be preserved."
        )
    if tuple(attn_mask.shape) != tuple(data.shape[:2]):
        raise ValueError(
            "new_add_action_tokens mask/input shape mismatch: "
            f"mask={tuple(attn_mask.shape)}, input={tuple(data.shape)}."
        )

    action_dim = _get_new_add_action_feature_dim(model)
    if action_dim <= 0 or action_dim > data.size(-1):
        raise ValueError(
            "Invalid New_Add action projection width: "
            f"action_dim={action_dim}, packed_dim={data.size(-1)}."
        )
    if data.size(1) <= 1:
        return data

    action_mask = _new_add_action_token_mask(attn_mask)[:, 1:]
    action_values = data[:, 1:, :action_dim]
    random_actions = _generate_random_feature_batch(
        shape=action_values.shape,
        batch_seed=None,
        distribution=distribution,
        scale=scale,
        dtype=data.dtype,
        device=data.device,
        generator=generator,
    )
    action_values.copy_(
        torch.where(action_mask.unsqueeze(-1), random_actions, action_values)
    )
    return data


def _apply_padding_mask_to_random_features(
    random_inputs: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Zero-out padded positions so random features keep the same valid lengths."""
    if attn_mask is None:
        return random_inputs
    if random_inputs.dim() != 3 or attn_mask.dim() != 2:
        return random_inputs
    if (
        random_inputs.size(0) != attn_mask.size(0)
        or random_inputs.size(1) != attn_mask.size(1)
    ):
        return random_inputs
    return random_inputs.masked_fill((~attn_mask).unsqueeze(-1), 0.0)


def _get_model_feature_grid_shape(
    model: nn.Module,
) -> Optional[Tuple[int, int]]:
    rows = getattr(model, "feature_rows", None)
    cols = getattr(model, "feature_cols", None)
    if rows is None or cols is None:
        return None
    rows_i = int(rows)
    cols_i = int(cols)
    if rows_i <= 0 or cols_i <= 0:
        return None
    return rows_i, cols_i


def _get_model_expected_feature_dim(model: nn.Module) -> Optional[int]:
    feature_grid_shape = _get_model_feature_grid_shape(model)
    if feature_grid_shape is not None:
        rows, cols = feature_grid_shape
        return rows * cols
    for attr_name in ("_expected_feature_dim", "input_features"):
        value = getattr(model, attr_name, None)
        if value is not None:
            return int(value)
    input_shape = getattr(model, "input_shape", None)
    if (
        isinstance(input_shape, (tuple, list))
        and len(input_shape) >= 2
        and getattr(model, "seq_len", None) is not None
    ):
        return int(input_shape[1])
    return None


def _get_random_feature_shape_for_model(
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Size:
    feature_grid_shape = _get_model_feature_grid_shape(model)
    if feature_grid_shape is not None:
        rows, cols = feature_grid_shape
        flat_dim = rows * cols
        batch_size = int(inputs.size(0))
        if inputs.dim() == 2:
            return torch.Size([batch_size, flat_dim])
        if inputs.dim() == 3:
            if int(inputs.size(1)) == 1:
                return torch.Size([batch_size, 1, flat_dim])
            if int(inputs.size(1)) == rows and int(inputs.size(2)) == cols:
                return torch.Size([batch_size, rows, cols])
            if int(inputs.size(1)) == cols and int(inputs.size(2)) == rows:
                return torch.Size([batch_size, cols, rows])
            return torch.Size([batch_size, 1, flat_dim])

    random_shape = inputs.shape
    expected_feature_dim = _get_model_expected_feature_dim(model)
    if expected_feature_dim is not None and inputs.dim() >= 3:
        random_shape = torch.Size(
            list(inputs.shape[:-1]) + [int(expected_feature_dim)]
        )
    return random_shape


def _view_old_single_feature_grid(
    tensor: torch.Tensor,
) -> Optional[torch.Tensor]:
    """View old_single inputs as (B, TOTAL_FEATURE_ROWS, FEAT_COLS)."""
    if tensor.dim() == 2:
        total = int(tensor.size(1))
        if total == TOTAL_FEATURE_ROWS * FEAT_COLS:
            return tensor.view(tensor.size(0), TOTAL_FEATURE_ROWS, FEAT_COLS)
    if tensor.dim() == 3 and int(tensor.size(1)) == 1:
        total = int(tensor.size(2))
        if total == TOTAL_FEATURE_ROWS * FEAT_COLS:
            return tensor.view(tensor.size(0), TOTAL_FEATURE_ROWS, FEAT_COLS)
    if tensor.dim() == 3:
        if (
            int(tensor.size(1)) == TOTAL_FEATURE_ROWS
            and int(tensor.size(2)) == FEAT_COLS
        ):
            return tensor
        if (
            int(tensor.size(1)) == FEAT_COLS
            and int(tensor.size(2)) == TOTAL_FEATURE_ROWS
        ):
            return tensor.transpose(1, 2)
    return None


def _mask_old_single_random_discards_by_turn(
    random_inputs: torch.Tensor,
    reference_inputs: torch.Tensor,
    enabled: bool = RANDOM_FEATURE_MASK_OLD_SINGLE_DISCARDS_BY_TURN,
) -> torch.Tensor:
    """Mask old_single random discard slots to match each sample's turn count."""
    if not INPUT_FORMAT_IS_OLD_SINGLE:
        return random_inputs
    if not bool(enabled):
        return random_inputs

    random_grid = _view_old_single_feature_grid(random_inputs)
    reference_grid = _view_old_single_feature_grid(reference_inputs)
    if random_grid is None or reference_grid is None:
        return random_inputs
    if random_grid.shape != reference_grid.shape:
        return random_inputs

    slot_specs = [(MY_DISCARD_ROWS, MY_TSUMOGIRI_ROWS)] + list(
        zip(OPP_DISCARD_ROWS, OPP_TSUMOGIRI_ROWS)
    )
    for discard_range, tsumogiri_range in slot_specs:
        discard_rows = torch.tensor(
            [BITSET_OFFSET + r for r in discard_range],
            device=reference_grid.device,
            dtype=torch.long,
        )
        tsumogiri_rows = torch.tensor(
            [BITSET_OFFSET + r for r in tsumogiri_range],
            device=reference_grid.device,
            dtype=torch.long,
        )
        if (
            int(discard_rows.max().item()) >= random_grid.size(1)
            or int(tsumogiri_rows.max().item()) >= random_grid.size(1)
        ):
            continue
        ref_discards = reference_grid.index_select(1, discard_rows)
        visible_counts = (ref_discards.abs().sum(dim=2) > 0).sum(dim=1)
        slot_ids = torch.arange(
            discard_rows.numel(),
            device=random_grid.device,
            dtype=torch.long,
        ).unsqueeze(0)
        keep = slot_ids < visible_counts.unsqueeze(1)
        clear = ~keep
        random_grid[:, discard_rows, :] = random_grid[
            :, discard_rows, :
        ].masked_fill(clear.unsqueeze(-1), 0.0)
        random_grid[:, tsumogiri_rows, :] = random_grid[
            :, tsumogiri_rows, :
        ].masked_fill(clear.unsqueeze(-1), 0.0)
    return random_inputs


def _maybe_replace_with_random_features(
    model: nn.Module,
    data: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    *,
    batch_index: int,
    random_feature_enabled: bool,
    random_feature_seed: int,
    random_feature_distribution: str,
    random_feature_scale: float,
    random_feature_preserve_padding: bool,
    random_feature_replacement_scope: str = RANDOM_FEATURE_REPLACEMENT_SCOPE,
    random_feature_mask_old_single_discards_by_turn: bool = (
        RANDOM_FEATURE_MASK_OLD_SINGLE_DISCARDS_BY_TURN
    ),
    reference_inputs: Optional[torch.Tensor] = None,
    init_sidecar: Optional[torch.Tensor] = None,
    random_feature_generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return deterministic random features when enabled; otherwise original data."""
    if not random_feature_enabled:
        return data, init_sidecar

    replacement_scope = str(random_feature_replacement_scope)
    if replacement_scope == "new_add_action_tokens":
        generator = random_feature_generator
        if generator is None:
            generator = _make_random_feature_generator(
                data.device,
                int(random_feature_seed) + int(batch_index),
            )
        random_data = _replace_new_add_action_tokens(
            model,
            data,
            attn_mask,
            distribution=str(random_feature_distribution),
            scale=float(random_feature_scale),
            generator=generator,
        )
        return random_data, init_sidecar
    if replacement_scope != "all_features":
        raise ValueError(
            "Unsupported random feature replacement scope: "
            f"{replacement_scope!r}."
        )

    model_for_shape = model.module if hasattr(model, "module") else model
    random_shape = _get_random_feature_shape_for_model(model_for_shape, data)
    if init_sidecar is not None and init_sidecar.numel() > 0:
        random_shape = data.shape
    if random_shape != data.shape and int(batch_index) == 0:
        logger.info(
            "[RandomFeature] input shape adjusted from "
            f"{tuple(data.shape)} to model-compatible {tuple(random_shape)}."
        )
    random_data = _generate_random_feature_batch(
        shape=random_shape,
        batch_seed=int(random_feature_seed) + int(batch_index),
        distribution=str(random_feature_distribution),
        scale=float(random_feature_scale),
        dtype=data.dtype,
        device=data.device,
        generator=random_feature_generator,
    )
    if bool(random_feature_preserve_padding):
        random_data = _apply_padding_mask_to_random_features(
            random_inputs=random_data,
            attn_mask=attn_mask,
        )
    if bool(random_feature_mask_old_single_discards_by_turn):
        random_data = _mask_old_single_random_discards_by_turn(
            random_inputs=random_data,
            reference_inputs=reference_inputs if reference_inputs is not None else data,
            enabled=random_feature_mask_old_single_discards_by_turn,
        )
    random_sidecar = init_sidecar
    if init_sidecar is not None and init_sidecar.numel() > 0:
        random_sidecar = _generate_random_feature_batch(
            shape=init_sidecar.shape,
            batch_seed=int(random_feature_seed) + int(batch_index) + 1_000_003,
            distribution=str(random_feature_distribution),
            scale=float(random_feature_scale),
            dtype=init_sidecar.dtype,
            device=init_sidecar.device,
            generator=random_feature_generator,
        )
    return random_data, random_sidecar


def _prepare_model_input_with_sidecar(
    model: nn.Module,
    data: torch.Tensor,
    init_sidecar: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Use compact input when supported; otherwise rebuild one dense batch."""
    if init_sidecar is None or init_sidecar.numel() == 0:
        return data, None

    model_impl = model.module if hasattr(model, "module") else model
    if bool(getattr(model_impl, "supports_init_sidecar", False)):
        return data, init_sidecar

    dense = data.new_zeros(
        data.size(0), data.size(1), ACTION_FEATURE_DIM
    )
    copy_dim = min(data.size(-1), dense.size(-1))
    dense[..., :copy_dim] = data[..., :copy_dim]
    dense[:, 0, F_INIT_COMPACT:F_INIT] = init_sidecar
    return dense, None


def _forward_with_optional_init_sidecar(
    model: nn.Module,
    data: torch.Tensor,
    init_sidecar: Optional[torch.Tensor],
    **kwargs: Any,
) -> Any:
    if init_sidecar is not None:
        kwargs["init_sidecar"] = init_sidecar
    return model(data, **kwargs)


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def _gradient_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambda_)


class LengthPredictor(nn.Module):
    """Predict sequence-length buckets from a classifier representation."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# TOTAL_FEATURE_ROWS (currently 336) captures the 336 × 34 feature grid layout

# ==============================================================================
# Transformer Components
# ==============================================================================


def _group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    if num_channels <= 0:
        raise ValueError("num_channels must be > 0")
    max_groups = max(1, min(max_groups, num_channels))
    for group_count in range(max_groups, 0, -1):
        if (
            num_channels % group_count == 0
            and num_channels // group_count >= 2
        ):
            return nn.GroupNorm(group_count, num_channels)
    return nn.GroupNorm(1, num_channels)


def _sequence_valid_mask(
    x: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(
            (x.size(0), x.size(1)), device=x.device, dtype=torch.bool
        )
    mask = attention_mask.to(device=x.device, dtype=torch.bool)
    if mask.numel() != x.size(0) * x.size(1):
        raise ValueError(
            "Sequence normalization mask must contain B*T values; "
            f"got x={tuple(x.shape)}, mask={tuple(mask.shape)}."
        )
    return mask.reshape(x.size(0), x.size(1))


class SequenceBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm over valid token rows of a Transformer tensor (B, T, C)."""

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.num_features:
            raise ValueError(
                "SequenceBatchNorm1d expects shape (B, T, C) with "
                f"C={self.num_features}, got {tuple(x.shape)}."
            )
        valid_mask = _sequence_valid_mask(x, attention_mask).reshape(-1)
        flat = x.reshape(-1, self.num_features)
        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            return torch.zeros_like(x)
        valid = flat.index_select(0, valid_indices)
        if self.training and valid.size(0) == 1:
            # PyTorch BatchNorm rejects a singleton training batch. Falling
            # back to stored running statistics keeps filtered/final batches
            # usable without silently changing the DataLoader contract.
            normalized = F.batch_norm(
                valid,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        else:
            normalized = super().forward(valid)
        output = torch.zeros_like(flat)
        output.index_copy_(0, valid_indices, normalized)
        return output.reshape_as(x)


class SequenceGroupNorm(nn.GroupNorm):
    """GroupNorm over channels independently for each valid token."""

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.num_channels:
            raise ValueError(
                "SequenceGroupNorm expects shape (B, T, C) with "
                f"C={self.num_channels}, got {tuple(x.shape)}."
            )
        valid_mask = _sequence_valid_mask(x, attention_mask).reshape(-1)
        flat = x.reshape(-1, self.num_channels)
        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            return torch.zeros_like(x)
        valid = flat.index_select(0, valid_indices).unsqueeze(-1)
        normalized = super().forward(valid).squeeze(-1)
        output = torch.zeros_like(flat)
        output.index_copy_(0, valid_indices, normalized)
        return output.reshape_as(x)


class ChannelLayerNorm1d(nn.LayerNorm):
    """Apply LayerNorm over channels at each Conv1d sequence position."""

    def __init__(self, num_channels: int):
        if num_channels <= 0:
            raise ValueError("num_channels must be > 0")
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != self.normalized_shape[0]:
            raise ValueError(
                "ChannelLayerNorm1d expects shape (B, C, L) with "
                f"C={self.normalized_shape[0]}, got {tuple(x.shape)}."
            )
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ChannelLayerNorm2d(nn.LayerNorm):
    """Apply LayerNorm over channels at each Conv2d spatial position."""

    def __init__(self, num_channels: int):
        if num_channels <= 0:
            raise ValueError("num_channels must be > 0")
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.normalized_shape[0]:
            raise ValueError(
                "ChannelLayerNorm2d expects shape (B, C, H, W) with "
                f"C={self.normalized_shape[0]}, got {tuple(x.shape)}."
            )
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _canonical_norm_mode(norm: str, *, convolutional: bool = False) -> str:
    normalized = str(norm).strip().lower()
    aliases = {
        "batchnorm": "bn",
        "batch_norm": "bn",
        "groupnorm": "gn",
        "group_norm": "gn",
        "layernorm": "ln",
        "layer_norm": "ln",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "hybrid":
        return "bn" if convolutional else "ln"
    if normalized not in {"bn", "gn", "ln"}:
        raise ValueError(f"Unknown normalization type: {norm}")
    return normalized


def _make_sequence_norm(
    norm: str, num_channels: int, gn_groups: int = 32
) -> nn.Module:
    mode = _canonical_norm_mode(norm, convolutional=False)
    if mode == "bn":
        return SequenceBatchNorm1d(num_channels)
    if mode == "gn":
        group_norm = _group_norm(num_channels, max_groups=gn_groups)
        return SequenceGroupNorm(group_norm.num_groups, num_channels)
    return nn.LayerNorm(num_channels)


def _apply_sequence_norm(
    normalization: nn.Module,
    x: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if isinstance(normalization, (SequenceBatchNorm1d, SequenceGroupNorm)):
        return normalization(x, attention_mask)
    return normalization(x)


def _make_conv1d_norm(
    norm: str, num_channels: int, gn_groups: int = 32
) -> nn.Module:
    mode = _canonical_norm_mode(norm, convolutional=True)
    if mode == "bn":
        return nn.BatchNorm1d(num_channels)
    if mode == "gn":
        return _group_norm(num_channels, max_groups=gn_groups)
    return ChannelLayerNorm1d(num_channels)


def _make_conv2d_norm(
    norm: str, num_channels: int, gn_groups: int = 32
) -> nn.Module:
    mode = _canonical_norm_mode(norm, convolutional=True)
    if mode == "bn":
        return nn.BatchNorm2d(num_channels)
    if mode == "gn":
        return _group_norm(num_channels, max_groups=gn_groups)
    return ChannelLayerNorm2d(num_channels)


class CNN2DClassificationHead(nn.Module):
    """Treat encoder output (B, seq_len, d_model) as a single-channel 2D feature map
    and apply Conv2d layers for classification, inspired by ViT post-processing.

    Architecture:
        (B, 1, seq_len, d_model)
        -> Conv2d(1, C, 3) + configured norm + GELU
        -> Conv2d(C, 2C, 3) + configured norm + GELU
        -> Conv2d(2C, 4C, 3) + configured norm + GELU
        -> AdaptiveAvgPool2d(1)
        -> Flatten -> Linear(4C, num_out)
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        num_out: int = 2,
        base_channels: int = 64,
        dropout: float = 0.1,
        norm: str = "bn",
    ):
        super().__init__()
        C = base_channels
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, C, kernel_size=3, padding=1),
            _make_conv2d_norm(norm, C),
            nn.GELU(),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            _make_conv2d_norm(norm, C),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(C, C * 2, kernel_size=3, padding=1),
            _make_conv2d_norm(norm, C * 2),
            nn.GELU(),
            nn.Conv2d(C * 2, C * 2, kernel_size=3, padding=1),
            _make_conv2d_norm(norm, C * 2),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(C * 2, C * 4, kernel_size=3, padding=1),
            _make_conv2d_norm(norm, C * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # -> (B, 4C, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(C * 4, C * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(C * 2, num_out),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """x: (B, seq_len, d_model)"""
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)  # zero out padded positions
        x = x.unsqueeze(1)  # (B, 1, seq_len, d_model)
        x = self.conv_layers(x)
        return self.head(x)


class PositionalEncoding(nn.Module):
    # (TODO) 改成對各家摸打牌順序進行
    def __init__(
        self, d_model: int, max_len: int = 5000, dropout: float = 0.1
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # seq_len = x.size(1)
        # pos_encoding = self.pe[:, :seq_len, :] * math.sqrt(self.d_model)
        # x = x + pos_encoding
        # return self.dropout(x)
        seq_len = x.size(1)
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 12, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        mode = str(HEAD_MODE).strip().lower()
        if mode == "whole":
            self.head_mode = "whole"
        elif mode == "split_custom_v":
            self.head_mode = "split_custom_v"
        elif mode == "split_cus_v":
            self.head_mode = "split_cus_v"
        elif mode == "split_rand_v":
            self.head_mode = "split_rand_v"
        elif mode == "split_keep":
            self.head_mode = "split_keep"
        elif mode == "split_cv":
            self.head_mode = "split_cv"
        else:
            raise ValueError(
                "HEAD_MODE must be one of {'whole', 'split_custom_v', 'split_cus_v', 'split_rand_v', 'split_keep', 'split_cv'}"
            )

        self.use_split_heads = self.head_mode != "whole"
        if self.use_split_heads:
            if n_heads < 2:
                raise ValueError(
                    "n_heads must be >= 2 when HEAD_MODE uses split heads"
                )
            self.n_heads_attack = max(1, n_heads // 2)
            self.n_heads_defense = n_heads - self.n_heads_attack
            if self.n_heads_defense == 0:
                self.n_heads_attack -= 1
                self.n_heads_defense = 1
        else:
            self.n_heads_attack = n_heads
            self.n_heads_defense = 0
        self.d_k = d_model // n_heads
        self.scale = self.d_k**-0.5  # (TODO) 預計算縮放因子 What's this for?

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.attack_merge = nn.Linear(
            self.n_heads_attack * self.d_k, d_model, bias=False
        )
        self.defense_merge = (
            None
            if self.n_heads_defense == 0
            else nn.Linear(
                self.n_heads_defense * self.d_k, d_model, bias=False
            )
        )
        if self.head_mode in ("split_keep", "split_cv", "split_cus_v"):
            # Separate V projections, merge layers, and output projections per
            # branch
            self.w_v_at = nn.Linear(d_model, d_model, bias=False)
            self.w_v_df = nn.Linear(d_model, d_model, bias=False)
            self.whole_merge_at = nn.Linear(
                self.n_heads * self.d_k, d_model, bias=False
            )
            self.whole_merge_df = nn.Linear(
                self.n_heads * self.d_k, d_model, bias=False
            )
            self.w_o_at = nn.Linear(d_model, d_model)
            self.w_o_df = nn.Linear(d_model, d_model)
        else:
            self.whole_merge = None
        self.w_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.use_rope = USE_ROPE
        self.rope_base = float(ROPE_BASE)
        self.use_debiased_qk_attention = bool(USE_DEBIASED_QK_ATTENTION)
        self.debiased_qk_use_abs = bool(DEBIASED_QK_USE_ABS)
        self.debiased_qk_momentum = float(DEBIASED_QK_MOMENTUM)
        self.debiased_qk_eps = float(DEBIASED_QK_EPS)
        # Storage for attention weights (for visualization)
        self.last_attention_weights: Optional[torch.Tensor] = None
        if self.use_debiased_qk_attention:
            stats_shape = (1, self.n_heads, 1, self.d_k)
            self.register_buffer(
                "debias_q_running_mean",
                torch.zeros(stats_shape, dtype=torch.float32),
            )
            self.register_buffer(
                "debias_q_running_std",
                torch.ones(stats_shape, dtype=torch.float32),
            )
            self.register_buffer(
                "debias_k_running_mean",
                torch.zeros(stats_shape, dtype=torch.float32),
            )
            self.register_buffer(
                "debias_k_running_std",
                torch.ones(stats_shape, dtype=torch.float32),
            )

    def _build_rope_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Build RoPE sin/cos cache with shape (1, 1, seq_len, d_k)
        dim = self.d_k
        if dim <= 0:
            raise ValueError("d_k must be > 0 for RoPE")
        dim_even = dim - (dim % 2)
        inv_freq = 1.0 / (
            self.rope_base
            ** (
                torch.arange(0, dim_even, 2, device=device, dtype=dtype)
                / dim_even
            )
        )
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        sin = freqs.sin()
        cos = freqs.cos()
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        if dim % 2 == 1:
            pad = torch.zeros((seq_len, 1), device=device, dtype=dtype)
            sin = torch.cat([sin, pad], dim=-1)
            cos = torch.cat([cos, pad + 1.0], dim=-1)
        return sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

    def _apply_rope(
        self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
    ) -> torch.Tensor:
        dim = x.size(-1)
        dim_even = dim - (dim % 2)
        x_even = x[..., :dim_even]
        x_odd = x[..., dim_even:] if dim_even < dim else None
        x1 = x_even[..., ::2]
        x2 = x_even[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x_even)
        out_even = x_even * cos[..., :dim_even] + x_rot * sin[..., :dim_even]
        if x_odd is None:
            return out_even
        return torch.cat([out_even, x_odd], dim=-1)

    def _qk_valid_token_mask(
        self, mask: Optional[torch.Tensor], x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if mask is None or not isinstance(mask, torch.Tensor):
            return None
        if mask.dim() == 4:
            valid = mask.any(dim=-2)
        elif mask.dim() == 3:
            valid = mask.any(dim=-2)
        elif mask.dim() == 2:
            valid = mask
        else:
            return None
        while valid.dim() > 2 and valid.size(1) == 1:
            valid = valid.squeeze(1)
        if valid.dim() != 2:
            return None
        if valid.size(0) != x.size(0) or valid.size(1) != x.size(2):
            return None
        return valid.to(device=x.device, dtype=torch.bool)

    def _masked_head_stats(
        self, x: torch.Tensor, valid_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_float = x.float()
        reduce_dims = (0, 2)
        if valid_mask is None:
            mean = x_float.mean(dim=reduce_dims, keepdim=True)
            centered = x_float - mean
            var = (centered * centered).mean(dim=reduce_dims, keepdim=True)
            std = torch.sqrt(var.clamp_min(0.0) + self.debiased_qk_eps)
            return mean, std

        weights = valid_mask[:, None, :, None].to(
            device=x.device, dtype=x_float.dtype
        )
        count = weights.sum(dim=reduce_dims, keepdim=True)
        safe_count = count.clamp_min(1.0)
        mean = (
            (x_float * weights).sum(dim=reduce_dims, keepdim=True)
            / safe_count
        )
        centered = (x_float - mean) * weights
        var = (
            (centered * centered).sum(dim=reduce_dims, keepdim=True)
            / safe_count
        )
        std = torch.sqrt(var.clamp_min(0.0) + self.debiased_qk_eps)
        has_tokens = count > 0.0
        mean = torch.where(has_tokens, mean, torch.zeros_like(mean))
        std = torch.where(has_tokens, std, torch.ones_like(std))
        return mean, std

    def _update_debiased_qk_stats(
        self,
        q_mean: torch.Tensor,
        q_std: torch.Tensor,
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
    ) -> None:
        momentum = self.debiased_qk_momentum
        with torch.no_grad():
            self.debias_q_running_mean.lerp_(
                q_mean.detach().to(dtype=self.debias_q_running_mean.dtype),
                momentum,
            )
            self.debias_q_running_std.lerp_(
                q_std.detach().to(dtype=self.debias_q_running_std.dtype),
                momentum,
            )
            self.debias_k_running_mean.lerp_(
                k_mean.detach().to(dtype=self.debias_k_running_mean.dtype),
                momentum,
            )
            self.debias_k_running_std.lerp_(
                k_std.detach().to(dtype=self.debias_k_running_std.dtype),
                momentum,
            )

    def _apply_debiased_qk_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_debiased_qk_attention:
            return Q, K

        valid_mask = self._qk_valid_token_mask(mask, Q)
        if self.training:
            q_mean, q_std = self._masked_head_stats(Q, valid_mask)
            k_mean, k_std = self._masked_head_stats(K, valid_mask)
            self._update_debiased_qk_stats(q_mean, q_std, k_mean, k_std)
        else:
            q_mean = self.debias_q_running_mean.float()
            q_std = self.debias_q_running_std.float()
            k_mean = self.debias_k_running_mean.float()
            k_std = self.debias_k_running_std.float()

        q_norm = (Q.float() - q_mean) / q_std.clamp_min(self.debiased_qk_eps)
        k_norm = (K.float() - k_mean) / k_std.clamp_min(self.debiased_qk_eps)
        if self.debiased_qk_use_abs:
            q_norm = q_norm.abs()
            k_norm = k_norm.abs()
        q_norm = torch.nan_to_num(q_norm, nan=0.0, posinf=0.0, neginf=0.0)
        k_norm = torch.nan_to_num(k_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return q_norm.to(dtype=Q.dtype), k_norm.to(dtype=K.dtype)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            Q: (batch_size, n_heads, seq_len, d_k)
            K: (batch_size, n_heads, seq_len, d_k)
            V: (batch_size, n_heads, seq_len, d_k)
            mask: (batch_size, 1, seq_len, seq_len) (TODO)
        """
        if hasattr(F, "scaled_dot_product_attention"):
            dropout_p = self.attn_dropout.p if self.training else 0.0
            sdpa_mask = None
            if mask is not None:
                sdpa_mask = mask.to(device=Q.device, dtype=torch.bool)
                if sdpa_mask.dim() == 4:
                    empty = ~sdpa_mask.any(dim=-1, keepdim=True)
                    if empty.any():
                        if sdpa_mask.is_contiguous():
                            sdpa_mask = sdpa_mask.clone()
                        else:
                            sdpa_mask = sdpa_mask.clone()
                        sdpa_mask[..., :1] = sdpa_mask[..., :1] | empty

            if dropout_p > 0.0 and STRICT_REPRODUCIBLE:
                # Flash/memory-efficient SDPA kernels with dropout are not fully
                # deterministic. Strict mode keeps the math backend; performance
                # mode allows PyTorch to choose faster fused kernels.
                _ctx = getattr(torch.nn.attention, "sdpa_kernel", None)
                if _ctx is not None:
                    from torch.nn.attention import SDPBackend

                    ctx = _ctx(SDPBackend.MATH)
                else:
                    ctx = torch.backends.cuda.sdp_kernel(
                        enable_flash=False,
                        enable_mem_efficient=False,
                        enable_math=True,
                    )
                with ctx:
                    output = F.scaled_dot_product_attention(
                        Q,
                        K,
                        V,
                        attn_mask=sdpa_mask,
                        dropout_p=dropout_p,
                        is_causal=False,
                    )
            else:
                output = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=sdpa_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )

            # Recompute attention weights for visualization (only in eval mode)
            with torch.no_grad():
                scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
                if mask is not None:
                    processed_mask = mask.to(device=scores.device, dtype=torch.bool)
                    if processed_mask.dim() == 4:
                        empty = ~processed_mask.any(dim=-1, keepdim=True)
                        if empty.any():
                            if processed_mask.is_contiguous():
                                processed_mask = processed_mask.clone()
                            else:
                                processed_mask = processed_mask.clone()
                            processed_mask[..., :1] = processed_mask[..., :1] | empty
                    if processed_mask.dim() == scores.dim():
                        fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
                        scores = scores.masked_fill(~processed_mask, fill_val)
                    elif processed_mask.dim() + 1 == scores.dim():
                        fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
                        scores = scores.masked_fill(~processed_mask.unsqueeze(1), fill_val)

                scores = scores.clamp(min=-1e4, max=1e4)
                attention_weights = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
                if ATTENTION_FINITE_CHECK and not torch.isfinite(attention_weights).all():
                    attention_weights = torch.nan_to_num(
                        attention_weights, nan=0.0, posinf=0.0, neginf=0.0
                    )
            return output, attention_weights

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.to(device=scores.device, dtype=torch.bool)
            if mask.dim() == 4:
                empty = ~mask.any(dim=-1, keepdim=True)
                if empty.any():
                    mask = mask.clone()
                    mask[..., :1] = mask[..., :1] | empty
            fill = (
                -1e4
                if scores.dtype in (torch.float16, torch.bfloat16)
                else -1e9
            )
            scores = scores.masked_fill(~mask, fill)

        scores = scores.clamp(min=-1e4, max=1e4)
        # Use fp32 softmax for numerical stability under AMP/bf16.
        attention_weights = F.softmax(scores.float(), dim=-1).to(
            dtype=scores.dtype
        )
        if ATTENTION_FINITE_CHECK and not torch.isfinite(attention_weights).all():
            attention_weights = torch.nan_to_num(
                attention_weights, nan=0.0, posinf=0.0, neginf=0.0
            )
        attention_weights = self.attn_dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def _forward_whole(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        value_modulators: Optional[torch.Tensor] = None,
        mod_index: int = 0,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = (
            self.w_q(query)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        # Use branch-specific V projection
        w_v_fn = self.w_v_at if mod_index == 0 else self.w_v_df
        V_projected = w_v_fn(value)
        # split_cv: store raw V projection for V-level auxiliary supervision
        if self.head_mode == "split_cv" and self.training:
            if mod_index == 0:
                self._last_v_at_raw = V_projected
            else:
                self._last_v_df_raw = V_projected
        V = V_projected.view(batch_size, -1, self.n_heads, self.d_k).transpose(
            1, 2
        )

        if self.use_rope:
            seq_len = Q.size(2)
            sin, cos = self._build_rope_cache(
                seq_len, device=Q.device, dtype=Q.dtype
            )
            Q = self._apply_rope(Q, sin, cos)
            K = self._apply_rope(K, sin, cos)

        Q, K = self._apply_debiased_qk_attention(Q, K, mask)

        if value_modulators is not None:
            seq_len = query.size(1)
            mods = value_modulators
            if mods.size(1) < seq_len:
                pad = seq_len - mods.size(1)
                pad_tensor = torch.full(
                    (mods.size(0), pad, 2),
                    0.5,
                    device=mods.device,
                    dtype=mods.dtype,
                )
                mods = torch.cat([mods, pad_tensor], dim=1)
            elif mods.size(1) > seq_len:
                mods = mods[:, :seq_len, :]

            mods = mods.to(device=V.device, dtype=V.dtype)
            scale = mods[:, :, mod_index].unsqueeze(1).unsqueeze(-1)
            V = V * scale
        elif self.training and self.head_mode == "split_rand_v":
            rand_vals = torch.rand(2, device=V.device, dtype=V.dtype)
            V = V * rand_vals[0] + rand_vals[1]

        attn, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        self.last_attention_weights = attn_weights
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_k)
        )
        # Use branch-specific merge and output projection
        merge_fn = (
            self.whole_merge_at if mod_index == 0 else self.whole_merge_df
        )
        combined = merge_fn(attn)
        w_o_fn = self.w_o_at if mod_index == 0 else self.w_o_df
        output = w_o_fn(combined)
        output = self.proj_dropout(output)
        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        value_modulators: Optional[torch.Tensor] = None,
    ):
        batch_size = query.size(0)

        Q = (
            self.w_q(query)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(value)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        if self.use_rope:
            seq_len = Q.size(2)
            sin, cos = self._build_rope_cache(
                seq_len, device=Q.device, dtype=Q.dtype
            )
            Q = self._apply_rope(Q, sin, cos)
            K = self._apply_rope(K, sin, cos)

        Q, K = self._apply_debiased_qk_attention(Q, K, mask)

        if self.use_split_heads:
            Q_at, Q_df = torch.split(
                Q, [self.n_heads_attack, self.n_heads_defense], dim=1
            )
            K_at, K_df = torch.split(
                K, [self.n_heads_attack, self.n_heads_defense], dim=1
            )
            V_at, V_df = torch.split(
                V, [self.n_heads_attack, self.n_heads_defense], dim=1
            )
        else:
            Q_at, K_at, V_at = Q, K, V
            Q_df = K_df = V_df = None

        if self.head_mode == "split_rand_v":
            V_at = torch.randn_like(V_at)
            if self.use_split_heads and V_df is not None:
                V_df = torch.randn_like(V_df)
        elif value_modulators is not None:
            seq_len = query.size(1)
            mods = value_modulators
            if mods.size(1) < seq_len:
                pad = seq_len - mods.size(1)
                pad_tensor = torch.full(
                    (mods.size(0), pad, 2),
                    0.5,
                    device=mods.device,
                    dtype=mods.dtype,
                )
                mods = torch.cat([mods, pad_tensor], dim=1)
            elif mods.size(1) > seq_len:
                mods = mods[:, :seq_len, :]

            mods = mods.to(device=V_at.device, dtype=V_at.dtype)
            v_at_scale = mods[:, :, 0].unsqueeze(1).unsqueeze(-1)
            V_at = V_at * v_at_scale
            if self.use_split_heads and V_df is not None:
                v_df_scale = mods[:, :, 1].unsqueeze(1).unsqueeze(-1)
                V_df = V_df * v_df_scale
        attn_at, attn_weights_at = self.scaled_dot_product_attention(Q_at, K_at, V_at, mask)
        self.last_attention_weights = attn_weights_at
        if self.use_split_heads and Q_df is not None:
            attn_df, attn_weights_df = self.scaled_dot_product_attention(
                Q_df, K_df, V_df, mask
            )

            attn_at = (
                attn_at.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, self.n_heads_attack * self.d_k)
            )
            attn_df = (
                attn_df.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, self.n_heads_defense * self.d_k)
            )

            combined_at = self.attack_merge(attn_at)
            combined_df = self.defense_merge(attn_df)
            if self.head_mode in ("split_keep", "split_cv", "split_cus_v"):
                out_at = self.proj_dropout(self.w_o_at(combined_at))
                out_df = self.proj_dropout(self.w_o_df(combined_df))
                return out_at, out_df
            combined = combined_at + combined_df
        else:
            attn_at = (
                attn_at.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, self.n_heads * self.d_k)
            )
            combined = self.attack_merge(attn_at)
        output = self.w_o(combined)
        output = self.proj_dropout(output)
        return output

    def forward_single_token_value_only(
        self, value: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the exact active value path for a length-1 sequence.

        With one token, softmax(QK^T) is identically one. Q and K therefore
        cannot affect the output or receive gradients. This method preserves
        the per-head attention dropout, value projection, merge projection,
        output projection, and projection dropout while skipping the inert
        Q/K and softmax computations.
        """
        if self.head_mode != "whole":
            raise ValueError(
                "single-token value_only ablation requires HEAD_MODE='whole'."
            )
        if value.dim() != 3 or value.size(1) != 1:
            raise ValueError(
                "single-token value_only ablation expects shape (B, 1, D); "
                f"got {tuple(value.shape)}."
            )

        batch_size = value.size(0)
        projected = (
            self.w_v(value)
            .view(batch_size, 1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        if self.training and self.attn_dropout.p > 0.0:
            head_keep = torch.ones(
                (batch_size, self.n_heads, 1, 1),
                device=projected.device,
                dtype=projected.dtype,
            )
            head_keep = F.dropout(
                head_keep,
                p=self.attn_dropout.p,
                training=True,
            )
            projected = projected * head_keep

        projected = (
            projected.transpose(1, 2)
            .contiguous()
            .view(batch_size, 1, self.n_heads * self.d_k)
        )
        output = self.w_o(self.attack_merge(projected))
        return self.proj_dropout(output)


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        single_token_ablation: str = "full",
        norm: str = "ln",
    ):
        """
        Encoder block consisting of multi-head self-attention and feed-forward.
        Args:
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(EncoderBlock, self).__init__()

        self.single_token_ablation = str(single_token_ablation).strip().lower()
        if self.single_token_ablation not in {
            "full",
            "value_only",
            "ffn_only",
        }:
            raise ValueError(
                "single_token_ablation must be one of "
                "{'full', 'value_only', 'ffn_only'}. "
                f"Got '{self.single_token_ablation}'."
            )

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.split_keep = self.attention.head_mode in (
            "split_keep",
            "split_cv",
            "split_cus_v",
        )
        self.normalization = norm
        self.norm1 = _make_sequence_norm(norm, d_model)
        self.norm2 = _make_sequence_norm(norm, d_model)
        self.residual_dropout = nn.Dropout(dropout)

        def _make_ff() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.Mish(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

        if self.split_keep:
            self.ln_attn_in_at = _make_sequence_norm(norm, d_model)
            self.ln_attn_out_at = _make_sequence_norm(norm, d_model)
            self.ln_ff_in_at = _make_sequence_norm(norm, d_model)
            self.ln_ff_out_at = _make_sequence_norm(norm, d_model)
            self.ln_attn_in_df = _make_sequence_norm(norm, d_model)
            self.ln_attn_out_df = _make_sequence_norm(norm, d_model)
            self.ln_ff_in_df = _make_sequence_norm(norm, d_model)
            self.ln_ff_out_df = _make_sequence_norm(norm, d_model)
            self.feed_forward_at = _make_ff()
            self.feed_forward_df = _make_ff()
        else:
            self.ln_attn_in = _make_sequence_norm(norm, d_model)
            self.ln_attn_out = _make_sequence_norm(norm, d_model)
            self.ln_ff_in = _make_sequence_norm(norm, d_model)
            self.ln_ff_out = _make_sequence_norm(norm, d_model)
            self.feed_forward = _make_ff()

        if self.single_token_ablation == "value_only":
            for projection in (self.attention.w_q, self.attention.w_k):
                for parameter in projection.parameters():
                    parameter.requires_grad_(False)
        elif self.single_token_ablation == "ffn_only":
            for parameter in self.attention.parameters():
                parameter.requires_grad_(False)
            if not self.split_keep:
                for parameter in self.ln_attn_in.parameters():
                    parameter.requires_grad_(False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        value_modulators: Optional[torch.Tensor] = None,
    ):
        # # Pre-LayerNorm
        # norm_x = self.norm1(x)
        # attention_output = self.attention(norm_x, norm_x, norm_x, mask)
        # x = x + self.dropout(attention_output)

        # # Pre-LayerNorm
        # norm_x = self.norm2(x)
        # ff_output = self.feed_forward(norm_x)
        # x = x + self.dropout(ff_output)

        # Peri-LayerNorm
        if self.split_keep:
            if isinstance(x, tuple):
                x_at, x_df = x
                attn_in_at = _apply_sequence_norm(
                    self.ln_attn_in_at, x_at, mask
                )
                attn_in_df = _apply_sequence_norm(
                    self.ln_attn_in_df, x_df, mask
                )
                attn_out_at = self.attention._forward_whole(
                    attn_in_at,
                    attn_in_at,
                    attn_in_at,
                    mask,
                    value_modulators=value_modulators,
                    mod_index=0,
                )
                attn_out_df = self.attention._forward_whole(
                    attn_in_df,
                    attn_in_df,
                    attn_in_df,
                    mask,
                    value_modulators=value_modulators,
                    mod_index=1,
                )
            else:
                x_at = x_df = x
                attn_in_at = _apply_sequence_norm(
                    self.ln_attn_in_at, x, mask
                )
                attn_in_df = _apply_sequence_norm(
                    self.ln_attn_in_df, x, mask
                )
                attn_out_at = self.attention._forward_whole(
                    attn_in_at,
                    attn_in_at,
                    attn_in_at,
                    mask,
                    value_modulators=value_modulators,
                    mod_index=0,
                )
                attn_out_df = self.attention._forward_whole(
                    attn_in_df,
                    attn_in_df,
                    attn_in_df,
                    mask,
                    value_modulators=value_modulators,
                    mod_index=1,
                )

            x_at = _apply_sequence_norm(
                self.ln_attn_out_at,
                x_at + self.residual_dropout(attn_out_at),
                mask,
            )
            x_df = _apply_sequence_norm(
                self.ln_attn_out_df,
                x_df + self.residual_dropout(attn_out_df),
                mask,
            )

            ff_in_at = _apply_sequence_norm(self.ln_ff_in_at, x_at, mask)
            ff_out_at = self.feed_forward_at(ff_in_at)
            x_at = _apply_sequence_norm(
                self.ln_ff_out_at,
                x_at + self.residual_dropout(ff_out_at),
                mask,
            )

            ff_in_df = _apply_sequence_norm(self.ln_ff_in_df, x_df, mask)
            ff_out_df = self.feed_forward_df(ff_in_df)
            x_df = _apply_sequence_norm(
                self.ln_ff_out_df,
                x_df + self.residual_dropout(ff_out_df),
                mask,
            )

            return x_at, x_df

        if self.single_token_ablation == "ffn_only":
            if x.dim() != 3 or x.size(1) != 1:
                raise ValueError(
                    "ffn_only ablation requires an encoder sequence length of 1; "
                    f"got {tuple(x.shape)}."
                )
        else:
            attn_in = _apply_sequence_norm(self.ln_attn_in, x, mask)
            if self.single_token_ablation == "value_only":
                attn_out = self.attention.forward_single_token_value_only(
                    attn_in
                )
            else:
                attn_out = self.attention(
                    attn_in,
                    attn_in,
                    attn_in,
                    mask,
                    value_modulators=value_modulators,
                )
            x = x + self.residual_dropout(attn_out)
        x = _apply_sequence_norm(self.ln_attn_out, x, mask)

        ff_in = _apply_sequence_norm(self.ln_ff_in, x, mask)
        ff_out = self.feed_forward(ff_in)
        x = x + self.residual_dropout(ff_out)
        x = _apply_sequence_norm(self.ln_ff_out, x, mask)

        return x


# ==============================================================================
# Conformer Components (Transformer + Depthwise Conv)
# ==============================================================================
class ConvModule(nn.Module):
    """Conformer-style convolution module.

    Default structure: LayerNorm → Pointwise Conv → GLU → Depthwise Conv →
    BatchNorm → Swish → Pointwise Conv → Dropout. Explicit normalization
    overrides replace both normalization sites consistently.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        expansion_factor: int = 2,
        norm: str = "hybrid",
    ):
        super().__init__()
        inner_dim = d_model * expansion_factor
        padding = (kernel_size - 1) // 2

        self.layer_norm = _make_sequence_norm(norm, d_model)

        # Pointwise expansion (×2 for GLU gating)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, inner_dim * 2, kernel_size=1, bias=False
        )
        self.glu = nn.GLU(dim=1)  # halves channels: inner_dim*2 → inner_dim

        # Depthwise separable conv — each channel has its own filter
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=inner_dim,
            bias=False,
        )
        self.batch_norm = _make_conv1d_norm(norm, inner_dim)
        self.activation = nn.SiLU()  # Swish

        # Pointwise projection back to d_model
        self.pointwise_conv2 = nn.Conv1d(
            inner_dim, d_model, kernel_size=1, bias=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        x = _apply_sequence_norm(self.layer_norm, x, attention_mask)
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        # Back to (B, T, C)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN(½) → MHSA → ConvModule → FFN(½) → norm.

    Following the "Macaron-Net" sandwich structure from the Conformer paper.
    The two feed-forward modules use half-step residual connections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
        conv_expansion_factor: int = 2,
        norm: str = "hybrid",
    ):
        super().__init__()

        # --- First half-step FFN ---
        self.normalization = norm
        self.ff1_norm = _make_sequence_norm(norm, d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # --- Multi-Head Self-Attention (reuse existing implementation) ---
        self.attn_norm = _make_sequence_norm(norm, d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # --- Conv Module ---
        self.conv_module = ConvModule(
            d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            expansion_factor=conv_expansion_factor,
            norm=norm,
        )

        # --- Second half-step FFN ---
        self.ff2_norm = _make_sequence_norm(norm, d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # --- Final normalization ---
        self.final_norm = _make_sequence_norm(norm, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        value_modulators: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1) Half-step FFN
        x = x + 0.5 * self.ff1(
            _apply_sequence_norm(self.ff1_norm, x, mask)
        )

        # 2) Multi-Head Self-Attention
        attn_in = _apply_sequence_norm(self.attn_norm, x, mask)
        attn_out = self.attention(
            attn_in, attn_in, attn_in, mask, value_modulators=value_modulators
        )
        x = x + self.attn_dropout(attn_out)

        # 3) Conv Module
        x = x + self.conv_module(x, attention_mask=mask)

        # 4) Half-step FFN
        x = x + 0.5 * self.ff2(
            _apply_sequence_norm(self.ff2_norm, x, mask)
        )

        # 5) Final normalization
        x = _apply_sequence_norm(self.final_norm, x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


__all__ = [name for name in globals() if not name.startswith("__")]
