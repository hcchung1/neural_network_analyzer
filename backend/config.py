"""CLI option registry and validated experiment configuration."""

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
import sys

"""============= Constants ==========="""

# Date string for file naming e.g. "20241218"
TODAY_DATE_STR = datetime.now().strftime("%Y%m%d")

"""==================================="""


def _parse_utils_option_overrides(argv: Sequence[str]) -> Dict[str, str]:
    """Parse repeatable CLI overrides in the form: --opt NAME=VALUE."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--opt",
        "--option",
        dest="options",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Override a utils.py option. Example: "
            "--opt ORACLE_GUIDING=true --opt CUR_MODEL=transOriginal"
        ),
    )
    parsed, _ = parser.parse_known_args(list(argv))

    overrides: Dict[str, str] = {}
    for entry in parsed.options:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --opt value '{entry}'. Expected NAME=VALUE."
            )
        name, raw_value = entry.split("=", 1)
        key = name.strip().upper()
        if not key:
            raise ValueError(
                f"Invalid --opt value '{entry}'. Option name is empty."
            )
        overrides[key] = raw_value.strip()
    return overrides


def _parse_bool_option(name: str, raw_value: str) -> bool:
    lowered = raw_value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value for option '{name}': '{raw_value}'."
    )


def _coerce_option_value(name: str, raw_value: str, default: Any) -> Any:
    if isinstance(default, bool):
        return _parse_bool_option(name, raw_value)
    if isinstance(default, int):
        return int(raw_value)
    if isinstance(default, float):
        return float(raw_value)
    if isinstance(default, str):
        return raw_value

    parsed_literal: Any = None
    literal_error: Optional[Exception] = None
    try:
        parsed_literal = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError) as exc:
        literal_error = exc

    if parsed_literal is None:
        try:
            parsed_literal = json.loads(raw_value)
        except json.JSONDecodeError:
            if literal_error is not None:
                raise ValueError(
                    f"Failed to parse option '{name}' from '{raw_value}'."
                ) from literal_error
            raise

    if isinstance(default, tuple):
        if not isinstance(parsed_literal, (list, tuple)):
            raise ValueError(
                f"Option '{name}' expects tuple-compatible value, got {type(parsed_literal).__name__}."
            )
        return tuple(parsed_literal)
    if isinstance(default, set):
        if not isinstance(parsed_literal, (list, tuple, set)):
            raise ValueError(
                f"Option '{name}' expects set-compatible value, got {type(parsed_literal).__name__}."
            )
        return set(parsed_literal)
    if isinstance(default, list):
        if not isinstance(parsed_literal, list):
            raise ValueError(
                f"Option '{name}' expects list value, got {type(parsed_literal).__name__}."
            )
        return parsed_literal
    if isinstance(default, dict):
        if not isinstance(parsed_literal, dict):
            raise ValueError(
                f"Option '{name}' expects dict value, got {type(parsed_literal).__name__}."
            )
        return parsed_literal

    return parsed_literal


_RAW_OPTION_OVERRIDES = _parse_utils_option_overrides(sys.argv[1:])
_SEEN_OPTION_NAMES: Set[str] = set()
_APPLIED_OPTION_NAMES: Set[str] = set()


def _option(name: str, default: Any) -> Any:
    key = name.upper()
    _SEEN_OPTION_NAMES.add(key)
    raw = _RAW_OPTION_OVERRIDES.get(key)
    if raw is None:
        return default
    value = _coerce_option_value(name, raw, default)
    _APPLIED_OPTION_NAMES.add(key)
    return value


"""============= Options ==========="""
# (TODO) write documentation for these options or parameters
DORA_FEATURE_SLOTS = _option("DORA_FEATURE_SLOTS", 5)
# Reduce worker count to avoid CPU oversubscription and context switching overhead
# Rule of thumb: leave 2-4 cores for main process, system, and GPU operations
NUM_WORKERS = _option("NUM_WORKERS", min(int(os.cpu_count() * 0.5), 16))
EVAL_NUM_WORKERS = _option(
    "EVAL_NUM_WORKERS", min(int(NUM_WORKERS * 0.75), 12)
)
# Final test/analysis is intentionally bounded independently from training.
# Training worker defaults remain unchanged for reproducibility.
FINAL_TEST_NUM_WORKERS = _option("FINAL_TEST_NUM_WORKERS", 8)
FINAL_TEST_PREFETCH_FACTOR = _option("FINAL_TEST_PREFETCH_FACTOR", 2)
PREDICTION_STREAM_CHUNK_SIZE = _option(
    "PREDICTION_STREAM_CHUNK_SIZE", 8192
)
INFO_LINE_CACHE_SIZE = _option("INFO_LINE_CACHE_SIZE", 256)
FINAL_TEST_RAM_BUDGET_GB = _option("FINAL_TEST_RAM_BUDGET_GB", 30.0)
if FINAL_TEST_NUM_WORKERS < 0:
    raise ValueError("FINAL_TEST_NUM_WORKERS must be >= 0.")
if FINAL_TEST_PREFETCH_FACTOR < 1:
    raise ValueError("FINAL_TEST_PREFETCH_FACTOR must be >= 1.")
if PREDICTION_STREAM_CHUNK_SIZE < 1:
    raise ValueError("PREDICTION_STREAM_CHUNK_SIZE must be >= 1.")
if INFO_LINE_CACHE_SIZE < 0:
    raise ValueError("INFO_LINE_CACHE_SIZE must be >= 0.")
if FINAL_TEST_RAM_BUDGET_GB <= 0.0:
    raise ValueError("FINAL_TEST_RAM_BUDGET_GB must be > 0.")

# Main training hyperparameters, exposed through the shared --opt registry.
BATCH_SIZE = _option("BATCH_SIZE", 64)
NUM_EPOCHS = _option("NUM_EPOCHS", 20)
LEARNING_RATE = _option("LEARNING_RATE", 1e-2)
EXPERIMENT_SEED = _option("EXPERIMENT_SEED", 42)
if BATCH_SIZE < 1:
    raise ValueError("BATCH_SIZE must be >= 1.")
if NUM_EPOCHS < 0:
    raise ValueError("NUM_EPOCHS must be >= 0.")
if LEARNING_RATE <= 0.0:
    raise ValueError("LEARNING_RATE must be > 0.")
if EXPERIMENT_SEED < 0:
    raise ValueError("EXPERIMENT_SEED must be >= 0.")

# Oracle guiding gradually removes leaked opponent hands during training.
# Set manually here (no environment variable override).
ORACLE_GUIDING = _option("ORACLE_GUIDING", False)
ORACLE_GUIDING_SHANTEN = _option("ORACLE_GUIDING_SHANTEN", False)
ORACLE_GUIDING_WAITS = _option("ORACLE_GUIDING_WAITS", False)
ORACLE_GUIDING_STEP_TENPAI = _option("ORACLE_GUIDING_STEP_TENPAI", False)
ORACLE_GUIDING_STEP_NEW_TENPAI = _option(
    "ORACLE_GUIDING_STEP_NEW_TENPAI", False
)
ORACLE_GUIDING_WARMUP_EPOCHS = _option("ORACLE_GUIDING_WARMUP_EPOCHS", 40)
# fraction of train steps per epoch that retain oracle view after warmup
ORACLE_GUIDING_PARTIAL_RATIO = _option("ORACLE_GUIDING_PARTIAL_RATIO", 0.0)
# when True, validation/test keep oracle rows visible (cheating upper bound)
ORACLE_GUIDING_EVAL_FORCE_REVEAL = _option(
    "ORACLE_GUIDING_EVAL_FORCE_REVEAL", False
)

# Embeddings
USE_TOKEN_TYPE_EMBEDDING = _option("USE_TOKEN_TYPE_EMBEDDING", False)
TOKEN_TYPE_EMBED_DIM = _option("TOKEN_TYPE_EMBED_DIM", 16)
USE_SEAT_ACTION_PHASE_EMBEDDING = _option(
    "USE_SEAT_ACTION_PHASE_EMBEDDING", False
)
SEAT_EMBED_DIM = _option("SEAT_EMBED_DIM", 16)
ACTION_EMBED_DIM = _option("ACTION_EMBED_DIM", 32)
PHASE_EMBED_DIM = _option("PHASE_EMBED_DIM", 8)
USE_ROPE = _option("USE_ROPE", False)
ROPE_BASE = _option("ROPE_BASE", 300.0)
USE_ABS_POS_ENCODING = _option("USE_ABS_POS_ENCODING", True)
if USE_ABS_POS_ENCODING and USE_ROPE:
    raise ValueError(
        "USE_ABS_POS_ENCODING and USE_ROPE are mutually exclusive positional encoding strategies. Enable at most one of them."
    )
# Debiasing Attention Mechanism step 1:
# q_de = abs((q - E[q]) / std(q)), k_de = abs((k - E[k]) / std(k)).
# Batch statistics are used in training, and running statistics are used in eval.
USE_DEBIASED_QK_ATTENTION = _option("USE_DEBIASED_QK_ATTENTION", False)
DEBIASED_QK_USE_ABS = _option("DEBIASED_QK_USE_ABS", True)
DEBIASED_QK_MOMENTUM = _option("DEBIASED_QK_MOMENTUM", 0.1)
DEBIASED_QK_EPS = _option("DEBIASED_QK_EPS", 1e-5)
if not 0.0 <= DEBIASED_QK_MOMENTUM <= 1.0:
    raise ValueError("DEBIASED_QK_MOMENTUM must be in [0, 1].")
if DEBIASED_QK_EPS <= 0.0:
    raise ValueError("DEBIASED_QK_EPS must be positive.")
# whole | split_custom_v | split_cus_v | split_rand_v | split_keep | split_cv
HEAD_MODE = _option("HEAD_MODE", "whole")
_HEAD_MODE_CHOICES = {
    "whole",
    "split_custom_v",
    "split_cus_v",
    "split_rand_v",
    "split_keep",
    "split_cv",
}
if HEAD_MODE not in _HEAD_MODE_CHOICES:
    raise ValueError(
        f"Unsupported HEAD_MODE='{HEAD_MODE}'. Valid options: {_HEAD_MODE_CHOICES}."
    )
# add | concat  (how to merge attack/defense branches in split_keep-family
# modes)
SPLIT_KEEP_MERGE_MODE = _option("SPLIT_KEEP_MERGE_MODE", "concat")
_SPLIT_KEEP_MERGE_CHOICES = {"add", "concat"}
if SPLIT_KEEP_MERGE_MODE not in _SPLIT_KEEP_MERGE_CHOICES:
    raise ValueError(
        f"Unsupported SPLIT_KEEP_MERGE_MODE='{SPLIT_KEEP_MERGE_MODE}'. Valid options: {_SPLIT_KEEP_MERGE_CHOICES}."
    )
USE_WARMUP_COSINE_SCHEDULER = _option("USE_WARMUP_COSINE_SCHEDULER", True)
WARMUP_HOLD_EPOCHS = _option(
    "WARMUP_HOLD_EPOCHS", 6
)  # epochs 1~6: keep base lr (1e-2 in main.py), epoch 7 starts cosine
WARMUP_EPOCHS = _option(
    "WARMUP_EPOCHS", 15
)  # epochs 8~15: cosine decay to min lr
COSINE_MIN_LR = _option("COSINE_MIN_LR", 1e-3)  # epochs 16+: hold at min lr
WARMUP_STRATEGY = _option(
    "WARMUP_STRATEGY", "hold_then_linear_down"
)  # linear_up | linear_down | hold_then_linear_down | hold_then_cosine_down
_WARMUP_STRATEGY_CHOICES = {
    "linear_up",
    "linear_down",
    "hold_then_linear_down",
    "hold_then_cosine_down",
}
if WARMUP_STRATEGY not in _WARMUP_STRATEGY_CHOICES:
    raise ValueError(
        f"Unsupported WARMUP_STRATEGY='{WARMUP_STRATEGY}'. Valid options: {_WARMUP_STRATEGY_CHOICES}."
    )
if WARMUP_HOLD_EPOCHS < 0:
    raise ValueError("WARMUP_HOLD_EPOCHS must be >= 0.")
if WARMUP_EPOCHS < 0:
    raise ValueError("WARMUP_EPOCHS must be >= 0.")
if COSINE_MIN_LR < 0.0:
    raise ValueError("COSINE_MIN_LR must be >= 0.")
# Explicit scheduler selection. The computed default preserves the legacy
# USE_WARMUP_COSINE_SCHEDULER switch for existing commands.
LR_SCHEDULER_MODE = str(
    _option(
        "LR_SCHEDULER_MODE",
        "warmup" if USE_WARMUP_COSINE_SCHEDULER else "plateau",
    )
).strip().lower()
_LR_SCHEDULER_MODE_CHOICES = {
    "warmup",
    "plateau",
    "warmup_then_plateau",
}
if LR_SCHEDULER_MODE not in _LR_SCHEDULER_MODE_CHOICES:
    raise ValueError(
        f"Unsupported LR_SCHEDULER_MODE='{LR_SCHEDULER_MODE}'. Valid options: "
        f"{sorted(_LR_SCHEDULER_MODE_CHOICES)}."
    )

PLATEAU_FACTOR = _option("PLATEAU_FACTOR", 0.5)
PLATEAU_PATIENCE = _option("PLATEAU_PATIENCE", 10)
PLATEAU_THRESHOLD = _option("PLATEAU_THRESHOLD", 1e-4)
PLATEAU_THRESHOLD_MODE = str(
    _option("PLATEAU_THRESHOLD_MODE", "rel")
).strip().lower()
PLATEAU_COOLDOWN = _option("PLATEAU_COOLDOWN", 0)
PLATEAU_MIN_LR = _option("PLATEAU_MIN_LR", 1e-5)
PLATEAU_EPS = _option("PLATEAU_EPS", 1e-8)

if not 0.0 < PLATEAU_FACTOR < 1.0:
    raise ValueError("PLATEAU_FACTOR must be between 0 and 1.")
if PLATEAU_PATIENCE < 0:
    raise ValueError("PLATEAU_PATIENCE must be >= 0.")
if PLATEAU_THRESHOLD < 0.0:
    raise ValueError("PLATEAU_THRESHOLD must be >= 0.")
if PLATEAU_THRESHOLD_MODE not in {"rel", "abs"}:
    raise ValueError("PLATEAU_THRESHOLD_MODE must be 'rel' or 'abs'.")
if PLATEAU_COOLDOWN < 0:
    raise ValueError("PLATEAU_COOLDOWN must be >= 0.")
if PLATEAU_MIN_LR < 0.0:
    raise ValueError("PLATEAU_MIN_LR must be >= 0.")
if PLATEAU_EPS < 0.0:
    raise ValueError("PLATEAU_EPS must be >= 0.")

if LR_SCHEDULER_MODE == "warmup_then_plateau":
    if WARMUP_STRATEGY != "hold_then_cosine_down":
        raise ValueError(
            "LR_SCHEDULER_MODE='warmup_then_plateau' requires "
            "WARMUP_STRATEGY='hold_then_cosine_down'."
        )
    if WARMUP_EPOCHS <= WARMUP_HOLD_EPOCHS:
        raise ValueError(
            "warmup_then_plateau requires WARMUP_EPOCHS > "
            "WARMUP_HOLD_EPOCHS so the cosine-down phase is non-empty."
        )
    if COSINE_MIN_LR >= LEARNING_RATE:
        raise ValueError(
            "warmup_then_plateau requires COSINE_MIN_LR < LEARNING_RATE."
        )
    if PLATEAU_MIN_LR >= COSINE_MIN_LR:
        raise ValueError(
            "warmup_then_plateau requires PLATEAU_MIN_LR < COSINE_MIN_LR "
            "so the plateau phase can reduce the learning rate further."
        )
USE_GRAD_CLIP = _option("USE_GRAD_CLIP", False)
GRAD_CLIP_NORM = _option("GRAD_CLIP_NORM", 10.0)
GRAD_FINITE_CHECK_EVERY_N_STEPS = _option(
    "GRAD_FINITE_CHECK_EVERY_N_STEPS", 0
)
ATTENTION_FINITE_CHECK = _option("ATTENTION_FINITE_CHECK", False)
USE_COMPACT_BINARY_TOKEN_PARSER = _option(
    "USE_COMPACT_BINARY_TOKEN_PARSER", True
)
VERIFY_COMPACT_BINARY_TOKEN_PARSER = _option(
    "VERIFY_COMPACT_BINARY_TOKEN_PARSER", False
)
if GRAD_FINITE_CHECK_EVERY_N_STEPS < 0:
    raise ValueError("GRAD_FINITE_CHECK_EVERY_N_STEPS must be >= 0.")
# Head mode after encoder:
# pooler | linear | first_token | last_token | flinear | multi_pool_flinear | cnn2d
#   pooler      : BERT-style Linear+Tanh+Dropout then classifier (default)
#   linear      : mean pool + single Linear(d_model, num_out)
#   first_token : take position-0 (init token) + Linear(d_model, num_out)
#   last_token  : take last valid token + Linear(d_model, num_out)
#   flinear     : flatten (B, seq_len*d_model) + Linear(seq_len*d_model, num_out)
#   multi_pool_flinear:
#                learn K query pools, flatten (B, K*d_model), then classify
# cnn2d       : treat (seq_len, d_model) as 2D feature map, apply Conv2d
# layers (ViT-style)
USE_SIMPLE_LINEAR_HEAD = _option("USE_SIMPLE_LINEAR_HEAD", "flinear")
OLD_SINGLE_ENCODER_ABLATION = str(
    _option("OLD_SINGLE_ENCODER_ABLATION", "full")
).strip().lower()
_OLD_SINGLE_ENCODER_ABLATION_CHOICES = {
    "full",
    "value_only",
    "ffn_only",
}
if OLD_SINGLE_ENCODER_ABLATION not in _OLD_SINGLE_ENCODER_ABLATION_CHOICES:
    raise ValueError(
        "Unsupported OLD_SINGLE_ENCODER_ABLATION="
        f"'{OLD_SINGLE_ENCODER_ABLATION}'. Valid options: "
        f"{sorted(_OLD_SINGLE_ENCODER_ABLATION_CHOICES)}."
    )
MULTI_POOL_QUERIES = _option("MULTI_POOL_QUERIES", 24)
MULTI_POOL_DROPOUT = _option("MULTI_POOL_DROPOUT", 0.1)
_SIMPLE_LINEAR_HEAD_CHOICES = {
    "pooler",
    "linear",
    "first_token",
    "last_token",
    "flinear",
    "multi_pool_flinear",
    "cnn2d",
}
if USE_SIMPLE_LINEAR_HEAD not in _SIMPLE_LINEAR_HEAD_CHOICES:
    raise ValueError(
        f"Unsupported USE_SIMPLE_LINEAR_HEAD='{USE_SIMPLE_LINEAR_HEAD}'. Valid options: {_SIMPLE_LINEAR_HEAD_CHOICES}."
    )
USE_CLS_TOKEN = _option("USE_CLS_TOKEN", False)
USE_LEARNABLE_POOL = _option("USE_LEARNABLE_POOL", False)
if USE_CLS_TOKEN and USE_LEARNABLE_POOL:
    raise ValueError(
        "USE_CLS_TOKEN and USE_LEARNABLE_POOL are mutually exclusive pooling strategies. "
        "Enable at most one of them."
    )
USE_ACTION_DANGER_VALUE_MODULATOR = _option(
    "USE_ACTION_DANGER_VALUE_MODULATOR", False
)
USE_FLAG_VALUE_MODULATOR = _option("USE_FLAG_VALUE_MODULATOR", False)
# symmetric -> [-logit, +logit], zero_pos -> [0, +logit]
TWO_LOGITS_MODE = _option("TWO_LOGITS_MODE", "symmetric")
_TWO_LOGITS_CHOICES = {"symmetric", "zero_pos"}
if TWO_LOGITS_MODE not in _TWO_LOGITS_CHOICES:
    raise ValueError(
        f"Unsupported TWO_LOGITS_MODE='{TWO_LOGITS_MODE}'. Valid options: {_TWO_LOGITS_CHOICES}."
    )
SPLIT_CV_LAMBDA_AT = _option(
    "SPLIT_CV_LAMBDA_AT",
    0.3  # auxiliary loss weight for attack branch in split_cv mode
)
SPLIT_CV_LAMBDA_DF = _option(
    "SPLIT_CV_LAMBDA_DF",
    0.3  # auxiliary loss weight for defense branch in split_cv mode
)
SPLIT_CV_LAMBDA_V = _option(
    "SPLIT_CV_LAMBDA_V",
    0.2  # V-projection level auxiliary loss weight in split_cv mode
)
# VOG (Variational Oracle Guiding): KL divergence weight between posterior
# z and prior z
VOG_KL_WEIGHT = _option("VOG_KL_WEIGHT", 1.0)

# ==============================================================================
# MTP (Masked Tile Prediction) Pre-training Configuration
# ==============================================================================
# When MTP_PRETRAINING is True, the model will be trained to predict masked tiles
# instead of the downstream classification task (tenpai prediction).
# This is similar to BERT's Masked Language Model (MLM) pre-training.
MTP_PRETRAINING = _option(
    "MTP_PRETRAINING",
    False  # Main switch: True for pre-training, False for fine-tuning
)

# Masking ratio: fraction of action tokens to mask (BERT uses 0.15)
MTP_MASK_RATIO = _option("MTP_MASK_RATIO", 0.15)

# Masking strategy:
#   "action_only"   : Only mask action tokens (position 1+), never mask init token
#   "tile_only"     : Only mask the tile portion of action tokens
MTP_MASK_STRATEGY = _option("MTP_MASK_STRATEGY", "action_only")
_MTP_MASK_STRATEGY_CHOICES = {"action_only", "tile_only"}
if MTP_MASK_STRATEGY not in _MTP_MASK_STRATEGY_CHOICES:
    raise ValueError(
        f"Unsupported MTP_MASK_STRATEGY='{MTP_MASK_STRATEGY}'. Valid options: {_MTP_MASK_STRATEGY_CHOICES}."
    )

# BERT-style replacement distribution for masked positions:
#   80% -> replace with [MASK] token (zero out tile features)
#   10% -> replace with random tile
#   10% -> keep original (model still needs to predict it)
MTP_REPLACE_WITH_MASK = _option("MTP_REPLACE_WITH_MASK", 0.8)
MTP_REPLACE_WITH_RANDOM = _option("MTP_REPLACE_WITH_RANDOM", 0.1)
MTP_KEEP_ORIGINAL = _option("MTP_KEEP_ORIGINAL", 0.1)
assert (
    abs(
        MTP_REPLACE_WITH_MASK
        + MTP_REPLACE_WITH_RANDOM
        + MTP_KEEP_ORIGINAL
        - 1.0
    )
    < 1e-6
), "MTP replacement ratios must sum to 1.0"

# Number of tile classes for prediction (34 tiles in Mahjong)
MTP_NUM_TILE_CLASSES = _option("MTP_NUM_TILE_CLASSES", 34)

# Minimum number of action tokens required for masking (skip if fewer)
MTP_MIN_ACTIONS = _option("MTP_MIN_ACTIONS", 3)

# Loss weight for MTP task (can be used for multi-task learning)
MTP_LOSS_WEIGHT = _option("MTP_LOSS_WEIGHT", 1.0)

# Whether to use auxiliary classification loss during pre-training
# (joint pre-training: MTP + classification, similar to BERT's NSP)
MTP_USE_AUX_CLASSIFICATION = _option("MTP_USE_AUX_CLASSIFICATION", False)
MTP_AUX_CLASSIFICATION_WEIGHT = _option(
    "MTP_AUX_CLASSIFICATION_WEIGHT", 0.1
)
# ==============================================================================

USE_NEW_ADD_INIT_EXTRA = _option("USE_NEW_ADD_INIT_EXTRA", False)
USE_NEW_ADD_INIT_OLD_SINGLE_RECENT = _option(
    "USE_NEW_ADD_INIT_OLD_SINGLE_RECENT", True
)
USE_NEW_ADD_FINAL_TOKEN_ONE = _option("USE_NEW_ADD_FINAL_TOKEN_ONE", False)
USE_NEW_ADD_FINAL_TOKEN_CONTEXT = _option(
    "USE_NEW_ADD_FINAL_TOKEN_CONTEXT", False
)
# When enabled (New_Add mode), each action token appends currently visible
# discard/meld set information for all 4 seats.
# Default False keeps legacy feature dimensions/checkpoint compatibility.
NEW_ADD_INCLUDE_VISIBLE_STATE_PER_ACTION = _option(
    "NEW_ADD_INCLUDE_VISIBLE_STATE_PER_ACTION", False
)

_OPP_HAND_ROWS_PER_OPP = 4
_OPPONENT_COUNT = 3
OPP_HAND_FEATURE_ROWS_BASE = (
    _OPP_HAND_ROWS_PER_OPP * _OPPONENT_COUNT
)  # 12 rows (3 opponents × 4 layers)
OPP_HAND_BYTES = 56  # serialized BinaryRecord::opp_hand_mask payload size
OPP_SHANTEN_BYTES = 3  # serialized BinaryRecord::opp_shanten payload size
OPP_WAIT_BYTES = 16  # serialized BinaryRecord::opp_wait_mask payload size
OPP_HAND_FEATURE_ROWS = OPP_HAND_FEATURE_ROWS_BASE if ORACLE_GUIDING else 0
OPP_WAIT_FEATURE_ROWS = _OPPONENT_COUNT if ORACLE_GUIDING_WAITS else 0
OPP_SHANTEN_FEATURE_ROWS = _OPPONENT_COUNT if ORACLE_GUIDING_SHANTEN else 0
# 0=disabled, 1=single one-hot column, 3=three-column block (early/mid/late)
TIME_SECTION_IN_METADATA = _option("TIME_SECTION_IN_METADATA", 3)
if TIME_SECTION_IN_METADATA not in (0, 1, 3):
    raise ValueError("TIME_SECTION_IN_METADATA must be one of {0, 1, 3}.")
TIME_SECTION_ROWS = (
    0 if TIME_SECTION_IN_METADATA == 0 else TIME_SECTION_IN_METADATA
)

# 控制前中後期特徵放置位置：metadata=放在INIT，action=放在每個 action 末端
TIME_SECTION_PLACEMENT = _option("TIME_SECTION_PLACEMENT", "action")
if TIME_SECTION_PLACEMENT not in ("metadata", "action"):
    raise ValueError(
        "TIME_SECTION_PLACEMENT must be one of {'metadata', 'action'}."
    )
TIME_SECTION_IN_ACTION = TIME_SECTION_PLACEMENT == "action"

# 巡目切分門檻 (早/中/晚)
TURN_MID_START = _option("TURN_MID_START", 24)
TURN_LATE_START = _option("TURN_LATE_START", 40)
PHASE_ID_EARLY = 0
PHASE_ID_MID = 1
PHASE_ID_LATE = 2
PHASE_ID_TO_NAME = {
    PHASE_ID_EARLY: "early",
    PHASE_ID_MID: "mid",
    PHASE_ID_LATE: "late",
}

# static layout: pred_id(3) + player_id(4) + round(8) + honba(1) +
# scores(4) + [phase(TIME_SECTION_ROWS)] + dora(5)
STATIC_FEATURE_ROWS_BASE = 3 + 4 + 8 + 1 + 4 + DORA_FEATURE_SLOTS
STATIC_PHASE_ROW_OFFSET = 3 + 4 + 8 + 1 + 4
STATIC_FEATURE_ROWS = STATIC_FEATURE_ROWS_BASE + (
    0 if TIME_SECTION_IN_ACTION else TIME_SECTION_ROWS
)
BITSET_ROWS_TOTAL = 308
TOTAL_FEATURE_ROWS = (
    STATIC_FEATURE_ROWS
    + BITSET_ROWS_TOTAL
    + OPP_HAND_FEATURE_ROWS
    + OPP_WAIT_FEATURE_ROWS
    + OPP_SHANTEN_FEATURE_ROWS
)
TOKEN_FEATURE_DIM = TOTAL_FEATURE_ROWS * 34
BITSET_OFFSET = STATIC_FEATURE_ROWS  # feature rows where board bitset begins
MY_DISCARD_ROWS = range(4, 4 + 30)
MY_TSUMOGIRI_ROWS = range(188, 188 + 30)
OPP_DISCARD_ROWS = [
    range(50, 50 + 30),  # 下家 (opp_idx = 0)
    range(80, 80 + 30),  # 對家 (opp_idx = 1)
    range(110, 110 + 30),  # 上家 (opp_idx = 2)
]
OPP_TSUMOGIRI_ROWS = [
    range(218, 218 + 30),  # 下家 (opp_idx = 0)
    range(248, 248 + 30),  # 對家 (opp_idx = 1)
    range(278, 278 + 30),  # 上家 (opp_idx = 2)
]
MY_HAND_ROWS = range(0, 0 + 4)
MY_MELD_ROWS = range(34, 34 + 16)
OPP_MELD_ROWS = [
    range(140, 140 + 16),
    range(156, 156 + 16),
    range(172, 172 + 16),
]
ORACLE_FEATURE_OFFSET = BITSET_OFFSET + BITSET_ROWS_TOTAL
ORACLE_FEATURE_ROWS = OPP_HAND_FEATURE_ROWS + OPP_WAIT_FEATURE_ROWS
ORACLE_FEATURE_FLAT_OFFSET = ORACLE_FEATURE_OFFSET * 34
ORACLE_FEATURE_FLAT_DIM = ORACLE_FEATURE_ROWS * 34
ORACLE_WAIT_FEATURE_OFFSET = ORACLE_FEATURE_OFFSET + OPP_HAND_FEATURE_ROWS
ORACLE_WAIT_FEATURE_ROWS = OPP_WAIT_FEATURE_ROWS
ORACLE_WAIT_FEATURE_FLAT_OFFSET = ORACLE_WAIT_FEATURE_OFFSET * 34
ORACLE_WAIT_FEATURE_FLAT_DIM = ORACLE_WAIT_FEATURE_ROWS * 34
ORACLE_SHANTEN_FEATURE_OFFSET = ORACLE_FEATURE_OFFSET + ORACLE_FEATURE_ROWS
ORACLE_SHANTEN_FEATURE_ROWS = OPP_SHANTEN_FEATURE_ROWS
ORACLE_SHANTEN_FEATURE_FLAT_OFFSET = ORACLE_SHANTEN_FEATURE_OFFSET * 34
ORACLE_SHANTEN_FEATURE_FLAT_DIM = ORACLE_SHANTEN_FEATURE_ROWS * 34

TEST_COPY = _option(
    "TEST_COPY", True
)  # 當 False 時，validation/test 階段跳過複製檔案 (label 2/3)
TRAIN_COPY = _option(
    "TRAIN_COPY", True
)  # 當 False 時，訓練階段會完全忽略複製檔案 (label >=2)
# Used only when num_epochs == 0 (Test only Mode) to load a specific
# checkpoint.
TEST_ONLY_MODEL_PATH = _option(
    "TEST_ONLY_MODEL_PATH",
    "output/transOriginal_TC_BF_SL_MG_FMT_New_Add_TP_last_n_actions32_REVACT_SRFP_LM_symmetric_BS64_NE30_ACT10_20260629_201953/transOriginal_TC_BF_SL_MG_FMT_New_Add_TP_last_n_actions32_REVACT_SRFP_LM_symmetric_BS64_NE30_ACT10_20260629_201953_all.pth",
)
# Resume training from a saved *_train_state.pth checkpoint.
RESUME_TRAINING = _option("RESUME_TRAINING", False)
# Optional explicit train-state checkpoint path. If empty and
# RESUME_TRAINING=True, main.py will auto-pick the latest
# *_train_state.pth under Transformer/output.
RESUME_TRAIN_STATE_PATH = _option("RESUME_TRAIN_STATE_PATH", "")
# Fine-tuning bootstrap: load model weights from a checkpoint before starting
# a fresh run (optimizer/scheduler/history are not restored).
FINETUNE_FROM_CHECKPOINT = _option("FINETUNE_FROM_CHECKPOINT", False)
# Accepts either:
# 1) plain model checkpoint (*.pth with state_dict), or
# 2) *_train_state.pth (main.py will extract phase model_state_dict).
FINETUNE_CHECKPOINT_PATH = _option("FINETUNE_CHECKPOINT_PATH", "")
# Keep False when transferring from pre-training (e.g., MTP) to downstream
# fine-tuning so task-specific heads can differ.
FINETUNE_LOAD_STRICT = _option("FINETUNE_LOAD_STRICT", False)
# True = map-style IndexedRoundDataset (deterministic order), False =
# IterRoundDataset (streaming)
USE_INDEXED_DATASET = _option("USE_INDEXED_DATASET", True)
USE_BINARY_FORMAT = _option(
    "USE_BINARY_FORMAT", True
)  # True to use binary data format, False for CSV
SOFTMAX_LOSS = (
    _option(
        "SOFTMAX_LOSS", True
    )  # If True, use CrossEntropyLoss, else use BCEWithLogitsLoss
)
# Early-turn (< N巡) positive-rate boosting switch (post-process thresholding)
EARLY_TURN_POSITIVE_RATE_BOOST = _option(
    "EARLY_TURN_POSITIVE_RATE_BOOST", False
)
EARLY_TURN_BOOST_MAX_TURN = _option("EARLY_TURN_BOOST_MAX_TURN", 15)
EARLY_TURN_MIN_POSITIVE_RATE = _option(
    "EARLY_TURN_MIN_POSITIVE_RATE", 0.20
)
EARLY_TURN_MIN_TENPAI_RATIO_IN_POSITIVES = _option(
    "EARLY_TURN_MIN_TENPAI_RATIO_IN_POSITIVES", 0.20
)
EARLY_TURN_BOOST_THRESHOLD_STEP = _option(
    "EARLY_TURN_BOOST_THRESHOLD_STEP", 0.01
)

if EARLY_TURN_BOOST_MAX_TURN < 1:
    raise ValueError("EARLY_TURN_BOOST_MAX_TURN must be >= 1.")
if not (0.0 <= EARLY_TURN_MIN_POSITIVE_RATE <= 1.0):
    raise ValueError("EARLY_TURN_MIN_POSITIVE_RATE must be in [0, 1].")
if not (0.0 <= EARLY_TURN_MIN_TENPAI_RATIO_IN_POSITIVES <= 1.0):
    raise ValueError(
        "EARLY_TURN_MIN_TENPAI_RATIO_IN_POSITIVES must be in [0, 1]."
    )
if not (0.0 < EARLY_TURN_BOOST_THRESHOLD_STEP <= 1.0):
    raise ValueError("EARLY_TURN_BOOST_THRESHOLD_STEP must be in (0, 1].")

# If True, apply per-sample loss reweighting by inverse P(label | seq_length):
#   w_i = 1 / P(y_i | length_i)
# Probability is estimated from each training mini-batch with Laplace smoothing.
USE_LENGTH_CONDITIONAL_REWEIGHT = _option(
    "USE_LENGTH_CONDITIONAL_REWEIGHT", False
)
LENGTH_CONDITIONAL_REWEIGHT_SMOOTHING = _option(
    "LENGTH_CONDITIONAL_REWEIGHT_SMOOTHING", 1.0
)
LENGTH_CONDITIONAL_REWEIGHT_CLIP_MIN = _option(
    "LENGTH_CONDITIONAL_REWEIGHT_CLIP_MIN", 0.25
)
LENGTH_CONDITIONAL_REWEIGHT_CLIP_MAX = _option(
    "LENGTH_CONDITIONAL_REWEIGHT_CLIP_MAX", 4.0
)
LENGTH_CONDITIONAL_REWEIGHT_NORMALIZE_MEAN = _option(
    "LENGTH_CONDITIONAL_REWEIGHT_NORMALIZE_MEAN", False
)
# Length-bias regularization:
# Penalize correlation between model score and sequence length.
# This targets shortcut-learning on seq_len directly.
USE_LENGTH_CORR_PENALTY = _option("USE_LENGTH_CORR_PENALTY", False)
LENGTH_CORR_PENALTY_WEIGHT = _option("LENGTH_CORR_PENALTY_WEIGHT", 0.05)
LENGTH_CORR_PENALTY_EPS = _option("LENGTH_CORR_PENALTY_EPS", 1e-8)
# Length adversarial training:
# A length predictor is trained from the classifier representation through a
# Gradient Reversal Layer, so the predictor learns normally while the encoder
# receives the negated length-prediction gradient.
USE_LENGTH_GRL = _option("USE_LENGTH_GRL", False)
LENGTH_GRL_WEIGHT = _option("LENGTH_GRL_WEIGHT", 0.05)
LENGTH_GRL_LAMBDA = _option("LENGTH_GRL_LAMBDA", 1.0)
LENGTH_PREDICTOR_BUCKET_SIZE = _option("LENGTH_PREDICTOR_BUCKET_SIZE", 1)
LENGTH_PREDICTOR_HIDDEN_DIM = _option("LENGTH_PREDICTOR_HIDDEN_DIM", 0)
LENGTH_PREDICTOR_DROPOUT = _option("LENGTH_PREDICTOR_DROPOUT", 0.1)
USE_IMPORTANCE_SAMPLING = _option("USE_IMPORTANCE_SAMPLING", False)
# Whether to predict shanten classes instead of 2-class tenpai.
PREDICT_SHANTEN = _option("PREDICT_SHANTEN", False)
SHANTEN_MAX_CLASS = _option("SHANTEN_MAX_CLASS", 4)
if SHANTEN_MAX_CLASS < 1:
    raise ValueError(
        f"SHANTEN_MAX_CLASS must be >= 1. Got {SHANTEN_MAX_CLASS}."
    )
SHANTEN_NUM_CLASSES = SHANTEN_MAX_CLASS + 1

def shanten_to_class(
    shanten_value: int, max_class: Optional[int] = None
) -> int:
    """Map raw shanten to classes 0..max-1 plus max for >=max/unknown."""
    class_cap = SHANTEN_MAX_CLASS if max_class is None else int(max_class)
    if class_cap < 1:
        raise ValueError(f"max_class must be >= 1. Got {class_cap}.")
    raw_value = int(shanten_value)
    if raw_value < 0:
        return class_cap
    return min(max(raw_value, 0), class_cap)

IMPORTANCE_SAMPLING_MODE = str(
    _option("IMPORTANCE_SAMPLING_MODE", "turn")
).strip().lower().replace("-", "_").replace(" ", "_")
# When True, training-time loss is multiplied by p(i)/q(i) after
# importance-sampling resampling, so the expectation stays aligned with the
# original empirical distribution.
USE_IMPORTANCE_SAMPLING_LOSS_CORRECTION = _option(
    "USE_IMPORTANCE_SAMPLING_LOSS_CORRECTION", True
)
IMPORTANCE_SAMPLING_MODE_CHOICES = {
    "turn",
    "turn_bin_10",
}
if IMPORTANCE_SAMPLING_MODE not in IMPORTANCE_SAMPLING_MODE_CHOICES:
    raise ValueError(
        "IMPORTANCE_SAMPLING_MODE must be one of "
        f"{sorted(IMPORTANCE_SAMPLING_MODE_CHOICES)}. "
        f"Got '{IMPORTANCE_SAMPLING_MODE}'."
    )
if LENGTH_CORR_PENALTY_WEIGHT < 0:
    raise ValueError(
        "LENGTH_CORR_PENALTY_WEIGHT must be >= 0. "
        f"Got {LENGTH_CORR_PENALTY_WEIGHT}."
    )
if LENGTH_CORR_PENALTY_EPS <= 0:
    raise ValueError(
        "LENGTH_CORR_PENALTY_EPS must be > 0. "
        f"Got {LENGTH_CORR_PENALTY_EPS}."
    )
if LENGTH_GRL_WEIGHT < 0:
    raise ValueError(
        "LENGTH_GRL_WEIGHT must be >= 0. "
        f"Got {LENGTH_GRL_WEIGHT}."
    )
if LENGTH_GRL_LAMBDA < 0:
    raise ValueError(
        "LENGTH_GRL_LAMBDA must be >= 0. "
        f"Got {LENGTH_GRL_LAMBDA}."
    )
if LENGTH_PREDICTOR_BUCKET_SIZE < 1:
    raise ValueError(
        "LENGTH_PREDICTOR_BUCKET_SIZE must be >= 1. "
        f"Got {LENGTH_PREDICTOR_BUCKET_SIZE}."
    )
if LENGTH_PREDICTOR_HIDDEN_DIM < 0:
    raise ValueError(
        "LENGTH_PREDICTOR_HIDDEN_DIM must be >= 0. "
        f"Got {LENGTH_PREDICTOR_HIDDEN_DIM}."
    )
if not (0.0 <= LENGTH_PREDICTOR_DROPOUT < 1.0):
    raise ValueError(
        "LENGTH_PREDICTOR_DROPOUT must be in [0, 1). "
        f"Got {LENGTH_PREDICTOR_DROPOUT}."
    )
if MULTI_POOL_QUERIES < 1:
    raise ValueError(
        "MULTI_POOL_QUERIES must be >= 1. "
        f"Got {MULTI_POOL_QUERIES}."
    )
if not (0.0 <= MULTI_POOL_DROPOUT < 1.0):
    raise ValueError(
        "MULTI_POOL_DROPOUT must be in [0, 1). "
        f"Got {MULTI_POOL_DROPOUT}."
    )
USE_MULTI_GPU = _option("USE_MULTI_GPU", True)  # True for multi-GPU training
# Strict reproducibility mode (paper-grade):
# - force deterministic kernels (error on nondeterministic ops)
# - disable mixed precision / TF32 paths
# - fix GPU/device ordering and dataloader workers
STRICT_REPRODUCIBLE = _option("STRICT_REPRODUCIBLE", True)
# Optional worker override when STRICT_REPRODUCIBLE=True.
# Keep defaults at 0 for the most conservative reproducibility baseline.
STRICT_REPRO_NUM_WORKERS = _option("STRICT_REPRO_NUM_WORKERS", int(os.cpu_count() * 0.9))
STRICT_REPRO_EVAL_NUM_WORKERS = _option("STRICT_REPRO_EVAL_NUM_WORKERS", int(STRICT_REPRO_NUM_WORKERS * 0.9))
# Optional deterministic speed path for paper-grade runs. Disabled by default
# so historical runs keep their exact data flow unless explicitly requested.
STRICT_REPRO_FAST_PIPELINE = _option("STRICT_REPRO_FAST_PIPELINE", False)
STRICT_REPRO_FAST_DIRECT_ALL_PHASE = _option(
    "STRICT_REPRO_FAST_DIRECT_ALL_PHASE", True
)
STRICT_REPRO_FAST_GROUPED_SAMPLER = _option(
    "STRICT_REPRO_FAST_GROUPED_SAMPLER", False
)
STRICT_REPRO_FAST_RECORD_CACHE_SIZE = _option(
    "STRICT_REPRO_FAST_RECORD_CACHE_SIZE", 1
)
STRICT_REPRO_FAST_PREFETCH_FACTOR = _option(
    "STRICT_REPRO_FAST_PREFETCH_FACTOR", 4
)
if STRICT_REPRO_FAST_RECORD_CACHE_SIZE < 0:
    raise ValueError("STRICT_REPRO_FAST_RECORD_CACHE_SIZE must be >= 0.")
if STRICT_REPRO_FAST_PREFETCH_FACTOR < 2:
    raise ValueError("STRICT_REPRO_FAST_PREFETCH_FACTOR must be >= 2.")
# Discard order mode: "original", "reverse", "recent_first", or "recent_reverse"
# - "original": Keep chronological order (earliest discard first)
# - "reverse": Reverse order (most recent discard first)
# - "recent_first": Put most recent 6 discards first, then the rest (e.g., 10 tiles: [5,6,7,8,9,10,1,2,3,4])
# - "recent_reverse": Put most recent 6 first in reverse order, then the rest (e.g., [10,9,8,7,6,5,1,2,3,4])
# "original" | "reverse" | "recent_first" | "recent_reverse"
DISCARD_ORDER_MODE = _option("DISCARD_ORDER_MODE", "reverse")
# Deprecated compatibility alias used by ct/fast_dataset.py.  The C++ parser
# exposes only a boolean reverse switch, so only the equivalent full-reverse
# mode maps to True; richer ordering modes remain Python-pipeline features.
REVERSE_DISCARD = str(DISCARD_ORDER_MODE).strip().lower() == "reverse"
RANDOM_FEATURE_VALIDATE = _option("RANDOM_FEATURE_VALIDATE", True)
RANDOM_FEATURE_TRAIN = _option("RANDOM_FEATURE_TRAIN", False)
RANDOM_FEATURE_WARNING_MIN_DELTA = _option(
    "RANDOM_FEATURE_WARNING_MIN_DELTA", 0.02
)
RANDOM_FEATURE_SEED = _option("RANDOM_FEATURE_SEED", 42)
RANDOM_FEATURE_DISTRIBUTION = _option("RANDOM_FEATURE_DISTRIBUTION", "normal")
RANDOM_FEATURE_SCALE = _option("RANDOM_FEATURE_SCALE", 1.0)
RANDOM_FEATURE_REPLACEMENT_SCOPE = _option(
    "RANDOM_FEATURE_REPLACEMENT_SCOPE", "all_features"
)
_RANDOM_FEATURE_REPLACEMENT_SCOPE_CHOICES = {
    "all_features",
    "new_add_action_tokens",
}
if (
    RANDOM_FEATURE_REPLACEMENT_SCOPE
    not in _RANDOM_FEATURE_REPLACEMENT_SCOPE_CHOICES
):
    raise ValueError(
        "RANDOM_FEATURE_REPLACEMENT_SCOPE must be one of "
        f"{sorted(_RANDOM_FEATURE_REPLACEMENT_SCOPE_CHOICES)}."
    )
RANDOM_FEATURE_PRESERVE_PADDING = _option(
    "RANDOM_FEATURE_PRESERVE_PADDING", True
)
RANDOM_FEATURE_MASK_OLD_SINGLE_DISCARDS_BY_TURN = _option(
    "RANDOM_FEATURE_MASK_OLD_SINGLE_DISCARDS_BY_TURN", True
)
RANDOM_FEATURE_TEST_ANALYSIS = _option("RANDOM_FEATURE_TEST_ANALYSIS", True)
RANDOM_FEATURE_TEST_ANALYSIS_MAX_SAMPLES = _option(
    "RANDOM_FEATURE_TEST_ANALYSIS_MAX_SAMPLES", 0
)
RANDOM_FEATURE_TEST_ANALYSIS_META_CSV = _option(
    "RANDOM_FEATURE_TEST_ANALYSIS_META_CSV", ""
)
RANDOM_FEATURE_TEST_ANALYSIS_META_CSV = _option(
    "RANDOM_FEATURE_TURN_METADATA_CSV",
    RANDOM_FEATURE_TEST_ANALYSIS_META_CSV,
)
RANDOM_FEATURE_TEST_ANALYSIS_META_CSV = _option(
    "REBUILT_TURN_METADATA_CSV",
    RANDOM_FEATURE_TEST_ANALYSIS_META_CSV,
)
if int(RANDOM_FEATURE_TEST_ANALYSIS_MAX_SAMPLES) < 0:
    raise ValueError("RANDOM_FEATURE_TEST_ANALYSIS_MAX_SAMPLES must be >= 0.")
RANDOM_FEATURE_TEST_ANALYSIS_MAX_SAMPLES = int(
    RANDOM_FEATURE_TEST_ANALYSIS_MAX_SAMPLES
)
if RANDOM_FEATURE_DISTRIBUTION not in {"normal", "uniform"}:
    raise ValueError(
        "RANDOM_FEATURE_DISTRIBUTION must be one of {'normal', 'uniform'}."
    )
if RANDOM_FEATURE_WARNING_MIN_DELTA < 0.0:
    raise ValueError("RANDOM_FEATURE_WARNING_MIN_DELTA must be >= 0.")
if RANDOM_FEATURE_SCALE <= 0.0:
    raise ValueError("RANDOM_FEATURE_SCALE must be > 0.")
# Token placement mode after packing/padding:
# - "contiguous": keep valid tokens packed at the front.
# - "even": keep init at 0, place the last valid token at the last sequence
#   slot, and spread the remaining valid tokens as evenly as possible.
# - "head_tail_even": keep init at 0; for action tokens, place a single
#   action at the last slot, or place action head at 1, tail at the last slot,
#   and spread the middle actions evenly.
# - "last_six_tail": keep init at 0, place the last six actions in the last
#   six slots, and keep earlier actions contiguous after init.
# - "last_n_actions": keep only the last N actions packed after init. The
#   init token source is controlled by TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE.
_RAW_TOKEN_PLACEMENT_MODE = _option("TOKEN_PLACEMENT_MODE", "last_n_actions")
_TOKEN_PLACEMENT_MODE_ALIASES = {
    "last_24_actions": "last_n_actions",
    "last_n_action": "last_n_actions",
}
TOKEN_PLACEMENT_MODE = _TOKEN_PLACEMENT_MODE_ALIASES.get(
    _RAW_TOKEN_PLACEMENT_MODE, _RAW_TOKEN_PLACEMENT_MODE
)
TOKEN_PLACEMENT_LAST_ACTION_COUNT = _option(
    "TOKEN_PLACEMENT_LAST_ACTION_COUNT", 24
)
TOKEN_PLACEMENT_LAST_ACTION_COUNT = _option(
    "LAST_N_ACTION_COUNT", TOKEN_PLACEMENT_LAST_ACTION_COUNT
)
if TOKEN_PLACEMENT_LAST_ACTION_COUNT < 1:
    raise ValueError(
        "LAST_N_ACTION_COUNT/TOKEN_PLACEMENT_LAST_ACTION_COUNT must be >= 1."
    )

TOKEN_PLACEMENT_LAST_ACTION_TENSOR_LENGTH = _option(
    "TOKEN_PLACEMENT_LAST_ACTION_TENSOR_LENGTH", 0
)
TOKEN_PLACEMENT_LAST_ACTION_TENSOR_LENGTH = _option(
    "LAST_N_ACTION_TENSOR_LENGTH",
    TOKEN_PLACEMENT_LAST_ACTION_TENSOR_LENGTH,
)
if int(TOKEN_PLACEMENT_LAST_ACTION_TENSOR_LENGTH) < 0:
    raise ValueError(
        "LAST_N_ACTION_TENSOR_LENGTH/TOKEN_PLACEMENT_LAST_ACTION_TENSOR_LENGTH must be >= 0. "
        "Use 0 to preserve the default N+1 tensor length."
    )
TOKEN_PLACEMENT_LAST_ACTION_TENSOR_LENGTH = int(
    TOKEN_PLACEMENT_LAST_ACTION_TENSOR_LENGTH
)
TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE = _option(
    "TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE", "window_start"
)
TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE = _option(
    "LAST_N_ACTION_INIT_STATE",
    TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE,
)
# "window_start" preserves the pre-window snapshot; "final" uses states[-1].
TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE = (
    str(TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE)
    .strip()
    .lower()
    .replace("-", "_")
)
_TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE_CHOICES = {
    "window_start",
    "final",
}
if (
    TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE
    not in _TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE_CHOICES
):
    raise ValueError(
        "Unsupported TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE="
        f"'{TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE}'. Valid options: "
        f"{sorted(_TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE_CHOICES)}."
    )
# New_Add only: reverse the action-token segment while leaving init,
# final/context tokens, and padding in their existing positions.
REVERSE_NEW_ADD_ACTION_TOKENS = _option(
    "REVERSE_NEW_ADD_ACTION_TOKENS", False
)
_TOKEN_PLACEMENT_MODE_CHOICES = {
    "contiguous",
    "even",
    "head_tail_even",
    "last_six_tail",
    "last_n_actions",
}
if TOKEN_PLACEMENT_MODE not in _TOKEN_PLACEMENT_MODE_CHOICES:
    raise ValueError(
        f"Unsupported TOKEN_PLACEMENT_MODE='{_RAW_TOKEN_PLACEMENT_MODE}'. "
        f"Valid options: {_TOKEN_PLACEMENT_MODE_CHOICES}."
    )
# Original: [1,2,3,4,5,6,7,8,9,10]
# Reverse: [10,9,8,7,6,5,4,3,2,1]
# Recent first (for 10 discards): [5,6,7,8,9,10,1,2,3,4] (most recent 6 first, then the rest in original order)
# Recent reverse (for 10 discards): [10,9,8,7,6,5,1,2,3,4] (most recent 6
# first in reverse, then the rest in original order)
RESNET_MODEL_POOL = _option(
    "RESNET_MODEL_POOL",
    [
        "resnet10",
        "resnet14",
        "resnet18",
        "resnet22",
    ],
)
CNN_MODEL_POOL = _option(
    "CNN_MODEL_POOL",
    [
        "cnn",
    ],
)
CONFORMER_MODEL_POOL = _option(
    "CONFORMER_MODEL_POOL",
    [
        "conformerTest",
        "conformerSmall",
        "conformerOriginal",
        "conformerLarge",
    ],
)
VOG_MODEL_POOL = _option("VOG_MODEL_POOL", ["transVOG"])
MODEL_POOL = [
    "trans", # 14M
    "transBertBase", # 93M
    "transBertLarge", # 328M
    "transOriginal", #21M
    "transTest",
    *RESNET_MODEL_POOL,
    *CNN_MODEL_POOL,
    *CONFORMER_MODEL_POOL,
    *VOG_MODEL_POOL,
]
CUR_MODEL = _option("CUR_MODEL", "transOriginal")
if CUR_MODEL not in MODEL_POOL:
    raise ValueError(
        f"Unsupported CUR_MODEL='{CUR_MODEL}'. Valid options: {MODEL_POOL}."
    )

# One normalization option is shared by every model family. ``auto`` preserves
# each architecture's historical default instead of forcing one normalization
# style across unrelated model families.
_MODEL_NORMALIZATION_REQUESTED = str(
    _option("MODEL_NORMALIZATION", "auto")
).strip().lower()
_LEGACY_RESNET_NORM = str(_option("RESNET_NORM", "")).strip().lower()
if _LEGACY_RESNET_NORM:
    if "MODEL_NORMALIZATION" in _RAW_OPTION_OVERRIDES:
        raise ValueError(
            "Specify only MODEL_NORMALIZATION; RESNET_NORM is a deprecated "
            "compatibility alias and cannot be combined with it."
        )
    if CUR_MODEL not in RESNET_MODEL_POOL:
        raise ValueError(
            "RESNET_NORM is only a deprecated alias for ResNet runs. Use "
            "MODEL_NORMALIZATION for other model families."
        )
    _MODEL_NORMALIZATION_REQUESTED = _LEGACY_RESNET_NORM
    logger.warning(
        "RESNET_NORM is deprecated; use MODEL_NORMALIZATION instead."
    )

_MODEL_NORMALIZATION_CHOICES = {"auto", "bn", "gn", "ln"}
if _MODEL_NORMALIZATION_REQUESTED not in _MODEL_NORMALIZATION_CHOICES:
    raise ValueError(
        "Unsupported MODEL_NORMALIZATION="
        f"'{_MODEL_NORMALIZATION_REQUESTED}'. Valid options: "
        f"{sorted(_MODEL_NORMALIZATION_CHOICES)}."
    )


def resolve_model_normalization(model_name: str, requested: str) -> str:
    """Resolve a shared normalization request to a model-family default."""
    requested = str(requested).strip().lower()
    if requested != "auto":
        return requested
    if model_name in RESNET_MODEL_POOL:
        return "gn"
    if model_name in CNN_MODEL_POOL:
        return "bn"
    if model_name in CONFORMER_MODEL_POOL:
        # Preserve Conformer's LayerNorm-around-blocks + BatchNorm-in-conv
        # architecture when no explicit override is requested.
        return "hybrid"
    # Transformer and VOG encoder blocks historically use LayerNorm.
    return "ln"


MODEL_NORMALIZATION_REQUESTED = _MODEL_NORMALIZATION_REQUESTED
MODEL_NORMALIZATION = resolve_model_normalization(
    CUR_MODEL, MODEL_NORMALIZATION_REQUESTED
)

# Phase-specific training controls
TRAIN_PHASE_FLAGS = _option(
    "TRAIN_PHASE_FLAGS",
    {
        "all": True,  # combined model
        "early": False,
        "mid": False,
        "late": False,
    },
)

# Buffer settings for balanced phase training
PHASE_BUFFER_CAPACITY = _option(
    "PHASE_BUFFER_CAPACITY", 50000
)  # cap per phase per flush to avoid RAM blowup
# flush when min phase buffer reaches this fraction of capacity
PHASE_BUFFER_FLUSH_RATIO = _option("PHASE_BUFFER_FLUSH_RATIO", 0.25)
PHASE_BUFFER_HARD_LIMIT = _option(
    "PHASE_BUFFER_HARD_LIMIT",
    100000  # drop oldest if a phase buffers more than this
)
PHASE_BUFFER_BALANCE = _option(
    "PHASE_BUFFER_BALANCE",
    "min"  # strategy to equalize phases; currently only 'min'
)
PHASE_FILE_SAMPLE_RATIO = _option(
    "PHASE_FILE_SAMPLE_RATIO", 0.2
)  # fraction of train files sampled per epoch
BALANCE_LABEL_BATCH = _option(
    "BALANCE_LABEL_BATCH", False
)  # when True, use weighted sampling to balance label ratio in each epoch
BALANCE_LABEL_TARGET_POS_RATIO = _option(
    "BALANCE_LABEL_TARGET_POS_RATIO", 0.5
)  # expected positive sampling ratio when BALANCE_LABEL_BATCH=True
if not (0.0 < BALANCE_LABEL_TARGET_POS_RATIO < 1.0):
    raise ValueError(
        "BALANCE_LABEL_TARGET_POS_RATIO must be in (0, 1). "
        f"Got {BALANCE_LABEL_TARGET_POS_RATIO}."
    )

# Length perturbation augmentation (train-only):
# - random cropping
# - random padding with masked dummy tokens
# - token dropout
LENGTH_PERTURBATION = False
LENGTH_PERTURBATION_PROB = 0.50
LENGTH_PERTURB_RANDOM_CROP = False
LENGTH_PERTURB_RANDOM_CROP_PROB = 0.60
LENGTH_PERTURB_MIN_ACTION_KEEP = 5
LENGTH_PERTURB_TOKEN_DROPOUT = False
LENGTH_PERTURB_TOKEN_DROPOUT_PROB = 0.50
LENGTH_PERTURB_TOKEN_DROPOUT_RATE = 0.08
LENGTH_PERTURB_RANDOM_PADDING = False
LENGTH_PERTURB_RANDOM_PADDING_PROB = 0.70

# Match per-turn label ratio to perturbation distribution (train-only).
TURN_LABEL_PERTURB_MATCH = _option("TURN_LABEL_PERTURB_MATCH", False)
TURN_LABEL_PERTURB_MATCH_SEED = _option("TURN_LABEL_PERTURB_MATCH_SEED", 42)
TURN_LABEL_PERTURB_MATCH_MIN_TOTAL = _option(
    "TURN_LABEL_PERTURB_MATCH_MIN_TOTAL", 20
)

# Turn/label statistics output controls
TURN_LABEL_STATS_EVERY_N_EPOCHS = _option("TURN_LABEL_STATS_EVERY_N_EPOCHS", 5)
TURN_LABEL_STATS_SAVE_PLOT = _option("TURN_LABEL_STATS_SAVE_PLOT", False)
TURN_LABEL_STATS_SAVE_CSV = _option("TURN_LABEL_STATS_SAVE_CSV", False)

if not (0.0 <= LENGTH_PERTURBATION_PROB <= 1.0):
    raise ValueError("LENGTH_PERTURBATION_PROB must be in [0, 1].")
if not (0.0 <= LENGTH_PERTURB_TOKEN_DROPOUT_RATE < 1.0):
    raise ValueError("LENGTH_PERTURB_TOKEN_DROPOUT_RATE must be in [0, 1).")
if LENGTH_PERTURB_MIN_ACTION_KEEP < 1:
    raise ValueError("LENGTH_PERTURB_MIN_ACTION_KEEP must be >= 1.")

# Input format selection
#   Old_All        : legacy single full-board snapshot
#   Old_All_Seq    : legacy board snapshot streamed for every action step
#   New_Multiply   : heterogeneous tokens (initial 25×34, actions encoded via seat×action-type×tile outer-product)
# New_Add        : heterogeneous tokens (initial 25×34, actions encoded as
# 4+9+34 additive features)
# "old_single", "old_seq", "New_Multiply", "New_Add"
USE_OLD_INPUT_FORMAT = _option("USE_OLD_INPUT_FORMAT", "New_Add")
_INPUT_FORMAT_CHOICES = {"old_single", "old_seq", "New_Multiply", "New_Add"}
if USE_OLD_INPUT_FORMAT not in _INPUT_FORMAT_CHOICES:
    raise ValueError(
        f"Unsupported USE_OLD_INPUT_FORMAT='{USE_OLD_INPUT_FORMAT}'. Valid options: {_INPUT_FORMAT_CHOICES}."
    )

if (
    CUR_MODEL in RESNET_MODEL_POOL or CUR_MODEL in CNN_MODEL_POOL
) and USE_OLD_INPUT_FORMAT != "old_single":
    raise ValueError(
        "CNN/ResNet models require USE_OLD_INPUT_FORMAT='old_single'."
    )

INPUT_FORMAT_IS_OLD_SINGLE = USE_OLD_INPUT_FORMAT == "old_single"
INPUT_FORMAT_IS_OLD_SEQUENCE = USE_OLD_INPUT_FORMAT == "old_seq"
INPUT_FORMAT_IS_OLD = (
    INPUT_FORMAT_IS_OLD_SINGLE or INPUT_FORMAT_IS_OLD_SEQUENCE
)
INPUT_FORMAT_IS_NEW_MULTIPLY = USE_OLD_INPUT_FORMAT == "New_Multiply"
INPUT_FORMAT_IS_NEW_ADD = USE_OLD_INPUT_FORMAT == "New_Add"

if (
    RANDOM_FEATURE_REPLACEMENT_SCOPE == "new_add_action_tokens"
    and not INPUT_FORMAT_IS_NEW_ADD
):
    raise ValueError(
        "RANDOM_FEATURE_REPLACEMENT_SCOPE='new_add_action_tokens' requires "
        "USE_OLD_INPUT_FORMAT='New_Add'."
    )

if OLD_SINGLE_ENCODER_ABLATION != "full":
    if not INPUT_FORMAT_IS_OLD_SINGLE:
        raise ValueError(
            "OLD_SINGLE_ENCODER_ABLATION modes other than 'full' require "
            "USE_OLD_INPUT_FORMAT='old_single'."
        )
    if CUR_MODEL != "transOriginal":
        raise ValueError(
            "OLD_SINGLE_ENCODER_ABLATION modes other than 'full' currently "
            "require CUR_MODEL='transOriginal'."
        )
    if HEAD_MODE != "whole":
        raise ValueError(
            "OLD_SINGLE_ENCODER_ABLATION modes other than 'full' require "
            "HEAD_MODE='whole'."
        )
    if USE_CLS_TOKEN:
        raise ValueError(
            "OLD_SINGLE_ENCODER_ABLATION modes other than 'full' require "
            "USE_CLS_TOKEN=false so the encoder sequence remains length 1."
        )

__unknown_option_names = sorted(set(_RAW_OPTION_OVERRIDES) - _SEEN_OPTION_NAMES)
if __unknown_option_names:
    logger.warning(
        f"Ignoring unknown utils option override(s): {', '.join(__unknown_option_names)}"
    )

if _APPLIED_OPTION_NAMES:
    logger.info(
        "Applied utils option override(s) from parser: "
        f"{', '.join(sorted(_APPLIED_OPTION_NAMES))}"
    )

"""================================"""

__all__ = [name for name in globals() if not name.startswith("__")]
