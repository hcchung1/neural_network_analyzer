"""Convolutional and residual baseline model families."""

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
from models.components import *


class CNN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (TOTAL_FEATURE_ROWS, FEAT_COLS),
        num_classes: int = 2,
        norm: str = "bn",
    ):
        """1D CNN that treats TOTAL_FEATURE_ROWS feature rows as channels and 34 tiles as the temporal axis."""
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.normalization = norm
        self.feature_rows = input_shape[0]
        self.feature_cols = input_shape[1]
        if self.feature_cols != FEAT_COLS:
            raise ValueError(
                f"CNN expects {FEAT_COLS} feature columns, got {self.feature_cols}."
            )
        if self.feature_rows <= 0:
            raise ValueError("CNN expects a positive feature_rows value.")

        self.conv1 = nn.Conv1d(
            in_channels=self.feature_rows,
            out_channels=1024,
            kernel_size=11,
            padding=11 // 2,
            bias=False,
        )
        self.bn1 = _make_conv1d_norm(norm, 1024)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv1d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=11,
            padding=11 // 2,
            bias=False,
        )
        self.bn2 = _make_conv1d_norm(norm, 1024)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv1d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=11,
            padding=11 // 2,
            bias=False,
        )
        self.bn3 = _make_conv1d_norm(norm, 1024)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv1d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=11,
            padding=11 // 2,
            bias=False,
        )
        self.bn4 = _make_conv1d_norm(norm, 1024)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.dropout4 = nn.Dropout(p=0.4)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, n, 34), (B, 34, n), (B, n*34), (B,1,n*34)
        if x.dim() == 2:
            total = x.size(1)
            if total % self.feature_cols != 0:
                raise ValueError(
                    f"Flattened input length {total} not divisible by feature_cols={self.feature_cols}."
                )
            seq_len = total // self.feature_cols
            if seq_len != self.feature_rows:
                raise ValueError(
                    f"Expected {self.feature_rows} feature rows after reshape, got {seq_len}."
                )
            x = x.view(x.size(0), seq_len, self.feature_cols)
        elif x.dim() == 3 and x.size(1) == 1:
            total = x.size(2)
            if total % self.feature_cols != 0:
                raise ValueError(
                    f"Flattened input length {total} not divisible by feature_cols={self.feature_cols}."
                )
            seq_len = total // self.feature_cols
            if seq_len != self.feature_rows:
                raise ValueError(
                    f"Expected {self.feature_rows} feature rows after reshape, got {seq_len}."
                )
            x = x.view(x.size(0), seq_len, self.feature_cols)

        if x.dim() != 3:
            raise ValueError(
                f"CNN expects input with dim=3, got {tuple(x.shape)}"
            )

        if x.size(1) == self.feature_rows and x.size(2) == self.feature_cols:
            return x
        if x.size(1) == self.feature_cols and x.size(2) == self.feature_rows:
            return x.transpose(1, 2)
        raise ValueError(
            f"Expected shape (B, {self.feature_rows}, {self.feature_cols}) or (B, {self.feature_cols}, {self.feature_rows}), got {tuple(x.shape)}"
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        x = self._reshape_input(x)

        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu4(self.bn4(self.conv4(x)))))

        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(self.fc(x))
        x = self.fc2(x)
        return x


# ==============================================================================
# Resnet
# ==============================================================================
# (TODO) Add Resnet model here


class BasicBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        k=11,
        dropout_rate=0.0,
        norm: str = "bn",
        gn_groups: int = 32,
    ):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = _make_conv1d_norm(norm, planes, gn_groups)

        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=k, stride=stride, padding=p, bias=False
        )
        self.bn2 = _make_conv1d_norm(norm, planes, gn_groups)

        self.conv3 = nn.Conv1d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = _make_conv1d_norm(norm, planes * self.expansion, gn_groups)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = (
            nn.Dropout(dropout_rate)
            if dropout_rate and dropout_rate > 0
            else nn.Identity()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _make_conv1d_norm(norm, planes * self.expansion, gn_groups),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(self.bn3(self.conv3(out)))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes: int = 2,
        input_shape: Tuple[int, int] = (TOTAL_FEATURE_ROWS, FEAT_COLS),
        dropout_rate: float = 0.1,
        base_planes: int = 64,
        stem_width: int = 64,
        norm: str = "bn",
        gn_groups: int = 32,
    ):
        super().__init__()
        self.in_planes = stem_width
        self.input_shape = input_shape
        self.feature_rows = input_shape[0]
        self.feature_cols = input_shape[1]
        if self.feature_cols != FEAT_COLS:
            raise ValueError(
                f"ResNet expects {FEAT_COLS} feature columns, got {self.feature_cols}."
            )
        if self.feature_rows <= 0:
            raise ValueError("ResNet expects a positive feature_rows value.")
        self.dropout_rate = dropout_rate
        self.norm = norm
        self.gn_groups = gn_groups
        if base_planes <= 0:
            raise ValueError("ResNet expects base_planes > 0")
        if stem_width <= 0:
            raise ValueError("ResNet expects stem_width > 0")
        self.base_planes = base_planes
        self.stem_width = stem_width

        # Use TOTAL_FEATURE_ROWS feature rows as channels (TOTAL_FEATURE_ROWS ×
        # FEAT_COLS layout)
        self.conv1 = nn.Conv1d(
            self.feature_rows,
            stem_width,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = _make_conv1d_norm(norm, stem_width, gn_groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        planes = base_planes
        self.layer1 = self._make_layer(
            block, planes, num_blocks[0], stride=1, dropout_rate=dropout_rate
        )
        self.layer2 = self._make_layer(
            block, planes, num_blocks[1], stride=2, dropout_rate=dropout_rate
        )
        self.layer3 = self._make_layer(
            block, planes, num_blocks[2], stride=2, dropout_rate=dropout_rate
        )
        self.layer4 = self._make_layer(
            block, planes, num_blocks[3], stride=2, dropout_rate=dropout_rate
        )
        self.layer5 = self._make_layer(
            block, planes, num_blocks[3], stride=2, dropout_rate=dropout_rate
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(base_planes * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    dropout_rate=dropout_rate,
                    norm=self.norm,
                    gn_groups=self.gn_groups,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(
                m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)
            ):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x, attention_mask: Optional[torch.Tensor] = None, **_: Any
    ):
        # Accept flat (B, n*34) or (B, 1, n*34) and reshape to (B,
        # TOTAL_FEATURE_ROWS, FEAT_COLS)
        if x.dim() == 2:
            total = x.size(1)
            if total % self.feature_cols != 0:
                raise ValueError(
                    f"Flattened input length {total} not divisible by feature_cols={self.feature_cols}."
                )
            seq_len = total // self.feature_cols
            if seq_len != self.feature_rows:
                raise ValueError(
                    f"Expected {self.feature_rows} feature rows after reshape, got {seq_len}."
                )
            x = x.view(x.size(0), seq_len, self.feature_cols)
        elif x.dim() == 3 and x.size(1) == 1:
            total = x.size(2)
            if total % self.feature_cols != 0:
                raise ValueError(
                    f"Flattened input length {total} not divisible by feature_cols={self.feature_cols}."
                )
            seq_len = total // self.feature_cols
            if seq_len != self.feature_rows:
                raise ValueError(
                    f"Expected {self.feature_rows} feature rows after reshape, got {seq_len}."
                )
            x = x.view(x.size(0), seq_len, self.feature_cols)

        if x.dim() != 3:
            raise ValueError(
                f"ResNet expects input with dim=3, got {tuple(x.shape)}"
            )

        if x.size(1) == self.feature_rows and x.size(2) == self.feature_cols:
            pass
        elif x.size(1) == self.feature_cols and x.size(2) == self.feature_rows:
            x = x.transpose(1, 2)
        else:
            raise ValueError(
                f"Expected shape (B, {self.feature_rows}, {self.feature_cols}) or (B, {self.feature_cols}, {self.feature_rows}), got {tuple(x.shape)}"
            )

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = self.avgpool(out)
        out = out.squeeze(-1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
def ResNet18(
    input_shape: Tuple[int, int] = (TOTAL_FEATURE_ROWS, FEAT_COLS),
    num_classes: int = 2,
    dropout_rate=0.1,
    norm: str = "gn",
):
    """ResNet-18 with 8 residual blocks (約18層)"""
    # Widened to match CNN-scale parameter count (~same order of magnitude)
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        base_planes=256,
        norm=norm,
    )
def ResNet10(
    input_shape: Tuple[int, int] = (TOTAL_FEATURE_ROWS, FEAT_COLS),
    num_classes: int = 2,
    dropout_rate=0.1,
    norm: str = "gn",
):
    """較小的 ResNet with 4 residual blocks (約10層)"""
    return ResNet(
        BasicBlock,
        [1, 1, 1, 1],
        num_classes=num_classes,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        base_planes=256,
        norm=norm,
    )


def ResNet14(
    input_shape: Tuple[int, int] = (TOTAL_FEATURE_ROWS, FEAT_COLS),
    num_classes: int = 2,
    dropout_rate=0.1,
    norm: str = "gn",
):
    """中等大小的 ResNet with 6 residual blocks (約14層)"""
    return ResNet(
        BasicBlock,
        [1, 2, 2, 1],
        num_classes=num_classes,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        base_planes=256,
        norm=norm,
    )


def ResNet22(
    input_shape: Tuple[int, int] = (TOTAL_FEATURE_ROWS, FEAT_COLS),
    num_classes: int = 2,
    dropout_rate=0.1,
    norm: str = "gn",
):
    """更深的 ResNet with 10 residual blocks (約22層)"""
    return ResNet(
        BasicBlock,
        [2, 3, 3, 3],
        num_classes=num_classes,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        base_planes=256,
        norm=norm,
    )


def ResNet15(
    input_shape: Tuple[int, int] = (TOTAL_FEATURE_ROWS, FEAT_COLS),
    num_classes=2,
    dropout_rate=0.1,
    norm: str = "gn",
):
    return ResNet(
        BasicBlock,
        [3, 4, 4, 4],
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        base_planes=256,
        norm=norm,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
