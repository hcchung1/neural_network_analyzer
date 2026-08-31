"""Transformer, Conformer, and variational-oracle model families."""

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


class MahjongTransformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (
            MAX_SEQ_LEN_FALLBACK,
            ACTION_FEATURE_DIM,
        ),
        d_model: int = 128 * 3,
        n_heads: int = 12,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 2,
        old_single_encoder_ablation: str = "full",
        norm: str = "ln",
    ):
        """
        Args:
            input_shape: Input Features Shape (seq_len, features) -> (預設 136, 333*34)
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super(MahjongTransformer, self).__init__()
        self.supports_init_sidecar = True
        self.input_shape = input_shape
        self.d_model = d_model
        self.seq_len = input_shape[0]
        self.input_features = input_shape[1]
        self.normalization = norm
        self.old_single_encoder_ablation = str(
            old_single_encoder_ablation
        ).strip().lower()
        if self.old_single_encoder_ablation not in {
            "full",
            "value_only",
            "ffn_only",
        }:
            raise ValueError(
                "old_single_encoder_ablation must be one of "
                "{'full', 'value_only', 'ffn_only'}. "
                f"Got '{self.old_single_encoder_ablation}'."
            )
        if self.old_single_encoder_ablation != "full":
            if not INPUT_FORMAT_IS_OLD_SINGLE or self.seq_len != 1:
                raise ValueError(
                    "Old-single encoder ablations require old_single input "
                    "with sequence length 1."
                )
            if USE_CLS_TOKEN:
                raise ValueError(
                    "Old-single encoder ablations require USE_CLS_TOKEN=false."
                )

        logger.info(
            f"MahjongTransformer initialized with padded input shape: {input_shape}"
        )
        logger.info(
            "Model parameters: "
            f"d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, "
            f"normalization={norm}"
        )

        # (TODO) 確認 embedding 之前的 input 格式

        # self.input_projection = nn.Linear(self.input_features, d_model)

        # two-branch projection for heterogeneous tokens
        self.cols = FEAT_COLS
        self.F_INIT = F_INIT
        self.F_ACT = F_ACT_ROWS * self.cols  # 1224
        self.F_ACT_ADD = F_ACT_ADD_DIM  # 4 + 10 + 34 + 1 + 1 = 50
        self.F_PAD = max(self.F_INIT, self.F_ACT)  # 1224
        self.F_PAD_ADD = max(self.F_INIT, self.F_ACT_ADD)  # 850

        self.token_mode: Optional[str] = None
        if INPUT_FORMAT_IS_NEW_MULTIPLY and self.input_features == self.F_PAD:
            self.token_mode = "multiply"
        elif INPUT_FORMAT_IS_NEW_ADD and self.input_features == self.F_PAD_ADD:
            self.token_mode = "add"
        elif self.input_features == self.F_PAD_ADD:
            self.token_mode = "add"
        elif self.input_features == self.F_PAD:
            self.token_mode = "multiply"

        self.use_heter_tokens = self.token_mode is not None
        if self.use_heter_tokens:
            self.proj_init = nn.Linear(self.F_INIT, d_model, bias=False)
            act_dim = (
                self.F_ACT if self.token_mode == "multiply" else self.F_ACT_ADD
            )
            self.proj_act = nn.Linear(act_dim, d_model, bias=False)
            self._act_feature_dim = act_dim
            self._expected_feature_dim = (
                self.F_PAD if self.token_mode == "multiply" else self.F_PAD_ADD
            )
        else:
            self.input_projection = nn.Linear(
                self.input_features, d_model, bias=False
            )

        self.use_abs_pos = bool(USE_ABS_POS_ENCODING)
        self.use_token_type_emb = bool(USE_TOKEN_TYPE_EMBEDDING)
        self.use_meta_emb = bool(USE_SEAT_ACTION_PHASE_EMBEDDING)
        self.use_cls_token = bool(USE_CLS_TOKEN)
        self.use_learnable_pool = bool(USE_LEARNABLE_POOL)
        if self.use_token_type_emb:
            token_type_vocab = 3 if self.use_cls_token else 2
            self.token_type_embed = nn.Embedding(
                token_type_vocab, TOKEN_TYPE_EMBED_DIM
            )
            self.token_type_proj = nn.Linear(
                TOKEN_TYPE_EMBED_DIM, d_model, bias=False
            )
        if self.use_meta_emb:
            self.seat_embed = nn.Embedding(
                NEW_ADD_PLAYER_ROWS + 1, SEAT_EMBED_DIM
            )
            self.action_embed = nn.Embedding(
                ACTION_TYPE_COUNT + 1, ACTION_EMBED_DIM
            )
            self.phase_embed = nn.Embedding(3 + 1, PHASE_EMBED_DIM)
            self.seat_proj = nn.Linear(SEAT_EMBED_DIM, d_model, bias=False)
            self.action_proj = nn.Linear(ACTION_EMBED_DIM, d_model, bias=False)
            self.phase_proj = nn.Linear(PHASE_EMBED_DIM, d_model, bias=False)

        if self.use_cls_token and self.use_learnable_pool:
            raise ValueError(
                "use_cls_token and use_learnable_pool are mutually exclusive. "
                "Enable at most one pooling strategy."
            )
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.use_learnable_pool:
            self.pool_query = nn.Parameter(torch.randn(d_model))

        # Encoder sequence length may include an extra CLS token.
        self.encoder_seq_len = self.seq_len + (1 if self.use_cls_token else 0)

        # self.cols = 34
        # self.F_INIT = int(STATIC_FEATURE_ROWS * self.cols)  # 25 * 34
        # self.ACTION_FEATURE_ROWS = 40
        # self.F_ACT = int(self.ACTION_FEATURE_ROWS * self.cols)

        # self.proj_init = nn.Linear(self.F_INIT, d_model, bias=False)
        # self.proj_act  = nn.Linear(self.F_ACT,  d_model, bias=False)

        self.pos_encoding = PositionalEncoding(
            d_model, max_len=self.encoder_seq_len, dropout=dropout
        )

        self.transformer_layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    single_token_ablation=self.old_single_encoder_ablation,
                    norm=norm,
                )
                for _ in range(n_layers)
            ]
        )

        inactive_parameters: Dict[int, nn.Parameter] = {}
        qk_parameters: Dict[int, nn.Parameter] = {}
        if (
            INPUT_FORMAT_IS_OLD_SINGLE
            and self.seq_len == 1
            and not self.use_cls_token
        ):
            for layer in self.transformer_layers:
                for module in (layer.attention.w_q, layer.attention.w_k):
                    for parameter in module.parameters():
                        qk_parameters[id(parameter)] = parameter
                if self.old_single_encoder_ablation in {"full", "value_only"}:
                    inactive_modules = (
                        layer.attention.w_q,
                        layer.attention.w_k,
                        layer.norm1,
                        layer.norm2,
                    )
                else:
                    inactive_modules = (
                        layer.attention,
                        layer.ln_attn_in,
                        layer.norm1,
                        layer.norm2,
                    )
                for module in inactive_modules:
                    for parameter in module.parameters():
                        inactive_parameters[id(parameter)] = parameter
        self.single_token_qk_parameter_count = sum(
            parameter.numel() for parameter in qk_parameters.values()
        )
        self.single_token_inactive_parameter_count = sum(
            parameter.numel() for parameter in inactive_parameters.values()
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Head mode:
        # pooler | linear | first_token | last_token | flinear | multi_pool_flinear | cnn2d
        self.head_mode_str = str(USE_SIMPLE_LINEAR_HEAD)
        # Use the requested class count for multi-class modes such as shanten.
        # Keep the historical binary head shape for ordinary tenpai training.
        _requested_out = int(num_classes)
        if _requested_out > 2:
            _num_out = _requested_out
        else:
            _num_out = 2 if SOFTMAX_LOSS else 1
        if self.head_mode_str == "pooler":
            # BERT-style pooler (Linear + Tanh + Dropout) then classifier
            self.pooler = nn.Linear(d_model, d_model)
            self.pooler_act = nn.Tanh()
            self.pooler_dropout = nn.Dropout(dropout)
            self.tenpai_classifier = nn.Linear(d_model, _num_out)
        elif self.head_mode_str == "flinear":
            # Flatten entire sequence then project
            self.tenpai_classifier = nn.Linear(
                self.encoder_seq_len * d_model, _num_out
            )
        elif self.head_mode_str == "multi_pool_flinear":
            self.multi_pool_query_count = int(MULTI_POOL_QUERIES)
            if self.multi_pool_query_count < 1:
                raise ValueError(
                    "MULTI_POOL_QUERIES must be >= 1. "
                    f"Got {self.multi_pool_query_count}."
                )
            self.multi_pool_queries = nn.Parameter(
                torch.empty(self.multi_pool_query_count, d_model)
            )
            nn.init.trunc_normal_(self.multi_pool_queries, std=0.02)
            self.multi_pool_dropout = nn.Dropout(float(MULTI_POOL_DROPOUT))
            self.tenpai_classifier = nn.Linear(
                self.multi_pool_query_count * d_model, _num_out
            )
        elif self.head_mode_str == "cnn2d":
            # ViT-style: treat (seq_len, d_model) as 2D feature map, apply
            # Conv2d
            self.cnn2d_head = CNN2DClassificationHead(
                d_model=d_model,
                seq_len=self.encoder_seq_len,
                num_out=_num_out,
                base_channels=64,
                dropout=dropout,
                norm=norm,
            )
            self.tenpai_classifier = (
                None  # classification is inside cnn2d_head
            )
        else:
            # linear / first_token / last_token all use a single Linear
            self.tenpai_classifier = nn.Linear(d_model, _num_out)

        self.use_length_grl = bool(USE_LENGTH_GRL)
        self.length_grl_weight = float(LENGTH_GRL_WEIGHT)
        self.length_grl_lambda = float(LENGTH_GRL_LAMBDA)
        self.length_predictor_bucket_size = max(
            1, int(LENGTH_PREDICTOR_BUCKET_SIZE)
        )
        self.length_predictor_max_length = max(1, int(self.seq_len))
        self.length_predictor_num_classes = (
            self.length_predictor_max_length
            // self.length_predictor_bucket_size
            + 1
        )
        self.length_predictor: Optional[LengthPredictor]
        if self.use_length_grl:
            if self.head_mode_str == "flinear":
                length_repr_dim = self.encoder_seq_len * d_model
            elif self.head_mode_str == "multi_pool_flinear":
                length_repr_dim = self.multi_pool_query_count * d_model
            else:
                length_repr_dim = d_model
            self.length_predictor = LengthPredictor(
                input_dim=length_repr_dim,
                num_classes=self.length_predictor_num_classes,
                hidden_dim=int(LENGTH_PREDICTOR_HIDDEN_DIM),
                dropout=float(LENGTH_PREDICTOR_DROPOUT),
            )
            logger.info(
                "Length GRL enabled: "
                f"classes={self.length_predictor_num_classes}, "
                f"bucket_size={self.length_predictor_bucket_size}, "
                f"lambda={self.length_grl_lambda}, "
                f"weight={self.length_grl_weight}"
            )
        else:
            self.length_predictor = None
        self._auxiliary_loss = 0.0
        self._length_predictor_loss = 0.0
        self._length_predictor_accuracy = 0.0

        # split_keep merge strategy: "add" (element-wise sum) or "concat" (cat
        # + project)
        self._split_keep_merge = SPLIT_KEEP_MERGE_MODE
        _is_split_keep_family = any(
            getattr(layer, "split_keep", False)
            for layer in self.transformer_layers
        )
        if _is_split_keep_family and self._split_keep_merge == "concat":
            self.branch_merge_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Mish(),
                nn.Dropout(dropout),
            )
        else:
            self.branch_merge_proj = None

        # split_cv: model learns attack/defense via auxiliary per-token
        # supervision
        self.split_cv = HEAD_MODE == "split_cv"
        if self.split_cv:
            self.aux_attack_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
            )
            self.aux_defense_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
            )
            self._auxiliary_loss = 0.0
            # V-projection level probes (shared across all encoder layers)
            self.v_aux_attack_probe = nn.Linear(d_model, 1)
            self.v_aux_defense_probe = nn.Linear(d_model, 1)

        # MTP (Masked Tile Prediction) head for pre-training
        self.mtp_pretraining = MTP_PRETRAINING
        if self.mtp_pretraining:
            self.mtp_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                _make_sequence_norm(norm, d_model),
                nn.Linear(d_model, MTP_NUM_TILE_CLASSES),
            )
            self.mtp_use_aux_cls = MTP_USE_AUX_CLASSIFICATION

    def _add_auxiliary_loss(self, loss: torch.Tensor) -> None:
        current = getattr(self, "_auxiliary_loss", 0.0)
        if isinstance(current, torch.Tensor):
            self._auxiliary_loss = current + loss
        elif isinstance(current, (int, float)) and current != 0.0:
            self._auxiliary_loss = (
                torch.as_tensor(
                    float(current), device=loss.device, dtype=loss.dtype
                )
                + loss
            )
        else:
            self._auxiliary_loss = loss

    def _length_labels_from_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        fallback_length: Optional[int] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            fallback = self.seq_len if fallback_length is None else fallback_length
            seq_lens = torch.full(
                (batch_size,),
                int(fallback),
                device=device,
                dtype=torch.long,
            )
        else:
            seq_lens = (
                attention_mask.to(device=device, dtype=torch.bool)
                .sum(dim=1)
                .long()
            )
        seq_lens = seq_lens.clamp(
            min=0, max=int(self.length_predictor_max_length)
        )
        labels = torch.div(
            seq_lens,
            int(self.length_predictor_bucket_size),
            rounding_mode="floor",
        )
        return labels.clamp(max=int(self.length_predictor_num_classes) - 1)

    @staticmethod
    def _masked_mean_representation(
        x: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if attention_mask is None:
            return x.mean(dim=1)
        mask = attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def _multi_pool_flinear_representation(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        scores = torch.matmul(x, self.multi_pool_queries.t()).transpose(1, 2)
        scores = scores / math.sqrt(x.size(-1))
        if attention_mask is not None:
            mask = attention_mask.to(device=x.device, dtype=torch.bool)
            mask = mask.unsqueeze(1)  # (B,1,T)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            weights = weights * mask.to(dtype=weights.dtype)
        else:
            weights = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(weights, x)  # (B,K,D)
        return self.multi_pool_dropout(pooled.flatten(start_dim=1))

    def _maybe_add_length_grl_loss(
        self,
        representation: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        fallback_length: Optional[int] = None,
    ) -> None:
        if (
            (not self.training)
            or self.length_predictor is None
            or self.length_grl_weight <= 0.0
        ):
            self._length_predictor_loss = 0.0
            self._length_predictor_accuracy = 0.0
            return

        length_targets = self._length_labels_from_mask(
            attention_mask=attention_mask,
            batch_size=representation.size(0),
            device=representation.device,
            fallback_length=fallback_length,
        )
        reversed_representation = _gradient_reverse(
            representation, self.length_grl_lambda
        )
        length_logits = self.length_predictor(reversed_representation)
        raw_loss = F.cross_entropy(length_logits, length_targets)
        weighted_loss = raw_loss * self.length_grl_weight
        self._add_auxiliary_loss(weighted_loss)
        self._length_predictor_loss = raw_loss.detach()
        with torch.no_grad():
            self._length_predictor_accuracy = (
                length_logits.argmax(dim=1).eq(length_targets).float().mean()
            ).detach()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mtp_mask_positions: Optional[torch.Tensor] = None,
        init_sidecar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function of the Transformer model
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim)
            attention_mask (torch.Tensor, optional): shape (batch_size, seq_len)
            mtp_mask_positions (torch.Tensor, optional): Boolean tensor indicating masked positions
                                                         for MTP pre-training. shape (batch_size, seq_len)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - tenpai_logits: 聽牌判斷 logits (batch_size,) or MTP logits (batch_size, seq_len, 34)
                - waits_logits: 等待預測 logits (batch_size, 34) - currently unused
        """

        # if x.dim() == 2:
        #     x = x.unsqueeze(1)
        # elif x.dim() != 3:
        #     raise ValueError(f"Expected 2D/3D input, got {tuple(x.shape)}")
        # B, T, F = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {tuple(x.shape)}")

        self._auxiliary_loss = 0.0
        self._length_predictor_loss = 0.0
        self._length_predictor_accuracy = 0.0
        seq_len = x.size(1)
        length_attention_mask = attention_mask
        x_raw = x
        value_modulators: Optional[torch.Tensor] = None
        x_pair_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # split_cv auxiliary attack target (per action token)
        _cv_at_labels = None
        # split_cv auxiliary defense target (per action token)
        _cv_df_labels = None

        if self.use_heter_tokens:
            expected_dim = getattr(
                self, "_expected_feature_dim", self.input_features
            )
            compact_add_input = bool(
                init_sidecar is not None and init_sidecar.numel() > 0
            )
            accepted_dim = (
                PACKED_ACTION_FEATURE_DIM if compact_add_input else expected_dim
            )
            if x.shape[-1] != accepted_dim:
                raise ValueError(
                    f"Expected feature dim {accepted_dim}, got {x.shape[-1]}"
                )

            if compact_add_input:
                if (
                    init_sidecar.dim() != 2
                    or init_sidecar.size(0) != x.size(0)
                    or init_sidecar.size(1) != NEW_ADD_INIT_SIDECAR_DIM
                ):
                    raise ValueError(
                        "Invalid init sidecar shape: "
                        f"expected ({x.size(0)}, {NEW_ADD_INIT_SIDECAR_DIM}), "
                        f"got {tuple(init_sidecar.shape)}"
                    )
                x_init = torch.cat(
                    [x[:, 0, :F_INIT_COMPACT], init_sidecar], dim=-1
                )
            else:
                x_init = x[:, 0, : self.F_INIT]
            act_dim = getattr(self, "_act_feature_dim", self.input_features)
            x_act = x[:, 1:, :act_dim]

            # split_cus_v: create two feature variants by forcing danger_flag to extreme values
            # - attack branch: danger_flag := 1
            # - defense branch: danger_flag := 0
            # This happens BEFORE projection, so branches learn different
            # representations.
            if (
                HEAD_MODE == "split_cus_v"
                and self.token_mode == "add"
                and seq_len > 1
            ):
                flag_base = (
                    NEW_ADD_PLAYER_ROWS
                    + NEW_ADD_ACTION_ROWS
                    + NEW_ADD_TILE_ROWS
                    + (TIME_SECTION_ROWS if TIME_SECTION_IN_ACTION else 0)
                )
                danger_offset = (
                    flag_base
                    + NEW_ADD_RIICHI_FLAG_ROWS
                    + NEW_ADD_DORA_FLAG_ROWS
                )
                danger_end = danger_offset + NEW_ADD_DANGER_FLAG_ROWS

                if danger_end <= act_dim and NEW_ADD_DANGER_FLAG_ROWS > 0:
                    x_act_at = x_act.clone()
                    x_act_df = x_act.clone()
                    x_act_at[:, :, danger_offset:danger_end] = 1.0
                    x_act_df[:, :, danger_offset:danger_end] = 0.0

                    x_init_proj = self.proj_init(x_init)
                    x_act_at_proj = self.proj_act(x_act_at)
                    x_act_df_proj = self.proj_act(x_act_df)

                    x_at = torch.cat(
                        [x_init_proj.unsqueeze(1), x_act_at_proj], dim=1
                    )
                    x_df = torch.cat(
                        [x_init_proj.unsqueeze(1), x_act_df_proj], dim=1
                    )
                    x_pair_override = (x_at, x_df)
                else:
                    logger.warning(
                        (
                            "split_cus_v enabled but danger_flag slice "
                            f"[{danger_offset}:{danger_end}] is out of bounds "
                            f"for act_dim={act_dim}; falling back to shared "
                            "features."
                        )
                    )

            # split_cv: extract attack/defense ground truth from raw action
            # features
            if self.split_cv and self.token_mode == "add" and seq_len > 1:
                _flag_base = (
                    NEW_ADD_PLAYER_ROWS
                    + NEW_ADD_ACTION_ROWS
                    + NEW_ADD_TILE_ROWS
                    + (TIME_SECTION_ROWS if TIME_SECTION_IN_ACTION else 0)
                )
                _v_at_off = (
                    _flag_base
                    + NEW_ADD_RIICHI_FLAG_ROWS
                    + NEW_ADD_DORA_FLAG_ROWS
                    + NEW_ADD_DANGER_FLAG_ROWS
                    + NEW_ADD_ONE_CHANCE_FLAG_ROWS
                    + NEW_ADD_NO_CHANCE_FLAG_ROWS
                )
                _v_df_off = _v_at_off + NEW_ADD_ATTACK_EMBED_ROWS
                if _v_df_off + NEW_ADD_DEFENSE_EMBED_ROWS <= act_dim:
                    _cv_at_labels = (
                        x_act[
                            :,
                            :,
                            _v_at_off : _v_at_off + NEW_ADD_ATTACK_EMBED_ROWS,
                        ]
                        .mean(dim=-1)
                        .detach()
                    )
                    _cv_df_labels = (
                        x_act[
                            :,
                            :,
                            _v_df_off : _v_df_off + NEW_ADD_DEFENSE_EMBED_ROWS,
                        ]
                        .mean(dim=-1)
                        .detach()
                    )
                    # Zero out v_at/v_df slots so the model cannot cheat by
                    # reading them directly
                    x_act = x_act.clone()
                    x_act[
                        :, :, _v_at_off : _v_at_off + NEW_ADD_ATTACK_EMBED_ROWS
                    ] = 0.0
                    x_act[
                        :,
                        :,
                        _v_df_off : _v_df_off + NEW_ADD_DEFENSE_EMBED_ROWS,
                    ] = 0.0

            if (
                self.token_mode == "add"
                and seq_len > 1
                and not self.split_cv
                and HEAD_MODE != "split_cus_v"
            ):
                base = torch.full(
                    (x.size(0), seq_len, 2),
                    0.0,
                    device=x.device,
                    dtype=x.dtype,
                )
                base[:, :, 0] = 1.0

                use_mods = False
                attack = base[:, :, 0]
                defense = base[:, :, 1]

                action_start = NEW_ADD_PLAYER_ROWS
                action_end = action_start + NEW_ADD_ACTION_ROWS

                flag_base = (
                    action_end
                    + NEW_ADD_TILE_ROWS
                    + (TIME_SECTION_ROWS if TIME_SECTION_IN_ACTION else 0)
                )
                riichi_offset = flag_base
                dora_offset = riichi_offset + NEW_ADD_RIICHI_FLAG_ROWS
                danger_offset = dora_offset + NEW_ADD_DORA_FLAG_ROWS
                one_chance_offset = danger_offset + NEW_ADD_DANGER_FLAG_ROWS
                no_chance_offset = (
                    one_chance_offset + NEW_ADD_ONE_CHANCE_FLAG_ROWS
                )

                if (
                    # split_custom_v handles this logic separately above, so
                    # skip if that mode is active
                    HEAD_MODE == "split_custom_v"
                    and action_end <= x_raw.size(-1)
                    and (no_chance_offset + NEW_ADD_NO_CHANCE_FLAG_ROWS)
                    <= x_raw.size(-1)
                ):
                    action_slice = x_raw[:, :, action_start:action_end]
                    action_sum = action_slice.sum(dim=-1)
                    action_ids = action_slice.argmax(dim=-1)

                    is_action_token = torch.zeros(
                        (x.size(0), seq_len), device=x.device, dtype=torch.bool
                    )
                    if seq_len > 1:
                        is_action_token[:, 1:] = True
                    has_action = is_action_token & (action_sum > 0)

                    riichi_safe_slice = x_raw[
                        :,
                        :,
                        riichi_offset : riichi_offset
                        + NEW_ADD_RIICHI_FLAG_ROWS,
                    ]
                    riichi_safe_flag = riichi_safe_slice.mean(dim=-1) > 0.5
                    dora_slice = x_raw[
                        :,
                        :,
                        dora_offset : dora_offset + NEW_ADD_DORA_FLAG_ROWS,
                    ]
                    dora_flag = dora_slice.mean(dim=-1) > 0.5
                    danger_slice = x_raw[
                        :,
                        :,
                        danger_offset : danger_offset
                        + NEW_ADD_DANGER_FLAG_ROWS,
                    ]
                    danger_flag = danger_slice.mean(dim=-1) > 0.5
                    one_chance_slice = x_raw[
                        :,
                        :,
                        one_chance_offset : one_chance_offset
                        + NEW_ADD_ONE_CHANCE_FLAG_ROWS,
                    ]
                    one_chance_flag = one_chance_slice.mean(dim=-1) > 0.5
                    no_chance_slice = x_raw[
                        :,
                        :,
                        no_chance_offset : no_chance_offset
                        + NEW_ADD_NO_CHANCE_FLAG_ROWS,
                    ]
                    no_chance_flag = no_chance_slice.mean(dim=-1) > 0.5

                    is_call = has_action & (action_ids <= 6)
                    is_discard = has_action & (action_ids >= 8)

                    attack = torch.where(
                        is_call, torch.full_like(attack, 0.65), attack
                    )
                    defense = torch.where(
                        is_call, torch.full_like(defense, 0.35), defense
                    )

                    attack = torch.where(
                        is_discard, torch.full_like(attack, 0.4), attack
                    )
                    defense = torch.where(
                        is_discard, torch.full_like(defense, 0.6), defense
                    )

                    # New_Add five-flag semantics:
                    # offense -> V_at: danger_flag, dora_flag
                    # defense -> V_df: riichi_safe_flag, one_chance_flag,
                    # no_chance_flag
                    attack = attack + (
                        (has_action & danger_flag).to(attack.dtype) * 0.2
                        + (has_action & dora_flag).to(attack.dtype) * 0.2
                    )
                    defense = defense + (
                        (has_action & riichi_safe_flag).to(defense.dtype) * 0.2
                        + (has_action & one_chance_flag).to(defense.dtype)
                        * 0.2
                        + (has_action & no_chance_flag).to(defense.dtype) * 0.2
                    )
                    use_mods = True
                else:
                    mod_tail = (
                        NEW_ADD_ATTACK_EMBED_ROWS + NEW_ADD_DEFENSE_EMBED_ROWS
                    )
                    if act_dim >= mod_tail:
                        tail = x_act[:, :, -mod_tail:]
                        v_at_vals = tail[
                            :, :, :NEW_ADD_ATTACK_EMBED_ROWS
                        ].mean(dim=-1)
                        v_df_vals = tail[
                            :, :, NEW_ADD_ATTACK_EMBED_ROWS:
                        ].mean(dim=-1)
                        attack[:, 1:] = v_at_vals
                        defense[:, 1:] = v_df_vals
                        use_mods = True

                use_flag_modulator = HEAD_MODE == "split_custom_v"
                if use_flag_modulator and (
                    no_chance_offset + NEW_ADD_NO_CHANCE_FLAG_ROWS
                ) <= x_raw.size(-1):
                    dora_flag = (
                        x_raw[
                            :,
                            :,
                            dora_offset : dora_offset + NEW_ADD_DORA_FLAG_ROWS,
                        ].mean(dim=-1)
                        > 0.5
                    )
                    danger_flag = (
                        x_raw[
                            :,
                            :,
                            danger_offset : danger_offset
                            + NEW_ADD_DANGER_FLAG_ROWS,
                        ].mean(dim=-1)
                        > 0.5
                    )
                    one_chance_flag = (
                        x_raw[
                            :,
                            :,
                            one_chance_offset : one_chance_offset
                            + NEW_ADD_ONE_CHANCE_FLAG_ROWS,
                        ].mean(dim=-1)
                        > 0.5
                    )
                    no_chance_flag = (
                        x_raw[
                            :,
                            :,
                            no_chance_offset : no_chance_offset
                            + NEW_ADD_NO_CHANCE_FLAG_ROWS,
                        ].mean(dim=-1)
                        > 0.5
                    )
                    riichi_safe_flag = (
                        x_raw[
                            :,
                            :,
                            riichi_offset : riichi_offset
                            + NEW_ADD_RIICHI_FLAG_ROWS,
                        ].mean(dim=-1)
                        > 0.5
                    )

                    attack = attack + (
                        danger_flag.to(attack.dtype)
                        + dora_flag.to(attack.dtype)
                    )
                    defense = defense + (
                        one_chance_flag.to(defense.dtype)
                        + no_chance_flag.to(defense.dtype)
                        + riichi_safe_flag.to(defense.dtype)
                    )
                    use_mods = True

                if use_mods:
                    base[:, :, 0] = attack.clamp(0.0, 1.0)
                    base[:, :, 1] = defense.clamp(0.0, 1.0)
                    value_modulators = base

            if x_pair_override is None:
                x_init = self.proj_init(x_init)
                x_act = self.proj_act(x_act)
                x = torch.cat([x_init.unsqueeze(1), x_act], dim=1)
            else:
                x = x_pair_override
        else:
            if x.shape[-1] != self.input_features:
                raise ValueError(
                    f"Expected feature dim {self.input_features}, got {x.shape[-1]}"
                )
            x = self.input_projection(x)

        batch_size = x[0].size(0) if isinstance(x, tuple) else x.size(0)

        if self.use_cls_token:
            cls_tok = self.cls_token.expand(batch_size, 1, -1)
            if isinstance(x, tuple):
                x = (
                    torch.cat([cls_tok, x[0]], dim=1),
                    torch.cat([cls_tok, x[1]], dim=1),
                )
            else:
                x = torch.cat([cls_tok, x], dim=1)
            seq_len = x[0].size(1) if isinstance(x, tuple) else x.size(1)
            if attention_mask is not None:
                cls_mask = torch.ones(
                    (batch_size, 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
            if value_modulators is not None:
                cls_mod = torch.ones(
                    (batch_size, 1, 2),
                    device=value_modulators.device,
                    dtype=value_modulators.dtype,
                )
                value_modulators = torch.cat(
                    [cls_mod, value_modulators], dim=1
                )

        if self.use_token_type_emb or self.use_meta_emb:
            x_ref = x[0] if isinstance(x, tuple) else x
            emb_acc = torch.zeros_like(x_ref)
            token_mask = (
                attention_mask.to(x_ref.device, dtype=torch.bool)
                if attention_mask is not None
                else None
            )

            if self.use_token_type_emb:
                token_type_ids = torch.zeros(
                    (batch_size, seq_len),
                    device=x_ref.device,
                    dtype=torch.long,
                )
                init_offset = 1 if self.use_cls_token else 0
                if seq_len > init_offset:
                    token_type_ids[:, init_offset] = 0
                if seq_len > init_offset + 1:
                    token_type_ids[:, init_offset + 1 :] = 1
                if self.use_cls_token:
                    token_type_ids[:, 0] = 2
                if token_mask is not None:
                    token_type_ids = token_type_ids * token_mask.to(
                        dtype=torch.long
                    )
                token_type_emb = self.token_type_proj(
                    self.token_type_embed(token_type_ids)
                )
                if token_mask is not None:
                    token_type_emb = token_type_emb * token_mask.unsqueeze(-1)
                emb_acc = emb_acc + token_type_emb

            if self.use_meta_emb and self.token_mode == "add":
                seat_slice = x_raw[:, :, :NEW_ADD_PLAYER_ROWS]
                action_slice = x_raw[
                    :,
                    :,
                    NEW_ADD_PLAYER_ROWS : NEW_ADD_PLAYER_ROWS
                    + NEW_ADD_ACTION_ROWS,
                ]

                seat_sum = seat_slice.sum(dim=-1)
                action_sum = action_slice.sum(dim=-1)

                seat_ids = seat_slice.argmax(dim=-1) + 1
                action_ids = action_slice.argmax(dim=-1) + 1

                orig_len = x_raw.size(1)
                is_action_token = torch.zeros(
                    (batch_size, orig_len),
                    device=x_ref.device,
                    dtype=torch.bool,
                )
                if orig_len > 1:
                    is_action_token[:, 1:] = True

                seat_ids = torch.where(
                    is_action_token & (seat_sum > 0),
                    seat_ids,
                    torch.zeros_like(seat_ids),
                )
                action_ids = torch.where(
                    is_action_token & (action_sum > 0),
                    action_ids,
                    torch.zeros_like(action_ids),
                )

                if self.use_cls_token:
                    zero_pad = torch.zeros(
                        (batch_size, 1),
                        device=x_ref.device,
                        dtype=seat_ids.dtype,
                    )
                    seat_ids = torch.cat([zero_pad, seat_ids], dim=1)
                    action_ids = torch.cat([zero_pad, action_ids], dim=1)

                phase_ids = torch.zeros(
                    (batch_size, orig_len),
                    device=x_ref.device,
                    dtype=torch.long,
                )
                if TIME_SECTION_IN_ACTION and TIME_SECTION_ROWS > 0:
                    phase_offset = (
                        NEW_ADD_PLAYER_ROWS
                        + NEW_ADD_ACTION_ROWS
                        + NEW_ADD_TILE_ROWS
                    )
                    phase_slice = x_raw[
                        :, :, phase_offset : phase_offset + TIME_SECTION_ROWS
                    ]
                    if TIME_SECTION_ROWS == 1:
                        phase_vals = phase_slice.squeeze(-1)
                        early_mask = phase_vals < 0.25
                        mid_mask = (phase_vals >= 0.25) & (phase_vals < 0.75)
                        late_mask = phase_vals >= 0.75
                        phase_ids = torch.where(
                            is_action_token & early_mask,
                            torch.ones_like(phase_ids),
                            phase_ids,
                        )
                        phase_ids = torch.where(
                            is_action_token & mid_mask,
                            torch.full_like(phase_ids, 2),
                            phase_ids,
                        )
                        phase_ids = torch.where(
                            is_action_token & late_mask,
                            torch.full_like(phase_ids, 3),
                            phase_ids,
                        )
                    else:
                        phase_sum = phase_slice.sum(dim=-1)
                        phase_argmax = phase_slice.argmax(dim=-1) + 1
                        phase_ids = torch.where(
                            is_action_token & (phase_sum > 0),
                            phase_argmax,
                            phase_ids,
                        )

                if self.use_cls_token:
                    zero_pad = torch.zeros(
                        (batch_size, 1),
                        device=x_ref.device,
                        dtype=phase_ids.dtype,
                    )
                    phase_ids = torch.cat([zero_pad, phase_ids], dim=1)
                else:
                    # Align to seq_len when沒有 CLS
                    pass

                seat_emb = self.seat_proj(self.seat_embed(seat_ids))
                action_emb = self.action_proj(self.action_embed(action_ids))
                phase_emb = self.phase_proj(self.phase_embed(phase_ids))
                if token_mask is not None:
                    seat_emb = seat_emb * token_mask.unsqueeze(-1)
                    action_emb = action_emb * token_mask.unsqueeze(-1)
                    phase_emb = phase_emb * token_mask.unsqueeze(-1)
                emb_acc = emb_acc + seat_emb + action_emb + phase_emb

            elif (
                self.use_meta_emb
                and self.token_mode != "add"
                and (INPUT_FORMAT_IS_NEW_MULTIPLY or INPUT_FORMAT_IS_NEW_ADD)
            ):
                logger.debug(
                    "Seat/action/phase embeddings are enabled but token mode is not 'add'; skipping meta embeddings."
                )

            if isinstance(x, tuple):
                x = (x[0] + emb_acc, x[1] + emb_acc)
            else:
                x = x + emb_acc

        # positional encoding
        if self.use_abs_pos:
            if isinstance(x, tuple):
                x = (self.pos_encoding(x[0]), self.pos_encoding(x[1]))
            else:
                x = self.pos_encoding(x)

        # 準備 attention mask
        if attention_mask is not None:
            x_dev = x[0].device if isinstance(x, tuple) else x.device
            attention_mask = attention_mask.to(x_dev, dtype=torch.bool)
            attn = attention_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
        else:
            attn = None

        if self.transformer_layers and getattr(
            self.transformer_layers[0], "split_keep", False
        ):
            x_pair = x if isinstance(x, tuple) else (x, x)
            for transformer_layer in self.transformer_layers:
                x_pair = transformer_layer(
                    x_pair, mask=attn, value_modulators=value_modulators
                )

            # split_cv: compute per-token auxiliary loss before merging
            # branches
            if self.split_cv:
                if (
                    self.training
                    and _cv_at_labels is not None
                    and _cv_df_labels is not None
                ):
                    x_at, x_df = x_pair
                    action_start_idx = 2 if self.use_cls_token else 1
                    n_act_enc = x_at.size(1) - action_start_idx
                    if n_act_enc > 0:
                        aux_at_logits = self.aux_attack_head(
                            x_at[:, action_start_idx:, :]
                        ).squeeze(-1)
                        aux_df_logits = self.aux_defense_head(
                            x_df[:, action_start_idx:, :]
                        ).squeeze(-1)
                        # Align lengths (action labels vs encoder output)
                        n_aux = min(
                            aux_at_logits.size(1), _cv_at_labels.size(1)
                        )
                        aux_at_logits = aux_at_logits[:, :n_aux]
                        aux_df_logits = aux_df_logits[:, :n_aux]
                        at_tgt = _cv_at_labels[:, :n_aux].to(
                            device=aux_at_logits.device,
                            dtype=aux_at_logits.dtype,
                        )
                        df_tgt = _cv_df_labels[:, :n_aux].to(
                            device=aux_df_logits.device,
                            dtype=aux_df_logits.dtype,
                        )
                        # Mask valid action tokens only
                        if attention_mask is not None:
                            act_mask = attention_mask[
                                :, action_start_idx : action_start_idx + n_aux
                            ].bool()
                        else:
                            act_mask = torch.ones(
                                aux_at_logits.shape,
                                device=aux_at_logits.device,
                                dtype=torch.bool,
                            )
                        if act_mask.any():
                            at_loss = F.binary_cross_entropy_with_logits(
                                aux_at_logits[act_mask], at_tgt[act_mask]
                            )
                            df_loss = F.binary_cross_entropy_with_logits(
                                aux_df_logits[act_mask], df_tgt[act_mask]
                            )
                            self._auxiliary_loss = (
                                SPLIT_CV_LAMBDA_AT * at_loss
                                + SPLIT_CV_LAMBDA_DF * df_loss
                            )
                        else:
                            self._auxiliary_loss = 0.0
                    else:
                        self._auxiliary_loss = 0.0
                else:
                    self._auxiliary_loss = 0.0

                # V-level auxiliary supervision: probe raw V projections at each layer
                # This directly guides w_v_at / w_v_df to project tokens into
                # attack / defense semantic subspaces (like "blue" modifying
                # "creature")
                if (
                    self.training
                    and _cv_at_labels is not None
                    and _cv_df_labels is not None
                ):
                    _v_level_total = torch.tensor(0.0, device=x_pair[0].device)
                    _n_v_layers = 0
                    _v_act_start = 2 if self.use_cls_token else 1
                    for _layer in self.transformer_layers:
                        _attn_mod = _layer.attention
                        _vr_at = getattr(_attn_mod, "_last_v_at_raw", None)
                        _vr_df = getattr(_attn_mod, "_last_v_df_raw", None)
                        if _vr_at is not None and _vr_df is not None:
                            _nv_act = _vr_at.size(1) - _v_act_start
                            if _nv_act > 0:
                                _vp_at = self.v_aux_attack_probe(
                                    _vr_at[:, _v_act_start:, :]
                                ).squeeze(-1)
                                _vp_df = self.v_aux_defense_probe(
                                    _vr_df[:, _v_act_start:, :]
                                ).squeeze(-1)
                                _nv = min(
                                    _vp_at.size(1), _cv_at_labels.size(1)
                                )
                                _vt_at = _cv_at_labels[:, :_nv].to(
                                    device=_vp_at.device, dtype=_vp_at.dtype
                                )
                                _vt_df = _cv_df_labels[:, :_nv].to(
                                    device=_vp_df.device, dtype=_vp_df.dtype
                                )
                                if attention_mask is not None:
                                    _vmask = attention_mask[
                                        :, _v_act_start : _v_act_start + _nv
                                    ].bool()
                                else:
                                    _vmask = torch.ones(
                                        (_vp_at.size(0), _nv),
                                        device=_vp_at.device,
                                        dtype=torch.bool,
                                    )
                                if _vmask.any():
                                    _v_level_total = (
                                        _v_level_total
                                        + F.binary_cross_entropy_with_logits(
                                            _vp_at[:, :_nv][_vmask],
                                            _vt_at[_vmask],
                                        )
                                        + F.binary_cross_entropy_with_logits(
                                            _vp_df[:, :_nv][_vmask],
                                            _vt_df[_vmask],
                                        )
                                    )
                                    _n_v_layers += 1
                            # Free stored V projections
                            _attn_mod._last_v_at_raw = None
                            _attn_mod._last_v_df_raw = None
                    if _n_v_layers > 0:
                        self._auxiliary_loss = (
                            self._auxiliary_loss
                            + SPLIT_CV_LAMBDA_V
                            * (_v_level_total / _n_v_layers)
                        )

            if (
                self._split_keep_merge == "concat"
                and self.branch_merge_proj is not None
            ):
                x = self.branch_merge_proj(
                    torch.cat([x_pair[0], x_pair[1]], dim=-1)
                )
            else:
                x = x_pair[0] + x_pair[1]
        else:
            for transformer_layer in self.transformer_layers:
                x = transformer_layer(
                    x, mask=attn, value_modulators=value_modulators
                )

        # ---- Pooling strategy ----
        if self.head_mode_str == "cnn2d":
            # CNN2D head handles masking + classification internally
            length_repr = self._masked_mean_representation(x, attention_mask)
            self._maybe_add_length_grl_loss(
                length_repr,
                length_attention_mask,
                fallback_length=seq_len,
            )
            tenpai_logits = self.cnn2d_head(x, attention_mask=attention_mask)
            return tenpai_logits
        elif self.head_mode_str == "flinear":
            # Flatten entire sequence (pad-included) to (B, seq_len * d_model)
            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1)  # (B,T,1)
                x = x * m  # zero out padded positions
            pooled = x.reshape(x.size(0), -1)  # (B, T*d_model)
            # Pad/truncate to expected input size of tenpai_classifier
            expected = self.tenpai_classifier.in_features
            if pooled.size(1) < expected:
                pooled = F.pad(pooled, (0, expected - pooled.size(1)))
            elif pooled.size(1) > expected:
                pooled = pooled[:, :expected]
        elif self.head_mode_str == "multi_pool_flinear":
            pooled = self._multi_pool_flinear_representation(
                x, attention_mask
            )
        elif self.head_mode_str == "first_token":
            # Directly take init token (position 0) — no pooling
            pooled = x[:, 0, :]
        elif self.head_mode_str == "last_token":
            # Take last valid token per sample
            if attention_mask is not None:
                mask_bool = attention_mask.bool()
                positions = torch.arange(
                    mask_bool.size(1), device=mask_bool.device
                ).unsqueeze(0)
                last_idx = torch.where(
                    mask_bool,
                    positions,
                    torch.zeros_like(positions),
                ).amax(dim=1)
                pooled = x[torch.arange(x.size(0), device=x.device), last_idx]
            else:
                pooled = x[:, -1, :]
        elif self.use_cls_token:
            pooled = x[:, 0, :]
        elif self.use_learnable_pool:
            scores = torch.matmul(x, self.pool_query)
            if attention_mask is not None:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        else:
            # "pooler" and "linear" both use masked mean pooling
            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1)  # (B,T,1)
                pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                pooled = x.mean(dim=1)

        # ---- Head ----
        if self.head_mode_str == "pooler":
            pooled = self.pooler_dropout(self.pooler_act(self.pooler(pooled)))
        self._maybe_add_length_grl_loss(
            pooled,
            length_attention_mask,
            fallback_length=seq_len,
        )
        tenpai_logits = self.tenpai_classifier(pooled)

        # MTP Pre-training mode: return per-token tile predictions instead of
        # classification
        if self.mtp_pretraining and mtp_mask_positions is not None:
            # x shape: (B, seq_len, d_model) from after encoder
            mtp_hidden = self.mtp_head[1](self.mtp_head[0](x))
            mtp_hidden = _apply_sequence_norm(
                self.mtp_head[2], mtp_hidden, attention_mask
            )
            mtp_logits = self.mtp_head[3](mtp_hidden)
            # Slice to exclude CLS token to match label seq_len
            # Note: heter_tokens processing doesn't add tokens - it processes existing tokens differently
            # Only CLS token adds an extra position at the front
            if self.use_cls_token:
                mtp_logits = mtp_logits[:, 1:, :]  # (B, original_seq_len, 34)
            if self.mtp_use_aux_cls:
                # Return both MTP logits and classification logits for joint
                # training
                return mtp_logits, tenpai_logits
            return mtp_logits, None

        return tenpai_logits

        # if x.dim() == 2:
        #     x = x.unsqueeze(1)
        # elif x.dim() != 3:
        #     raise ValueError(f"MahjongTransformer expected 2D or 3D input, got shape {tuple(x.shape)}")

        # batch_size, seq_len, features = x.shape
        # if features != self.input_features:
        #     raise ValueError(
        #         f"Input feature dim mismatch: expected {self.input_features}, got {features}"
        #     )

        # if attention_mask is not None:
        #     # 轉換為 (batch, 1, 1, seq_len)，1 表示有效 token
        #     attn = attention_mask.to(x.device)
        #     attn = attn.unsqueeze(1).unsqueeze(2)
        # else:
        #     attn = None

        # x = self.input_projection(x)

        # x = self.pos_encoding(x)

        # for transformer_layer in self.transformer_layers:
        #     x = transformer_layer(x, mask=attn)

        # x = x.mean(dim=1)
        # tenpai_logits = self.tenpai_classifier(x)

        # return tenpai_logits

        # total_needed = self.F_INIT + self.F_ACT
        # if x.dim() == 2:
        #     x = x.unsqueeze(1)
        # elif x.dim() != 3:
        #     raise ValueError(f"MahjongTransformer expected 2D or 3D input, got shape {tuple(x.shape)}")

        # batch_size, seq_len, features = x.shape

        # if features == total_needed or features == total_needed + 1:
        #     offset = 1 if features == total_needed + 1 else 0
        #     x_init = x[..., offset:offset + self.F_INIT]
        #     x_act  = x[..., offset + self.F_INIT:offset + total_needed]
        # elif features == TOTAL_FEATURE_ROWS * 34:
        #     x4 = x.view(batch_size, seq_len, TOTAL_FEATURE_ROWS, 34)
        #     x_init = x4[..., :STATIC_FEATURE_ROWS, :].reshape(batch_size, seq_len, self.F_INIT)  # 25×34
        #     x_act  = x4[..., BITSET_OFFSET:BITSET_OFFSET + self.ACTION_FEATURE_ROWS, :] \
        #             .reshape(batch_size, seq_len, self.F_ACT)                                  # 36×34（可調）
        # else:
        #     raise ValueError(
        #         f"Unsupported feature dim: got {features}, expected {total_needed} (packed) "
        #         f"or {TOTAL_FEATURE_ROWS*34} (full)"
        #     )

        # # 兩條線性路徑後相加（initial 位置的 x_act 會是 0；action 位置的 x_init 會是 0）
        # x = self.proj_init(x_init) + self.proj_act(x_act)

        # x = self.pos_encoding(x)
        # if attention_mask is not None:
        #     attn = attention_mask.to(x.device).unsqueeze(1).unsqueeze(2)
        # else:
        #     attn = None

        # for layer in self.transformer_layers:
        #     x = layer(x, mask=attn)

        # # --- masked mean pooling（避免 padding 汙染）---
        # if attention_mask is not None:
        #     m = attention_mask.to(x.device).unsqueeze(-1)  # (B,T,1), 1=有效
        #     x_pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        # else:
        #     x_pooled = x.mean(dim=1)

        # tenpai_logits = self.tenpai_classifier(x_pooled)
        # return tenpai_logits


# ==============================================================================
# Conformer (Transformer + Conv) Model
# ==============================================================================
class MahjongConformer(MahjongTransformer):
    """Drop-in replacement for MahjongTransformer that swaps EncoderBlocks with ConformerBlocks.

    Inherits *all* input projection, embedding, pooling, and head logic from
    MahjongTransformer.  The only structural change is in the encoder stack.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (
            MAX_SEQ_LEN_FALLBACK,
            ACTION_FEATURE_DIM,
        ),
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 2,
        conv_kernel_size: int = 31,
        conv_expansion_factor: int = 2,
        norm: str = "hybrid",
    ):
        # Parent __init__ builds everything including self.transformer_layers (EncoderBlocks).
        # We then *replace* that module list with ConformerBlocks.
        super().__init__(
            input_shape=input_shape,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            num_classes=num_classes,
            norm=norm,
        )
        # Overwrite encoder stack ─ ConformerBlock has the same forward(x,
        # mask, value_modulators) API
        self.transformer_layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                    conv_expansion_factor=conv_expansion_factor,
                    norm=norm,
                )
                for _ in range(n_layers)
            ]
        )
        logger.info(
            f"MahjongConformer: replaced {n_layers} EncoderBlocks with ConformerBlocks "
            f"(kernel={conv_kernel_size}, expansion={conv_expansion_factor}, "
            f"normalization={norm})"
        )
def TransformerSmall(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "ln",
):
    """小型 Transformer with 4 layers (輕量級版本)"""
    return MahjongTransformer(
        input_shape=input_shape,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        dropout=dropout_rate,
        num_classes=num_classes,
        norm=norm,
    )


def TransformerMedium(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "ln",
):
    """中型 Transformer with 6 layers (平衡版本)"""
    return MahjongTransformer(
        input_shape=input_shape,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=dropout_rate,
        num_classes=num_classes,
        norm=norm,
    )


def TransformerLarge(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "ln",
):
    """大型 Transformer with 8 layers (強力版本)"""
    return MahjongTransformer(
        input_shape=input_shape,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        dropout=dropout_rate,
        num_classes=num_classes,
        norm=norm,
    )


def TransformerXL(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "ln",
):
    """超大型 Transformer with 12 layers (最強版本)"""
    return MahjongTransformer(
        input_shape=input_shape,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        dropout=dropout_rate,
        num_classes=num_classes,
        norm=norm,
    )


def TransformerTiny(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "ln",
):
    """微型 Transformer with 2 layers (測試版本)"""
    return MahjongTransformer(
        input_shape=input_shape,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        dropout=dropout_rate,
        num_classes=num_classes,
        norm=norm,
    )


def TransformerBERTBASE(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "ln",
):
    """BERT Base Model equivalent Transformer"""
    return MahjongTransformer(
        input_shape=input_shape,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        dropout=dropout_rate,
        num_classes=num_classes,
        norm=norm,
    )


def TransformerBERTLARGE(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "ln",
):
    """BERT Large Model equivalent Transformer"""
    return MahjongTransformer(
        input_shape=input_shape,
        d_model=1024,
        n_heads=16,
        n_layers=24,
        d_ff=4096,
        dropout=dropout_rate,
        num_classes=num_classes,
        norm=norm,
    )


def TransformerORI(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    old_single_encoder_ablation: str = OLD_SINGLE_ENCODER_ABLATION,
    norm: str = "ln",
):
    """BERT Large Model equivalent Transformer"""
    return MahjongTransformer(
        input_shape=input_shape,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=dropout_rate,
        num_classes=num_classes,
        old_single_encoder_ablation=old_single_encoder_ablation,
        norm=norm,
    )


# ==============================================================================
# Variational Oracle Guiding (VOG) — Figure 1 dual-encoder architecture
# ==============================================================================


class _VOGEncoderPath(nn.Module):
    """One encoder branch for MahjongVOG.

    Supports the New_Add heterogeneous token format (init token + action tokens).
    Input projection weights are completely independent between prior and posterior paths.

    forward() returns a flat vector of shape (B, seq_len × d_model) ready for z projection.
    The sequence length is clamped/padded to the value fixed at construction time so the
    downstream Linear always receives a constant-width tensor.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        f_init: int,
        f_act: int,
        norm: str = "ln",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.flat_dim = seq_len * d_model
        self.act_feature_dim = int(f_act)

        self.proj_init = nn.Linear(f_init, d_model, bias=False)
        self.proj_act = (
            nn.Linear(f_act, d_model, bias=False) if f_act > 0 else None
        )

        if USE_ABS_POS_ENCODING:
            self.pos_encoding: Optional[nn.Module] = PositionalEncoding(
                d_model, max_len=seq_len, dropout=dropout
            )
        else:
            self.pos_encoding = None

        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(d_model, n_heads, d_ff, dropout, norm=norm)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        init_sidecar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, feature_dim) in New_Add format — position 0 is init token,
               positions 1..seq_len-1 are action tokens.
            attention_mask: optional (B, seq_len) bool mask.
        Returns:
            flat: (B, self.flat_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.size(0)

        # — project init and action tokens separately —
        if init_sidecar is not None and init_sidecar.numel() > 0:
            if (
                init_sidecar.dim() != 2
                or init_sidecar.size(0) != x.size(0)
                or init_sidecar.size(1) != NEW_ADD_INIT_SIDECAR_DIM
            ):
                raise ValueError(
                    "Invalid VOG init sidecar shape: "
                    f"expected ({x.size(0)}, {NEW_ADD_INIT_SIDECAR_DIM}), "
                    f"got {tuple(init_sidecar.shape)}"
                )
            x_init = torch.cat(
                [x[:, 0, :F_INIT_COMPACT], init_sidecar], dim=-1
            )
        else:
            x_init = x[:, 0, : self.proj_init.in_features]

        x_init_proj = self.proj_init(x_init)  # (B, d_model)
        if self.proj_act is not None:
            act_dim = self.proj_act.in_features
            x_act = x[:, 1:, :act_dim]
            # (B, act_seq_len, d_model)
            x_act_proj = self.proj_act(x_act)
            # (B, 1+act_seq_len, d_model)
            x_enc = torch.cat([x_init_proj.unsqueeze(1), x_act_proj], dim=1)
        else:
            # old_single mode: treat the full snapshot as a single token.
            x_enc = x_init_proj.unsqueeze(1)

        if self.pos_encoding is not None:
            x_enc = self.pos_encoding(x_enc)

        # build attention mask for encoder blocks
        if attention_mask is not None:
            attn = (
                attention_mask.to(x_enc.device, dtype=torch.bool)
                .unsqueeze(1)
                .unsqueeze(2)
            )
        else:
            attn = None

        for layer in self.encoder_layers:
            x_enc = layer(x_enc, mask=attn)

        # — flatten to fixed size —
        cur_len = x_enc.size(1)
        if cur_len < self.seq_len:
            pad = torch.zeros(
                batch_size,
                self.seq_len - cur_len,
                self.d_model,
                device=x_enc.device,
                dtype=x_enc.dtype,
            )
            x_enc = torch.cat([x_enc, pad], dim=1)
        elif cur_len > self.seq_len:
            x_enc = x_enc[:, : self.seq_len, :]

        # apply attention-mask zeroing before flatten
        if attention_mask is not None:
            am = attention_mask.bool()
            if am.size(1) < self.seq_len:
                pad_am = torch.zeros(
                    batch_size,
                    self.seq_len - am.size(1),
                    device=am.device,
                    dtype=am.dtype,
                )
                am = torch.cat([am, pad_am], dim=1)
            elif am.size(1) > self.seq_len:
                am = am[:, : self.seq_len]
            x_enc = x_enc * am.unsqueeze(-1)

        return x_enc.reshape(batch_size, -1)  # (B, flat_dim)


class MahjongVOG(nn.Module):
    """Variational Oracle Guiding Transformer — Figure 1 of the VOG paper.

    Two completely independent encoder paths:
            prior   : receives masked input  (oracle features zeroed)
            posterior: receives full oracle input (oracle features visible)

        Supports both New_Add (sequence tokens) and old_single (single-token
        snapshot) input formats.

    During training, if x_oracle is provided the posterior path runs and
    KL(N(μ_post,σ_post) || N(μ_prior,σ_prior)) is added to self._auxiliary_loss.
    At evaluation/inference only the prior path runs (x_oracle=None).

    Architecture:
        input (masked)   → prior_encoder  → flatten → prior_z_proj  → (μ_p, logσ_p) → z → classifier → logits
        input (oracle)   → post_encoder   → flatten → post_z_proj   → (μ_q, logσ_q)
                                                             ↓
                                        KL(q || p) added to self._auxiliary_loss
    """

    is_vog: bool = True  # marker consumed by train_epoch

    def __init__(
        self,
        input_shape: Tuple[int, int] = (
            MAX_SEQ_LEN_FALLBACK,
            ACTION_FEATURE_DIM,
        ),
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 2,
        kl_weight: float = 1.0,
        norm: str = "ln",
    ):
        super().__init__()
        self.supports_init_sidecar = True
        from utils.mahjong_utils import (
            F_INIT,
            F_ACT_ADD_DIM as _F_ACT_ADD,
            INPUT_FORMAT_IS_NEW_ADD as _IS_NEW_ADD,
            INPUT_FORMAT_IS_OLD_SINGLE as _IS_OLD_SINGLE,
        )

        if not (_IS_NEW_ADD or _IS_OLD_SINGLE):
            raise ValueError(
                "MahjongVOG only supports USE_OLD_INPUT_FORMAT='New_Add' "
                "or 'old_single'. "
                f"Got INPUT_FORMAT_IS_NEW_ADD={_IS_NEW_ADD}, "
                f"INPUT_FORMAT_IS_OLD_SINGLE={_IS_OLD_SINGLE}."
            )

        self.seq_len = input_shape[0]
        self.d_model = d_model
        self.normalization = norm
        self.kl_weight = kl_weight
        self._auxiliary_loss: Any = 0.0

        flat_dim = self.seq_len * d_model
        z_dim = d_model
        self.z_dim = z_dim
        _requested_out = int(num_classes)
        if _requested_out > 2:
            _num_out = _requested_out
        else:
            _num_out = 2 if SOFTMAX_LOSS else 1

        self.prior_encoder = _VOGEncoderPath(
            seq_len=self.seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            f_init=F_INIT if _IS_NEW_ADD else input_shape[1],
            f_act=_F_ACT_ADD if _IS_NEW_ADD else 0,
            norm=norm,
        )
        self.post_encoder = _VOGEncoderPath(
            seq_len=self.seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            f_init=F_INIT if _IS_NEW_ADD else input_shape[1],
            f_act=_F_ACT_ADD if _IS_NEW_ADD else 0,
            norm=norm,
        )

        # Each z_proj outputs (μ, logvar) concatenated → size z_dim * 2
        self.prior_z_proj = nn.Linear(flat_dim, z_dim * 2)
        self.post_z_proj = nn.Linear(flat_dim, z_dim * 2)
        self.classifier = nn.Linear(z_dim, _num_out)

        # Zero-initialise the logvar heads so both branches start at logvar=0
        # (unit variance). This keeps the initial KL near zero and prevents the
        # /tiny-var explosion that occurs when logvar collapses at startup.
        with torch.no_grad():
            self.prior_z_proj.weight[z_dim:].zero_()
            self.prior_z_proj.bias[z_dim:].zero_()
            self.post_z_proj.weight[z_dim:].zero_()
            self.post_z_proj.bias[z_dim:].zero_()

        logger.info(
            f"MahjongVOG initialized: d_model={d_model}, n_heads={n_heads}, "
            f"n_layers={n_layers}, z_dim={z_dim}, kl_weight={kl_weight}, "
            f"flat_dim={flat_dim}, normalization={norm}"
        )

    @staticmethod
    def _reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε·σ  (ε ~ N(0,I)).
        logvar is clamped to [-4, 2] for training stability (tight range keeps
        std in [~0.14, ~2.72] and prevents extreme KL values).
        At inference (no grad) returns μ directly."""
        logvar = logvar.clamp(-4.0, 2.0)
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_divergence(
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> torch.Tensor:
        """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)) averaged over batch.

        Closed-form:
            KL = 0.5 * Σ_d [ logvar_p - logvar_q
                              + (exp(logvar_q) + (mu_q - mu_p)^2) / exp(logvar_p)
                              - 1 ]

        Stability measures:
          - logvar clamped to [-4, 2]  → std in [~0.14, 2.72], no catastrophic precision
          - var_p floored at 0.1       → prevents division-by-near-zero explosion
          - mu clamped to [-10, 10]    → prevents huge (mu_q-mu_p)^2 terms
          - reduction: sum over z_dim then mean over batch, consistent with train_epoch /z_dim
        """
        mu_q = mu_q.clamp(-10.0, 10.0)
        mu_p = mu_p.clamp(-10.0, 10.0)
        logvar_q = logvar_q.clamp(-4.0, 2.0)
        logvar_p = logvar_p.clamp(-4.0, 2.0)
        var_p = logvar_p.exp().clamp(min=0.1)  # prevent tiny-variance blowup
        kl = 0.5 * (
            logvar_p
            - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / var_p
            - 1.0
        )
        return kl.sum(dim=-1).mean()  # sum over z_dim, mean over batch

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        x_oracle: Optional[torch.Tensor] = None,
        init_sidecar: Optional[torch.Tensor] = None,
        init_sidecar_oracle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: masked input (oracle features zeroed), shape (B, seq_len, feat_dim)
            attention_mask: optional (B, seq_len) bool mask
            x_oracle: full oracle input; if provided AND training, runs posterior and computes KL
        Returns:
            logits: (B, num_out)
        """
        # — Prior path (always runs) —
        prior_flat = self.prior_encoder(
            x,
            attention_mask=attention_mask,
            init_sidecar=init_sidecar,
        )
        prior_params = self.prior_z_proj(prior_flat)
        mu_prior, logvar_prior = prior_params.chunk(2, dim=-1)
        z = self._reparameterize(mu_prior, logvar_prior)
        logits = self.classifier(z)

        # — Posterior path (training only, when oracle input is provided) —
        if self.training and x_oracle is not None:
            post_flat = self.post_encoder(
                x_oracle,
                attention_mask=attention_mask,
                init_sidecar=init_sidecar_oracle,
            )
            post_params = self.post_z_proj(post_flat)
            mu_post, logvar_post = post_params.chunk(2, dim=-1)
            kl = self._kl_divergence(
                mu_post, logvar_post, mu_prior, logvar_prior
            )
            self._auxiliary_loss = self.kl_weight * kl
        else:
            self._auxiliary_loss = 0.0

        return logits


def TransformerVOG(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "ln",
) -> "MahjongVOG":
    """Variational Oracle Guiding Transformer — TransformerORI 規格 (d_model=512, 8 heads, 6 layers)."""
    return MahjongVOG(
        input_shape=input_shape,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=dropout_rate,
        num_classes=num_classes,
        kl_weight=VOG_KL_WEIGHT,
        norm=norm,
    )


# ==============================================================================
# Conformer factory functions
# ==============================================================================
def ConformerTiny(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "hybrid",
):
    """微型 Conformer with 2 layers (測試版本)"""
    return MahjongConformer(
        input_shape=input_shape,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        dropout=dropout_rate,
        num_classes=num_classes,
        conv_kernel_size=15,
        norm=norm,
    )
def ConformerSmall(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "hybrid",
):
    """小型 Conformer with 4 layers"""
    return MahjongConformer(
        input_shape=input_shape,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        dropout=dropout_rate,
        num_classes=num_classes,
        conv_kernel_size=15,
        norm=norm,
    )


def ConformerORI(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "hybrid",
):
    """Conformer — 與 TransformerORI 同規格 (d_model=512, 6 layers)"""
    return MahjongConformer(
        input_shape=input_shape,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=dropout_rate,
        num_classes=num_classes,
        conv_kernel_size=31,
        norm=norm,
    )


def ConformerLarge(
    input_shape: Tuple[int, int] = (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM),
    dropout_rate: float = 0.1,
    num_classes: int = 2,
    norm: str = "hybrid",
):
    """大型 Conformer with 8 layers"""
    return MahjongConformer(
        input_shape=input_shape,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        dropout=dropout_rate,
        num_classes=num_classes,
        conv_kernel_size=31,
        norm=norm,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
