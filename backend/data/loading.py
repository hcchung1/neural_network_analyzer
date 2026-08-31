"""CSV/binary parsing, datasets, mmap indexes, and data samplers."""

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
from config import *
from data.schema import *
from data.features import *

class CsvMahjongDataset(Dataset):
    """
    一個高效的 Dataset，直接從 CSV 檔案讀取資料。
    它會在初始化時將 CSV 載入到一個 pandas DataFrame，
    然後在 __getitem__ 中即時解析每一行。
    """

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV data file not found at: {csv_path}")

        logger.info(f"Loading CSV data into memory from: {csv_path} ...")
        # 為了優化記憶體，只讀取需要的欄位，並指定較小的資料類型
        # 您可以根據您的 CSV 欄位來調整 dtypes
        dtypes = {
            "label": "int8",
            "player_id": "int8",
            "round": "int8",
            "honba": "int8",
            "score1": "int32",
            "score2": "int32",
            "score3": "int32",
            "score4": "int32",
        }
        # low_memory=False 可以稍微加速讀取，但會用更多記憶體，可以自行取捨
        self.df = pd.read_csv(csv_path, dtype=dtypes, low_memory=False)
        logger.info(
            f"Successfully loaded {len(self.df)} records into DataFrame."
        )

        self.df = self.df.reset_index(drop=True)
        invalid_indices: List[int] = []
        fallback_sample: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        for row_idx in range(len(self.df)):
            row = self.df.iloc[row_idx]
            try:
                features, label = parse_row_to_features(row)
                if fallback_sample is None:
                    fallback_sample = (features.clone(), label.clone())
            except Exception as exc:
                invalid_indices.append(row_idx)
                logger.warning(
                    f"[CsvMahjongDataset] Drop row {row_idx}: {exc}"
                )

        if invalid_indices:
            self.df = self.df.drop(index=invalid_indices).reset_index(
                drop=True
            )
            logger.info(
                f"[CsvMahjongDataset] Filtered out {len(invalid_indices)} invalid rows; remaining {len(self.df)} samples."
            )

        if fallback_sample is None:
            raise RuntimeError(
                "CsvMahjongDataset contains no valid rows after filtering."
            )

        self._fallback_feature, self._fallback_label = fallback_sample

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        try:
            return parse_row_to_features(row)
        except Exception as e:
            logger.error(
                f"[CsvMahjongDataset] Fallback on row {idx}: {e}. Returning cached valid sample."
            )
            return self._fallback_feature.clone(), self._fallback_label.clone()


class BinaryDataset(Dataset):
    """ """

    def __init__(self, binary_path: str, preload_to_memory: bool = True):
        if not os.path.exists(binary_path):
            raise FileNotFoundError(
                f"Binary data file not found at: {binary_path}"
            )

        (
            self.features,
            self.labels,
            self.line_numbers,
            self.seq_lengths,
            self.player_ids,
            self.pred_ids,
        ) = read_binary_data(binary_path)

        if self.features.size == 0:
            self.attention_masks = np.zeros((0, 1), dtype=np.float32)
        else:
            seq_len = self.features.shape[1]
            time_axis = np.arange(seq_len, dtype=np.int32)
            self.attention_masks = (
                time_axis[None, :] < self.seq_lengths[:, None]
            ).astype(np.float32)
            if self.attention_masks.size > 0:
                empty_mask = self.attention_masks.sum(axis=1) == 0
                if np.any(empty_mask):
                    self.attention_masks[empty_mask, 0] = 1.0
                    if (
                        self.seq_lengths is not None
                        and self.seq_lengths.size > 0
                    ):
                        self.seq_lengths[empty_mask] = 1

    # logger.info(f"Successfully loaded {len(self.labels)} samples from binary file.")
    # transpose features from (batch size, 34, TOTAL_FEATURE_ROWS) to (batch size, TOTAL_FEATURE_ROWS, 34)
    # self.features = self.features.transpose(0, 2, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(
            self.features[idx].copy()
        )  # copy() 避免共享内存问题
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        line_num = int(self.line_numbers[idx])
        mask = torch.from_numpy(self.attention_masks[idx].copy())
        meta = torch.tensor(
            [self.player_ids[idx], self.pred_ids[idx]], dtype=torch.int64
        )
        return x, y, line_num, mask, meta

    def get_feature_dimension(self) -> Tuple[int, int]:
        """返回特徵維度 (height, width)"""
        if len(self.features) > 0:
            return self.features[0].shape
        return (1, TOKEN_FEATURE_DIM)  # 預設維度


def parse_bitset_to_features(bitset_bytes: bytes) -> np.ndarray:
    """
    Transform a 8-byte bitset into a numpy array of 34 features.
    Optimized with vectorized bit operations.
    """
    if len(bitset_bytes) != 8:
        # 填充或截斷到 8 字節
        if len(bitset_bytes) < 8:
            bitset_bytes = bitset_bytes + b"\x00" * (8 - len(bitset_bytes))
        else:
            bitset_bytes = bitset_bytes[:8]

    try:
        value = struct.unpack("<Q", bitset_bytes)[0]
    except struct.error:
        return np.zeros(34, dtype=np.int32)

    # Vectorized: generate all bit indices at once
    return ((value >> np.arange(34, dtype=np.uint64)) & 1).astype(np.int32)


def _decode_bit_matrix(bitset_bytes: bytes, row_count: int) -> np.ndarray:
    """Decode packed uint64 bitset rows into a (row_count, 34) matrix.
    Optimized with vectorized numpy operations.
    """
    matrix = np.zeros((row_count, FEAT_COLS), dtype=np.int32)
    if row_count <= 0 or not bitset_bytes:
        return matrix

    total_bits = row_count * FEAT_COLS
    expected_words = (total_bits + 63) // 64
    usable_bytes = min(len(bitset_bytes), expected_words * 8)
    word_count = usable_bytes // 8
    if word_count == 0:
        return matrix

    max_bits = min(total_bits, word_count * 64)

    raw_bytes = np.frombuffer(bitset_bytes[: word_count * 8], dtype=np.uint8)
    try:
        bits = np.unpackbits(raw_bytes, bitorder="little")
    except TypeError:
        bits = np.unpackbits(raw_bytes)
        bits = bits.reshape(-1, 8)[:, ::-1].reshape(-1)
    bits = bits[:max_bits]

    matrix.reshape(-1)[:max_bits] = bits.astype(np.int32, copy=False)

    return matrix


_BINARY_RECORD_HEADER_FMT = "<I3b3b4i5b"
_BINARY_RECORD_HEADER_SIZE = struct.calcsize(_BINARY_RECORD_HEADER_FMT)
_BINARY_RECORD_BITSET_BYTES = 1312
_TURN_DISCARD_ROW_INDICES = np.array(
    [
        *MY_DISCARD_ROWS,
        *OPP_DISCARD_ROWS[0],
        *OPP_DISCARD_ROWS[1],
        *OPP_DISCARD_ROWS[2],
    ],
    dtype=np.int32,
)


def _estimate_turn_number_from_bitset_bytes(bitset_bytes: bytes) -> int:
    """Estimate turn count directly from packed record bitsets."""
    if not bitset_bytes:
        return 0

    total_bits = BITSET_ROWS_TOTAL * FEAT_COLS
    max_bytes = (total_bits + 7) // 8
    raw_bytes = np.frombuffer(bitset_bytes[:max_bytes], dtype=np.uint8)
    if raw_bytes.size == 0:
        return 0

    try:
        bits = np.unpackbits(raw_bytes, bitorder="little")
    except TypeError:
        bits = np.unpackbits(raw_bytes)
        bits = bits.reshape(-1, 8)[:, ::-1].reshape(-1)

    if bits.size < total_bits:
        padded_bits = np.zeros(total_bits, dtype=np.uint8)
        padded_bits[: bits.size] = bits
        bits = padded_bits
    else:
        bits = bits[:total_bits]

    row_bits = bits.reshape(BITSET_ROWS_TOTAL, FEAT_COLS)
    discard_rows = row_bits[_TURN_DISCARD_ROW_INDICES]
    return int(np.count_nonzero(np.any(discard_rows != 0, axis=1)))


def _parse_record_index_metadata(
    record_data: bytes,
) -> Tuple[Tuple[int, int, int], int]:
    """Extract labels and turn metadata for index construction."""
    if len(record_data) < _BINARY_RECORD_HEADER_SIZE:
        return (0, 0, 0), 0

    try:
        unpacked = struct.unpack(
            _BINARY_RECORD_HEADER_FMT,
            record_data[:_BINARY_RECORD_HEADER_SIZE],
        )
    except struct.error:
        return (0, 0, 0), 0

    labels = (
        int(unpacked[1]),
        int(unpacked[2]),
        int(unpacked[3]),
    )
    bitset_start = _BINARY_RECORD_HEADER_SIZE
    bitset_end = bitset_start + _BINARY_RECORD_BITSET_BYTES
    if len(record_data) < bitset_end:
        return labels, 0

    turn_no = _estimate_turn_number_from_bitset_bytes(
        record_data[bitset_start:bitset_end]
    )
    return labels, turn_no


def parse_single_binary_record(
    record_data: bytes,
    record_id: int = 0,
    debug_output: bool = False,
    parse_tokens: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    Tuple[int, ...],
    int,
    int,
    Tuple[int, ...],
    Tuple[int, ...],
]:
    """
    Parse a single binary record into numpy arrays for each opponent.
    格式：單一大 bitset<10472>，包含308行×34列，記錄大小為1349字節（包含4字節 line_number）

    Args:
    record_data: 1349 bytes binary data representing a single record (4 bytes line_number + payload)
        record_id: record index for logging purposes
        debug_output: if True, will log detailed debug information
        parse_tokens: 是否解析 bitset 並生成完整序列 (預設 True)

    Returns:
    features: np.ndarray of shape (seq_len, TOTAL_FEATURE_ROWS, 34) token states per action
    labels: np.ndarray of shape (3,) - labels for each opponent
    line_number: int - the line number from info.csv for this record
    seq_len: int - actual sequence length for this record
    player_id: int - dealer/player seat id associated with record
    dora_ids: Tuple[int, ...] - 原始寶牌編號 (長度 5)
    action_type: int - 動作類型 (對應 Binary_record.hpp::ActionType 列舉)
    action_tile_count: int - 動作涉及的 tile 數量
    action_tiles: Tuple[int, ...] - 相關 tile id (長度 4)
    opp_shanten: Tuple[int, ...] - 其他三家向聽 (長度 3)
    """
    if len(record_data) < BINARY_RECORD_SIZE_BASE:
        logger.warning(
            f"Incomplete record data: {len(record_data)} bytes, expected at least {BINARY_RECORD_SIZE_BASE} bytes"
        )
        return (
            np.zeros((1, TOTAL_FEATURE_ROWS, 34), dtype=np.float32),
            np.zeros(3, dtype=np.int32),
            0,
            0,
            0,
            tuple([-1] * DORA_FEATURE_SLOTS),
            -1,
            0,
            tuple([-1] * MAX_ACTION_TILES),
            tuple([-1] * _OPPONENT_COUNT),
        )

    # Optimized: batch unpack all fixed header fields in one call
    # Format: line_number(I) + 3 labels(3b) + game_info(3b) + scores(4i) + doras(5b) + bitset_start_marker
    # Total: 4 + 3 + 3 + 16 + 5 = 31 bytes before bitset
    try:
        unpacked = struct.unpack(
            _BINARY_RECORD_HEADER_FMT,
            record_data[:_BINARY_RECORD_HEADER_SIZE],
        )
    except struct.error:
        logger.warning(f"Record {record_id}: failed to unpack header")
        return (
            np.zeros((1, TOTAL_FEATURE_ROWS, 34), dtype=np.float32),
            np.zeros(3, dtype=np.int32),
            0,
            0,
            0,
            tuple([-1] * DORA_FEATURE_SLOTS),
            -1,
            0,
            tuple([-1] * MAX_ACTION_TILES),
            tuple([-1] * _OPPONENT_COUNT),
        )

    # Extract all header fields
    line_number = unpacked[0]
    labeled_opp1, labeled_opp2, labeled_opp3 = unpacked[1:4]
    riichi_seen = (
        (labeled_opp1 == 5) or (labeled_opp2 == 5) or (labeled_opp3 == 5)
    )
    player_id, round_num, honba = unpacked[4:7]
    scores = np.array(unpacked[7:11], dtype=np.int32)
    dora_ids = unpacked[11:16]

    idx = _BINARY_RECORD_HEADER_SIZE

    # 位盤資料（164 個 uint64_t = 1312 bytes）
    if idx + _BINARY_RECORD_BITSET_BYTES > len(record_data):
        logger.warning(
            f"Record {record_id}: insufficient data for bitset, remaining bytes: {len(record_data) - idx}"
        )
        return (
            np.zeros((1, TOTAL_FEATURE_ROWS, 34), dtype=np.float32),
            np.zeros(3, dtype=np.int32),
            line_number,
            1,
            player_id,
            dora_ids,
            -1,
            0,
            tuple([-1] * MAX_ACTION_TILES),
            tuple([-1] * _OPPONENT_COUNT),
        )

    bitset_data = record_data[idx : idx + _BINARY_RECORD_BITSET_BYTES]
    idx += _BINARY_RECORD_BITSET_BYTES

    # 動作類型與相關牌 (6 bytes)
    action_type = struct.unpack("<b", record_data[idx : idx + 1])[0]
    idx += 1
    action_tile_count = struct.unpack("<b", record_data[idx : idx + 1])[0]
    idx += 1
    action_tiles = struct.unpack("<4b", record_data[idx : idx + 4])
    idx += 4

    record_size = len(record_data)
    has_shanten_payload = record_size in (
        BINARY_RECORD_SIZE_WITH_SHANTEN,
        BINARY_RECORD_SIZE_WITH_SHANTEN_AND_WAITS,
        BINARY_RECORD_SIZE_WITH_SHANTEN_AND_ORACLE,
        BINARY_RECORD_SIZE_WITH_SHANTEN_AND_WAITS_AND_ORACLE,
    )
    has_wait_payload = record_size in (
        BINARY_RECORD_SIZE_WITH_WAITS,
        BINARY_RECORD_SIZE_WITH_SHANTEN_AND_WAITS,
        BINARY_RECORD_SIZE_WITH_WAITS_AND_ORACLE,
        BINARY_RECORD_SIZE_WITH_SHANTEN_AND_WAITS_AND_ORACLE,
    )
    has_opp_hand_payload = record_size in (
        BINARY_RECORD_SIZE_WITH_ORACLE,
        BINARY_RECORD_SIZE_WITH_WAITS_AND_ORACLE,
        BINARY_RECORD_SIZE_WITH_SHANTEN_AND_ORACLE,
        BINARY_RECORD_SIZE_WITH_SHANTEN_AND_WAITS_AND_ORACLE,
    )
    remaining_bytes = len(record_data) - idx
    if (
        not has_shanten_payload
        and not has_wait_payload
        and not has_opp_hand_payload
    ):
        if remaining_bytes == OPP_SHANTEN_BYTES:
            has_shanten_payload = True
        elif remaining_bytes == OPP_WAIT_BYTES:
            has_wait_payload = True
        elif remaining_bytes == OPP_HAND_BYTES:
            has_opp_hand_payload = True
        elif remaining_bytes == OPP_SHANTEN_BYTES + OPP_WAIT_BYTES:
            has_shanten_payload = True
            has_wait_payload = True
        elif remaining_bytes == OPP_WAIT_BYTES + OPP_HAND_BYTES:
            has_wait_payload = True
            has_opp_hand_payload = True
        elif remaining_bytes == OPP_SHANTEN_BYTES + OPP_HAND_BYTES:
            has_shanten_payload = True
            has_opp_hand_payload = True
        elif (
            remaining_bytes
            == OPP_SHANTEN_BYTES + OPP_WAIT_BYTES + OPP_HAND_BYTES
        ):
            has_shanten_payload = True
            has_wait_payload = True
            has_opp_hand_payload = True

    opp_shanten: Tuple[int, ...] = tuple([-1] * _OPPONENT_COUNT)
    if has_shanten_payload and (len(record_data) - idx) >= OPP_SHANTEN_BYTES:
        raw_shanten = struct.unpack(
            "<3b", record_data[idx : idx + OPP_SHANTEN_BYTES]
        )
        opp_shanten = tuple(max(0, x - 1) if x >= 0 else -1 for x in raw_shanten)
        idx += OPP_SHANTEN_BYTES

    oracle_wait_matrix_full: Optional[np.ndarray] = None
    if has_wait_payload and (len(record_data) - idx) >= OPP_WAIT_BYTES:
        wait_bytes = record_data[idx : idx + OPP_WAIT_BYTES]
        idx += OPP_WAIT_BYTES
        oracle_wait_matrix_full = _decode_bit_matrix(
            wait_bytes, _OPPONENT_COUNT
        )

    oracle_hand_matrix_full: Optional[np.ndarray] = None
    remaining_bytes = len(record_data) - idx
    if has_opp_hand_payload and remaining_bytes >= OPP_HAND_BYTES:
        oracle_bytes = record_data[idx : idx + OPP_HAND_BYTES]
        idx += OPP_HAND_BYTES
        oracle_hand_matrix_full = _decode_bit_matrix(
            oracle_bytes, OPP_HAND_FEATURE_ROWS_BASE
        )

    # Debug output for record metadata
    if debug_output:
        logger.info("=== Debug Record Data ===")
        logger.info(f"line_number: {line_number}")
        logger.info(f"labeled_opp1 (下家): {labeled_opp1}")
        logger.info(f"labeled_opp2 (對家): {labeled_opp2}")
        logger.info(f"labeled_opp3 (上家): {labeled_opp3}")
        logger.info(f"player_id: {player_id}")
        logger.info(f"round_num: {round_num}")
        logger.info(f"honba: {honba}")
        logger.info(f"scores: {scores}")
        logger.info(f"dora_ids: {dora_ids}")
        logger.info(f"current idx: {idx} bytes")

    # Initialize a 2D tensor for features (34 tiles x TOTAL_FEATURE_ROWS
    # features)
    features_2d = np.zeros((34, TOTAL_FEATURE_ROWS), dtype=np.float32)
    feature_col = 0  # 當前特徵列的索引

    # pred_id, player_id, round, honba: 各佔對應的層數
    # pred_id 改成三層 one-hot 編碼 (3列)
    # player_id 改成四層 one-hot 編碼 (4列)
    # round 改成8層編碼 (8列)
    # honba 正規化 (1列)

    # pred_id 會在後面根據對手設定，這裡先預留3列
    feature_col += 3  # 預留3列給 pred_id

    # player_id 四層編碼 (4列)
    # 整個特徵列（所有34個 tile）都設為1
    if 0 <= player_id < 4:
        features_2d[:, feature_col + player_id] = (
            1.0  # 設定對應列的所有位置為1
        )
    feature_col += 4

    # round_num 八層編碼 (8列)
    # 整個特徵列（所有34個 tile）都設為1
    if 0 <= round_num < 8:
        features_2d[:, feature_col + round_num] = (
            1.0  # 設定對應列的所有位置為1
        )
    feature_col += 8

    # honba 正規化：除以8，如果超過8就當8來除 (1列)
    normalized_honba = min(honba, 8) / 8.0
    features_2d[:, feature_col] = normalized_honba
    feature_col += 1

    # scores: 4個分數各佔一列 (4列) - 除以10萬進行正規化
    for score_idx, score in enumerate(scores):
        features_2d[:, feature_col] = (
            float(score) / 100000.0
        )  # 除以10萬進行正規化
        feature_col += 1

    # phase: optional time section block (only when placed in metadata)
    phase_col_idx = None
    if TIME_SECTION_IN_METADATA and not TIME_SECTION_IN_ACTION:
        phase_col_idx = feature_col
        feature_col += TIME_SECTION_ROWS

    # dora_ids: 佔 DORA_FEATURE_SLOTS 列（預留最多5個寶牌）
    dora_slots = list(dora_ids)
    for slot_idx, slot_dora in enumerate(dora_slots):
        if 0 <= slot_dora < 34:
            features_2d[slot_dora, feature_col] = 1.0  # 單熱編碼寶牌所在位置
        feature_col += 1

    if parse_tokens:
        bitset_matrix = _decode_bit_matrix(bitset_data, BITSET_ROWS_TOTAL)
        oracle_rows_active: Optional[np.ndarray] = None
        if ORACLE_FEATURE_ROWS > 0:
            oracle_rows_parts: List[np.ndarray] = []
            if OPP_HAND_FEATURE_ROWS > 0:
                if (
                    oracle_hand_matrix_full is not None
                    and oracle_hand_matrix_full.shape[0]
                    >= OPP_HAND_FEATURE_ROWS
                ):
                    oracle_rows_parts.append(
                        oracle_hand_matrix_full[:OPP_HAND_FEATURE_ROWS]
                    )
                else:
                    oracle_rows_parts.append(
                        np.zeros(
                            (OPP_HAND_FEATURE_ROWS, FEAT_COLS), dtype=np.int32
                        )
                    )
            if OPP_WAIT_FEATURE_ROWS > 0:
                if (
                    oracle_wait_matrix_full is not None
                    and oracle_wait_matrix_full.shape[0]
                    >= OPP_WAIT_FEATURE_ROWS
                ):
                    oracle_rows_parts.append(
                        oracle_wait_matrix_full[:OPP_WAIT_FEATURE_ROWS]
                    )
                else:
                    oracle_rows_parts.append(
                        np.zeros(
                            (OPP_WAIT_FEATURE_ROWS, FEAT_COLS), dtype=np.int32
                        )
                    )
            if oracle_rows_parts:
                oracle_rows_active = np.concatenate(oracle_rows_parts, axis=0)

        # phase block uses turn estimate; mark mid-game if any label==5
        # (riichi) observed in this record
        if (
            TIME_SECTION_IN_METADATA
            and not TIME_SECTION_IN_ACTION
            and phase_col_idx is not None
        ):
            phase_block = _build_phase_feature(
                bitset_matrix, riichi_seen=riichi_seen
            )
            col_span = phase_block.shape[1]
            features_2d[:, phase_col_idx : phase_col_idx + col_span] = (
                phase_block
            )

        remaining_cols = TOTAL_FEATURE_ROWS - feature_col
        bitset_rows_to_copy = min(BITSET_ROWS_TOTAL, max(0, remaining_cols))
        if bitset_rows_to_copy > 0:
            features_2d[
                :, feature_col : feature_col + bitset_rows_to_copy
            ] = bitset_matrix[:bitset_rows_to_copy].T
            feature_col += bitset_rows_to_copy

        if oracle_rows_active is not None:
            remaining_cols = TOTAL_FEATURE_ROWS - feature_col
            oracle_rows_to_copy = min(
                ORACLE_FEATURE_ROWS,
                oracle_rows_active.shape[0],
                max(0, remaining_cols),
            )
            if oracle_rows_to_copy > 0:
                features_2d[
                    :, feature_col : feature_col + oracle_rows_to_copy
                ] = oracle_rows_active[:oracle_rows_to_copy].T
                feature_col += oracle_rows_to_copy

        if ORACLE_SHANTEN_FEATURE_ROWS > 0:
            remaining_cols = TOTAL_FEATURE_ROWS - feature_col
            shanten_rows_to_copy = min(
                ORACLE_SHANTEN_FEATURE_ROWS, max(0, remaining_cols)
            )
            if shanten_rows_to_copy > 0:
                raw_shanten = np.full(
                    shanten_rows_to_copy, -1, dtype=np.int32
                )
                usable_shanten = min(shanten_rows_to_copy, len(opp_shanten))
                if usable_shanten > 0:
                    raw_shanten[:usable_shanten] = np.asarray(
                        opp_shanten[:usable_shanten], dtype=np.int32
                    )
                shanten_values = np.zeros(
                    shanten_rows_to_copy, dtype=np.float32
                )
                valid_shanten = raw_shanten >= 0
                if np.any(valid_shanten):
                    shanten_values[valid_shanten] = (
                        8
                        - np.clip(raw_shanten[valid_shanten], 0, 8)
                    ) / 8.0
                features_2d[
                    :, feature_col : feature_col + shanten_rows_to_copy
                ] = shanten_values
                feature_col += shanten_rows_to_copy

        expected_features = TOTAL_FEATURE_ROWS
        if feature_col != expected_features:
            logger.warning(
                f"Record {record_id}: generated {feature_col} feature columns, expected {expected_features}"
            )

        base_matrix = features_2d.transpose(1, 0).astype(
            np.float32
        )  # (TOTAL_FEATURE_ROWS, 34)
        order_mode = _normalize_discard_order_mode(DISCARD_ORDER_MODE)
        if order_mode != "original" and INPUT_FORMAT_IS_OLD:
            if order_mode == "reverse":
                _reverse_discard_blocks(base_matrix)
            elif order_mode == "recent_first":
                _recent_first_discard_blocks(base_matrix)
            elif order_mode == "recent_reverse":
                _recent_reverse_discard_blocks(base_matrix)
        token_states, seq_len = build_action_tokens(base_matrix, player_id)
    else:
        # 略過 bitset，僅提取標註資料
        token_states = np.zeros((0, TOTAL_FEATURE_ROWS, 34), dtype=np.float32)
        seq_len = 0

    labels_batch = np.zeros(3, dtype=np.int32)
    labels_batch[0] = labeled_opp1
    labels_batch[1] = labeled_opp2
    labels_batch[2] = labeled_opp3

    return (
        token_states,
        labels_batch,
        line_number,
        seq_len,
        player_id,
        dora_ids,
        action_type,
        action_tile_count,
        action_tiles,
        opp_shanten,
    )


def _parse_one_record(args):
    record_bytes, idx = args
    (
        token_states,
        labels_batch,
        line_number,
        seq_len,
        player_id,
        _,
        _,
        _,
        _,
        opp_shanten,
    ) = parse_single_binary_record(record_bytes, idx)

    # Pre-allocate result list
    clean = []
    opponents = [0, 1, 2]

    # Reshape once, reuse for all opponents
    seq_shape = (token_states.shape[0], TOKEN_FEATURE_DIM)

    for i, pred_id in enumerate(opponents):
        # Skip early if label should be filtered
        if labels_batch[i] in (2, 5):
            continue

        # Make copy only when needed
        seq = token_states.copy()
        if 0 <= pred_id < 3:
            seq[:, pred_id, :] = 1.0
        seq_flat = seq.reshape(seq_shape).astype(np.float32)
        clean.append(
            {
                "features_seq": seq_flat,
                "label": int(labels_batch[i]),
                "shanten": int(opp_shanten[i]) if opp_shanten is not None else -1,
                "line_number": line_number,
                "seq_len": seq_len,
                "player_id": player_id,
                "pred_id": pred_id,
            }
        )

    return clean


def read_binary_data(
    binary_path: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Returns:
        features: np.ndarray of shape (N, max_seq_len, TOKEN_FEATURE_DIM)
        labels: np.ndarray of shape (N,)
        line_numbers: np.ndarray of shape (N,)
        seq_lengths: np.ndarray of shape (N,)
    """
    with open(binary_path, "rb") as f:
        data = f.read()
    record_size = detect_record_size_from_length(len(data))
    if record_size is None or record_size == 0:
        logger.error(
            f"File {binary_path} size {len(data)} is not aligned with known record sizes"
        )
        return (
            np.zeros((0, 1, TOKEN_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    num_records = len(data) // record_size

    # Pre-allocate lists with estimated capacity for better performance
    features_list: List[np.ndarray] = []
    labels_list: List[int] = []
    line_numbers_list: List[int] = []
    seq_lengths: List[int] = []
    player_ids: List[int] = []
    pred_ids: List[int] = []

    # Reserve memory
    features_list.clear()
    labels_list.clear()
    line_numbers_list.clear()
    seq_lengths.clear()
    player_ids.clear()
    pred_ids.clear()

    tasks = (
        (data[i * record_size : (i + 1) * record_size], i)
        for i in range(num_records)
    )
    # Optimize worker count: use physical cores * 2, capped at 24 for best
    # performance
    workers = 16

    with ProcessPoolExecutor(max_workers=workers) as exe:
        for rec_samples in exe.map(_parse_one_record, tasks):
            for sample in rec_samples:
                features_list.append(sample["features_seq"])
                labels_list.append(sample["label"])
                line_numbers_list.append(sample["line_number"])
                seq_lengths.append(sample["seq_len"])
                player_ids.append(sample["player_id"])
                pred_ids.append(sample["pred_id"])

    if not features_list:
        return (
            np.zeros((0, 1, TOKEN_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    max_seq_observed = max(seq_lengths)
    if max_seq_observed > MAX_SEQ_LEN_FALLBACK:
        logger.warning(
            f"Detected sequence length {max_seq_observed} exceeds fallback upper bound {MAX_SEQ_LEN_FALLBACK}. Truncating to fallback length."
        )

    target_seq_len = MAX_SEQ_LEN_FALLBACK
    total_samples = len(features_list)

    padded_features = np.zeros(
        (total_samples, target_seq_len, TOKEN_FEATURE_DIM), dtype=np.float32
    )
    seq_len_array = np.zeros(total_samples, dtype=np.int32)

    for idx, seq in enumerate(features_list):
        cur_len = min(seq.shape[0], target_seq_len)
        if cur_len <= 0:
            cur_len = 1
        padded_features[idx, :cur_len] = seq[:cur_len]
        seq_len_array[idx] = cur_len

    labels_array = np.array(labels_list, dtype=np.int32)
    labels_array = np.isin(labels_array, (1, 3, 5)).astype(np.int64)
    line_numbers_array = np.array(line_numbers_list, dtype=np.uint32)
    player_ids_array = np.array(player_ids, dtype=np.int32)
    pred_ids_array = np.array(pred_ids, dtype=np.int32)

    return (
        padded_features,
        labels_array,
        line_numbers_array,
        seq_len_array,
        player_ids_array,
        pred_ids_array,
    )


class IterBinaryDataset(IterableDataset):
    """
    Iterable dataset for reading multiple binary files.
    """

    def __init__(self, files: List[str]):
        self.files = files
        total_recs = 0
        for f in files:
            try:
                size = os.path.getsize(f)
            except OSError as exc:
                logger.warning(
                    f"[IterBinaryDataset] Skip size check for {f}: {exc}"
                )
                continue
            rec_size = detect_record_size_from_length(size)
            if rec_size is None or rec_size == 0 or size < rec_size:
                logger.warning(
                    f"[IterBinaryDataset] Skip {f}: size {size} not aligned to known record sizes"
                )
                continue
            num_records = size // rec_size
            if num_records == 0:
                logger.warning(
                    f"[IterBinaryDataset] Skip {f}: insufficient bytes for a single record"
                )
                continue
            remainder = size - num_records * rec_size
            if remainder > 0:
                logger.debug(
                    f"[IterBinaryDataset] Ignoring {remainder} trailing bytes in {f} (size={size}, record_size={rec_size})"
                )
            total_recs += num_records
        self.total_samples = total_recs * 3

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            files = self.files
        else:
            per_worker = int(math.ceil(len(self.files) / worker.num_workers))
            start = worker.id * per_worker
            end = min(start + per_worker, len(self.files))
            files = self.files[start:end]

        for fpath in files:
            try:
                with open(fpath, "rb") as f:
                    raw = f.read()

                record_size, num_records = resolve_record_size_from_buffer(raw)
                if record_size == 0 or num_records == 0:
                    logger.warning(
                        f"[IterBinaryDataset] Skip {fpath}: unable to detect record size for {len(raw)} bytes"
                    )
                    continue
                if num_records == 0:
                    logger.warning(f"File {fpath} is empty or too small")
                    continue

                for idx in range(num_records):
                    record_bytes = raw[
                        idx * record_size : (idx + 1) * record_size
                    ]
                    samples = _parse_one_record((record_bytes, idx))

                    for sample in samples:
                        seq = sample["features_seq"]
                        seq_len = min(sample["seq_len"], MAX_SEQ_LEN_FALLBACK)
                        if seq_len <= 0:
                            seq_len = 1

                        padded = np.zeros(
                            (MAX_SEQ_LEN_FALLBACK, TOKEN_FEATURE_DIM),
                            dtype=np.float32,
                        )
                        padded[:seq_len] = seq[:seq_len]

                        mask = np.zeros(MAX_SEQ_LEN_FALLBACK, dtype=np.float32)
                        mask[:seq_len] = 1.0

                        label = 1 if sample["label"] in (1, 3) else 0
                        meta = torch.tensor(
                            [sample["player_id"], sample["pred_id"]],
                            dtype=torch.int32,
                        )

                        yield (
                            torch.from_numpy(padded),
                            torch.tensor(float(label), dtype=torch.float32),
                            int(sample["line_number"]),
                            torch.from_numpy(mask),
                            meta,
                        )

            except Exception as e:
                logger.error(f"Error reading file {fpath}: {e}")
                continue


def split_files_by_copy_status(
    files: List[str], record_size: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """Split binary files into non-copy and copy-containing groups.

    Returns:
        Tuple[List[str], List[str]]: (non_copy_files, copy_files)
    """

    non_copy_files: List[str] = []
    copy_files: List[str] = []

    for fpath in files:
        try:
            with open(fpath, "rb") as f:
                raw = f.read()
        except Exception as exc:
            logger.warning(f"[split_files_by_copy_status] Skip {fpath}: {exc}")
            continue

        if not raw:
            logger.warning(
                f"[split_files_by_copy_status] File {fpath} is empty"
            )
            continue

        length = len(raw)
        rec_size = record_size or detect_record_size_from_length(length)
        if rec_size is None or rec_size == 0 or length < rec_size:
            logger.warning(
                f"[split_files_by_copy_status] Skip {fpath}: size {length} not aligned to known record sizes"
            )
            continue

        num_records = length // rec_size
        if num_records == 0:
            logger.warning(
                f"[split_files_by_copy_status] Skip {fpath}: insufficient bytes for a single record"
            )
            continue

        usable_bytes = num_records * rec_size
        remainder = length - usable_bytes
        if remainder > 0:
            logger.debug(
                f"[split_files_by_copy_status] Ignoring {remainder} trailing bytes in {fpath} (size={length}, record_size={rec_size})"
            )

        payload_view = memoryview(raw)[:usable_bytes]
        labels_view = np.frombuffer(payload_view, dtype=np.int8)
        try:
            labels_view = labels_view.reshape(num_records, rec_size)
        except ValueError:
            logger.warning(
                f"[split_files_by_copy_status] Skip {fpath}: unable to reshape buffer of size {length}"
            )
            continue

        # Label bytes sit right after the 4-byte line number (offsets 4,5,6)
        label_block = labels_view[:, 4:7]
        has_copy = bool(np.any((label_block == 2) | (label_block == 3)))

        target = copy_files if has_copy else non_copy_files
        target.append(fpath)

    return non_copy_files, copy_files


def parse_single_binary_record_base(
    record_data: bytes, record_id: int = -1, debug_output: bool = False
) -> Tuple[np.ndarray, int, int, np.ndarray, int, int, np.ndarray, np.ndarray]:
    """
    Returns:
        labels_batch: (3,) int32, labels for each opponent
        line_number: int
        player_id: int
        dora_ids: (5,) int32
        action_type: int (Binary_record.hpp::ActionType ordinal)
        action_tile_count: int
        action_tiles: (4,) int32
        opp_shanten: (3,) int32
    """
    (
        _tokens,
        labels_batch,
        line_number,
        _seq_len,
        player_id,
        dora_ids,
        action_type,
        action_tile_count,
        action_tiles,
        _opp_shanten,
    ) = parse_single_binary_record(
        record_data, record_id, debug_output, parse_tokens=False
    )

    labels_np = np.array(labels_batch, dtype=np.int32)
    dora_np = np.array(dora_ids, dtype=np.int32)
    action_tiles_np = np.array(action_tiles, dtype=np.int32)
    opp_shanten_np = np.array(_opp_shanten, dtype=np.int32)

    return (
        labels_np,
        int(line_number),
        int(player_id),
        dora_np,
        int(action_type),
        int(action_tile_count),
        action_tiles_np,
        opp_shanten_np,
    )
class IterRoundDataset(IterableDataset):
    """Stream round-scoped binary records and emit per-opponent action sequences.

    Pipeline sketch:
        binary record
            -> parse_single_binary_record (build raw token_states)
            -> _extract_action_sequence + _pack_token_sequence (pad to MAX_SEQ_LEN × ACTION_FEATURE_DIM)
            -> per-opponent pred_id seat one-hot + label binarization
            -> apply is_training/copied file filters
            -> optional phase filtering (early/mid/late)
            -> yield batches to DataLoader
    """

    def __init__(
        self,
        files: List[str],
        record_size: Optional[int] = None,
        is_training: bool = True,
        test_copy: bool = False,
        train_copy: bool = True,
        allowed_phases: Optional[List[str]] = None,
    ):
        self.files = files
        self.record_size = record_size
        self.is_training = is_training
        self._length_cache: Optional[int] = None
        self.test_copy = test_copy
        self.train_copy = train_copy
        self.allowed_phases = None
        if allowed_phases:
            self.allowed_phases = {p.lower() for p in allowed_phases}

    def __len__(self) -> int:
        if self._length_cache is not None:
            return self._length_cache

        total = 0
        for fpath in self.files:
            try:
                with open(fpath, "rb") as f:
                    raw = f.read()
            except Exception as exc:
                logger.warning(
                    f"[IterRoundDataset.__len__] Skip {fpath}: {exc}"
                )
                continue

            record_size = self.record_size or detect_record_size_from_length(
                len(raw)
            )
            if (
                record_size is None
                or record_size == 0
                or len(raw) < record_size
            ):
                logger.warning(
                    f"[IterRoundDataset.__len__] Skip {fpath}: size {len(raw)} not aligned to known record sizes"
                )
                continue

            num_records = len(raw) // record_size
            if num_records <= 0:
                continue
            remainder = len(raw) - num_records * record_size
            if remainder > 0:
                logger.debug(
                    f"[IterRoundDataset.__len__] Ignoring {remainder} trailing bytes in {fpath} (size={len(raw)}, record_size={record_size})"
                )

            labels_per_rec: List[np.ndarray] = []
            for idx in range(num_records):
                rec = raw[idx * record_size : (idx + 1) * record_size]
                labels_np, *_ = parse_single_binary_record_base(
                    rec, idx, debug_output=False
                )
                labels_per_rec.append(labels_np.astype(np.int32))

            if not labels_per_rec:
                continue

            all_labels = np.stack(labels_per_rec, axis=0)
            is_copied_file = bool(np.any(all_labels >= 2))

            if not self.is_training and not self.test_copy and is_copied_file:
                continue
            if self.is_training and not self.train_copy and is_copied_file:
                continue

            valid_indices = range(len(labels_per_rec))
            if is_copied_file and self.is_training and not self.test_copy:
                tenpai_indices = np.where(np.any(all_labels == 3, axis=1))[0]
                if tenpai_indices.size > 0:
                    start_idx = int(tenpai_indices[0])
                    valid_indices = range(start_idx, len(labels_per_rec))

            for ridx in valid_indices:
                labels_vec = labels_per_rec[ridx]
                for raw_label in labels_vec:
                    if int(raw_label) in (2, 5):
                        continue
                    total += 1

        self._length_cache = total
        return total

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            files = self.files
        else:
            per_worker = (
                len(self.files) + worker.num_workers - 1
            ) // worker.num_workers
            start = worker.id * per_worker
            end = min(start + per_worker, len(self.files))
            files = self.files[start:end]

        for fpath in files:
            try:
                with open(fpath, "rb") as f:
                    raw = f.read()
            except Exception as exc:
                logger.error(
                    f"[IterRoundDataset] Error reading file {fpath}: {exc}"
                )
                continue

            record_size = self.record_size or detect_record_size_from_length(
                len(raw)
            )
            if record_size is None or record_size <= 0:
                logger.warning(
                    f"[IterRoundDataset] Skip {fpath}: unable to infer record size from buffer length {len(raw)}"
                )
                continue

            remainder = len(raw) % record_size
            if remainder != 0:
                logger.warning(
                    f"[IterRoundDataset] Skip {fpath}: size {len(raw)} not aligned to record size {record_size} (remainder {remainder})"
                )
                continue

            num_records = len(raw) // record_size
            if num_records <= 0:
                logger.warning(
                    f"[IterRoundDataset] Empty or too small file: {fpath}"
                )
                continue
            remainder = len(raw) - num_records * record_size
            if remainder > 0:
                logger.debug(
                    f"[IterRoundDataset] Ignoring {remainder} trailing bytes in {fpath} (size={len(raw)}, record_size={record_size})"
                )

            metadata: List[Dict[str, Any]] = []
            labels_per_rec: List[np.ndarray] = []

            for idx in range(num_records):
                offset = idx * record_size
                rec = raw[offset : (offset + record_size)]
                # Step 1: parse metadata/labels from binary record
                (
                    labels_np,
                    line_number,
                    player_id,
                    dora_ids,
                    action_type,
                    _action_tile_count,
                    action_tiles_np,
                    opp_shanten_np,
                ) = parse_single_binary_record_base(
                    rec, idx, debug_output=False
                )

                labels_int = labels_np.astype(np.int32)
                tile_id = next((int(t) for t in action_tiles_np if t >= 0), -1)
                metadata.append(
                    {
                        "labels": labels_int,
                        "line_no": int(line_number),
                        "player_id": int(player_id),
                        "action_type": int(action_type),
                        "tile_id": tile_id,
                        "dora_ids": tuple(int(t) for t in dora_ids),
                        "opp_shanten": opp_shanten_np,
                        "offset": offset,
                    }
                )
                labels_per_rec.append(labels_int)

            if not metadata:
                continue

            all_labels = np.stack(labels_per_rec, axis=0)
            is_copied_file = bool(np.any(all_labels >= 2))

            # Step 4: skip copied files by mode (validation/test)
            if not self.is_training and not self.test_copy and is_copied_file:
                continue
            if self.is_training and not self.train_copy and is_copied_file:
                continue

            first_tenpai_idx = -1
            if is_copied_file and self.is_training and not self.test_copy:
                tenpai_indices = np.where(np.any(all_labels == 3, axis=1))[0]
                if tenpai_indices.size > 0:
                    first_tenpai_idx = int(tenpai_indices[0])

            for ridx, meta in enumerate(metadata):
                if (
                    is_copied_file
                    and self.is_training
                    and not self.test_copy
                    and first_tenpai_idx != -1
                    and ridx < first_tenpai_idx
                ):
                    continue

                rec_offset = meta["offset"]
                rec_bytes = raw[rec_offset : (rec_offset + record_size)]
                # Step 2: materialize token_states for the record
                token_states, *_ = parse_single_binary_record(
                    rec_bytes, ridx, debug_output=False, parse_tokens=True
                )
                action_seq = _extract_action_sequence(
                    token_states, meta["player_id"]
                )

                # Estimate phase for routing/filters
                riichi_seen = bool(np.any(meta["labels"] == 5))
                phase_id = _phase_id_from_state(
                    token_states[-1] if token_states.size > 0 else None,
                    riichi_seen=riichi_seen,
                )
                phase_name = PHASE_ID_TO_NAME.get(phase_id, "early")

                if (
                    self.allowed_phases
                    and phase_name not in self.allowed_phases
                ):
                    continue

                # Step 3: pack heterogeneous tokens into fixed-length action
                # features
                riichi_seats = tuple(
                    (meta["player_id"] + offset) % 4
                    for offset, raw_label in enumerate(meta["labels"], start=1)
                    if int(raw_label) == 5
                )
                packed_tokens, init_sidecar, valid_len = _pack_token_sequence(
                    token_states.astype(np.float32),
                    meta["player_id"],
                    meta["action_type"],
                    meta["tile_id"],
                    riichi_seen=riichi_seen,
                    riichi_seats=riichi_seats,
                    dora_ids=meta["dora_ids"],
                )

                if valid_len <= 0:
                    continue

                labels_vec = meta["labels"]
                line_no = meta["line_no"]
                player_id = meta["player_id"]
                opp_shanten_vec = meta.get("opp_shanten", [-1, -1, -1])

                for pred_id, raw_label in enumerate(labels_vec):
                    if int(raw_label) in (2, 5):
                        continue
                    # Step 5: per-opponent seat one-hot and label binarization
                    if PREDICT_SHANTEN:
                        s_val = int(opp_shanten_vec[pred_id])
                        label_bin = max(0, min(s_val, 4)) if s_val >= 0 else 4
                    else:
                        label_bin = 1 if int(raw_label) in (1, 3, 5) else 0

                    per_pred_tokens = packed_tokens.copy()
                    init_base_len = F_INIT_BASE
                    if per_pred_tokens.shape[1] < init_base_len:
                        logger.warning(
                            f"Skip sample line={line_no}: token dim {per_pred_tokens.shape[1]} < init base {init_base_len}"
                        )
                        continue
                    init_block = per_pred_tokens[0, :init_base_len].reshape(
                        F_INIT_ROWS, FEAT_COLS
                    )
                    init_block[:3, :] = 0.0
                    if 0 <= pred_id < 3:
                        # encode target opponent seat
                        init_block[pred_id, :] = 1.0
                    per_pred_tokens[0, :init_base_len] = init_block.reshape(-1)

                    if INPUT_FORMAT_IS_NEW_ADD and NEW_ADD_OG_FLAG_DIM > 0:
                        target_seat = (player_id + pred_id + 1) % 4
                        target_is_tenpai = int(raw_label) in (1, 3, 5)
                        packed_action_seq = (
                            _select_action_window_for_valid_tokens(
                                action_seq, valid_len
                            )
                        )
                        tenpai_flags, new_tenpai_flags = _build_og_step_flags(
                            valid_len,
                            packed_action_seq,
                            target_seat,
                            target_is_tenpai,
                        )
                        tenpai_flags = _reverse_new_add_action_segment(
                            tenpai_flags, valid_len
                        )
                        new_tenpai_flags = _reverse_new_add_action_segment(
                            new_tenpai_flags, valid_len
                        )
                        if (
                            NEW_ADD_OG_TENPAI_FLAG_ROWS > 0
                            and NEW_ADD_OG_TENPAI_OFFSET
                            < per_pred_tokens.shape[1]
                        ):
                            per_pred_tokens[
                                :valid_len, NEW_ADD_OG_TENPAI_OFFSET
                            ] = tenpai_flags[:valid_len]
                        if (
                            NEW_ADD_OG_NEW_TENPAI_FLAG_ROWS > 0
                            and NEW_ADD_OG_NEW_TENPAI_OFFSET
                            < per_pred_tokens.shape[1]
                        ):
                            per_pred_tokens[
                                :valid_len, NEW_ADD_OG_NEW_TENPAI_OFFSET
                            ] = new_tenpai_flags[:valid_len]

                    if self.is_training and not MTP_PRETRAINING:
                        sample_rng = make_deterministic_sample_rng(
                            line_no,
                            pred_id,
                            player_id,
                            phase_id,
                            valid_len,
                            ridx,
                        )
                        per_pred_tokens, mask_arr, current_valid_len = (
                            apply_length_perturbation(
                                per_pred_tokens,
                                valid_len,
                                rng=sample_rng,
                            )
                        )
                    else:
                        mask_arr = np.zeros(
                            MAX_SEQ_LEN_FALLBACK, dtype=np.float32
                        )
                        mask_arr[:valid_len] = 1.0
                        current_valid_len = valid_len

                    per_pred_tokens, mask_arr, current_valid_len = (
                        apply_token_placement(per_pred_tokens, mask_arr)
                    )

                    feat_tensor = torch.from_numpy(per_pred_tokens)
                    sidecar_tensor = torch.from_numpy(init_sidecar)
                    mask_tensor = torch.from_numpy(mask_arr)
                    turn_no = _estimate_turn_number(
                        token_states[-1, BITSET_OFFSET : BITSET_OFFSET + BITSET_ROWS_TOTAL, :]
                        if token_states.size > 0
                        else np.zeros((BITSET_ROWS_TOTAL, FEAT_COLS), dtype=np.float32)
                    )
                    meta_tensor = torch.tensor(
                        [player_id, pred_id, phase_id, int(turn_no)],
                        dtype=torch.int32,
                    )

                    # MTP Pre-training: apply masking to action tokens
                    if MTP_PRETRAINING:
                        sample_rng = make_deterministic_sample_rng(
                            line_no,
                            pred_id,
                            player_id,
                            phase_id,
                            current_valid_len,
                            ridx,
                            "mtp",
                        )
                        masked_tokens, mtp_mask_positions, mtp_tile_labels = (
                            apply_mtp_masking(
                                per_pred_tokens,
                                current_valid_len,
                                attention_mask=mask_arr,
                                rng=sample_rng,
                            )
                        )
                        feat_tensor = torch.from_numpy(masked_tokens)
                        yield (
                            feat_tensor,
                            torch.tensor(
                                float(label_bin), dtype=torch.float32
                            ),
                            int(line_no),
                            mask_tensor.clone(),
                            meta_tensor,
                            sidecar_tensor,
                            torch.from_numpy(mtp_mask_positions),
                            torch.from_numpy(mtp_tile_labels),
                        )
                    else:
                        # Step 6: yield sample to DataLoader
                        yield (
                            feat_tensor,
                            torch.tensor(
                                float(label_bin), dtype=torch.float32
                            ),
                            int(line_no),
                            mask_tensor.clone(),
                            meta_tensor,
                            sidecar_tensor,
                        )


class IndexedRoundDataset(Dataset):
    """Map-style dataset with deterministic ordering and mmap-backed file access.

    Builds a flat index of (file_path, record_offset, pred_id) at init time.
    DataLoader's sampler controls iteration order → fully reproducible with a
    fixed seed.  File contents are memory-mapped so multiple workers share the
    same OS page cache (low RSS).
    """

    def __init__(
        self,
        files: List[str],
        record_size: Optional[int] = None,
        is_training: bool = True,
        test_copy: bool = False,
        train_copy: bool = True,
        allowed_phases: Optional[List[str]] = None,
    ):
        self.files = files
        self.record_size = record_size
        self.is_training = is_training
        self.test_copy = test_copy
        self.train_copy = train_copy
        self.allowed_phases: Optional[Set[str]] = None
        if allowed_phases:
            self.allowed_phases = {p.lower() for p in allowed_phases}

        # Per-worker mmap cache (populated lazily after fork)
        # Use OrderedDict for LRU eviction to avoid too many open files
        self._mmap_cache: OrderedDict[str, mmap.mmap] = OrderedDict()
        # keep file objects alive
        self._fd_cache: OrderedDict[str, Any] = OrderedDict()
        self._max_open_files = 1024  # Limit concurrent open files per worker
        self._record_cache: OrderedDict[
            Tuple[int, int], Optional[Dict[str, Any]]
        ] = OrderedDict()
        self._record_cache_size = (
            int(STRICT_REPRO_FAST_RECORD_CACHE_SIZE)
            if STRICT_REPRODUCIBLE and STRICT_REPRO_FAST_PIPELINE
            else 0
        )

        # Compact index storage to avoid huge Python-object overhead:
        # (file_id, byte_offset, pred_id) are stored in NumPy arrays.
        self._files_by_id: List[str] = []
        self._file_id_by_path: Dict[str, int] = {}
        self._record_size_by_file_id: List[int] = []
        self._index_file_ids = np.empty(0, dtype=np.int32)
        self._index_offsets = np.empty(0, dtype=np.int64)
        self._index_pred_ids = np.empty(0, dtype=np.int8)
        # Binary labels aligned with compact index order (0/1), used by optional
        # balanced label sampling in main.py
        self.sample_labels = np.empty(0, dtype=np.uint8)
        self.sample_turns = np.empty(0, dtype=np.uint16)
        # Optional p/q correction weights aligned with index order, used when
        # training enables importance-sampling loss correction.
        self._importance_sampling_loss_correction: Optional[np.ndarray] = None
        self._build_index()

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """Scan all files, filter by copy status, and populate compact index arrays."""
        index_file_id_chunks: List[np.ndarray] = []
        index_offset_chunks: List[np.ndarray] = []
        index_pred_id_chunks: List[np.ndarray] = []
        sample_label_chunks: List[np.ndarray] = []
        sample_turn_chunks: List[np.ndarray] = []

        def _ensure_file_id(path: str, rec_size: int) -> int:
            existing = self._file_id_by_path.get(path)
            if existing is not None:
                return existing
            new_id = len(self._files_by_id)
            self._files_by_id.append(path)
            self._file_id_by_path[path] = new_id
            self._record_size_by_file_id.append(int(rec_size))
            return new_id

        for fpath in self.files:
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                continue

            rec_size = self.record_size or detect_record_size_from_length(
                fsize
            )
            if rec_size is None or rec_size <= 0 or fsize < rec_size:
                continue

            num_records = fsize // rec_size
            if num_records <= 0:
                continue

            # Quick label scan to decide copy status and valid record range
            try:
                with open(fpath, "rb") as f:
                    raw = f.read()
            except Exception:
                continue

            labels_per_rec = np.empty((num_records, 3), dtype=np.int8)
            turns_per_rec = np.zeros(num_records, dtype=np.uint16)
            shantens_per_rec = np.empty((num_records, 3), dtype=np.int8)

            for idx in range(num_records):
                rec = raw[idx * rec_size : (idx + 1) * rec_size]
                if PREDICT_SHANTEN:
                    (
                        l_np,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        s_np,
                    ) = parse_single_binary_record_base(rec, idx, False)
                    labels_vec = l_np[:3]
                    s_vec = s_np[:3]
                    # Estimate turn metadata roughly if needed, or proper parse
                    _, turn_no = _parse_record_index_metadata(rec)
                    labels_per_rec[idx] = labels_vec
                    shantens_per_rec[idx] = s_vec
                    turns_per_rec[idx] = np.uint16(max(int(turn_no), 0))
                else:
                    labels_vec, turn_no = _parse_record_index_metadata(rec)
                    labels_per_rec[idx] = labels_vec
                    turns_per_rec[idx] = np.uint16(max(int(turn_no), 0))

            is_copied_file = bool(np.any(labels_per_rec >= 2))

            if not self.is_training and not self.test_copy and is_copied_file:
                continue
            if self.is_training and not self.train_copy and is_copied_file:
                continue

            file_id = _ensure_file_id(fpath, rec_size)
            record_keep = np.ones(num_records, dtype=bool)
            if is_copied_file and self.is_training and not self.test_copy:
                tenpai_indices = np.where(np.any(labels_per_rec == 3, axis=1))[0]
                if tenpai_indices.size > 0:
                    record_keep[: int(tenpai_indices[0])] = False

            eligible = record_keep[:, None] & ~np.isin(labels_per_rec, (2, 5))
            record_ids, pred_ids = np.nonzero(eligible)
            if record_ids.size == 0:
                continue

            selected_labels = labels_per_rec[record_ids, pred_ids]
            if PREDICT_SHANTEN:
                selected_shanten = shantens_per_rec[record_ids, pred_ids]
                labels_out = np.where(
                    selected_shanten >= 0,
                    np.clip(selected_shanten, 0, 4),
                    4,
                ).astype(np.uint8, copy=False)
            else:
                labels_out = np.isin(selected_labels, (1, 3, 5)).astype(
                    np.uint8, copy=False
                )

            index_file_id_chunks.append(
                np.full(record_ids.size, file_id, dtype=np.int32)
            )
            index_offset_chunks.append(
                record_ids.astype(np.int64, copy=False) * int(rec_size)
            )
            index_pred_id_chunks.append(pred_ids.astype(np.int8, copy=False))
            sample_label_chunks.append(labels_out)
            sample_turn_chunks.append(
                turns_per_rec[record_ids].astype(np.uint16, copy=False)
            )

        def _concat_chunks(
            chunks: List[np.ndarray], dtype: np.dtype
        ) -> np.ndarray:
            if not chunks:
                return np.empty(0, dtype=dtype)
            return np.concatenate(chunks).astype(dtype, copy=False)

        self._index_file_ids = _concat_chunks(index_file_id_chunks, np.int32)
        self._index_offsets = _concat_chunks(index_offset_chunks, np.int64)
        self._index_pred_ids = _concat_chunks(index_pred_id_chunks, np.int8)
        self.sample_labels = _concat_chunks(sample_label_chunks, np.uint8)
        self.sample_turns = _concat_chunks(sample_turn_chunks, np.uint16)

        pre_mask_count = int(self.sample_labels.size)

        if (
            self.is_training
            and TURN_LABEL_PERTURB_MATCH
            and not PREDICT_SHANTEN
            and self.sample_labels.size > 0
        ):
            keep_mask = _match_turn_label_ratios_with_perturbation(
                self.sample_turns,
                self.sample_labels,
                seed=int(TURN_LABEL_PERTURB_MATCH_SEED),
                min_total=int(TURN_LABEL_PERTURB_MATCH_MIN_TOTAL),
            )
            if keep_mask.shape[0] == self.sample_labels.shape[0]:
                self._index_file_ids = self._index_file_ids[keep_mask]
                self._index_offsets = self._index_offsets[keep_mask]
                self._index_pred_ids = self._index_pred_ids[keep_mask]
                self.sample_labels = self.sample_labels[keep_mask]
                self.sample_turns = self.sample_turns[keep_mask]
                logger.info(
                    "[IndexedRoundDataset] Turn label ratio match: "
                    f"kept {int(self.sample_labels.size)}/{pre_mask_count} samples"
                )

        sample_count = int(self._index_pred_ids.size)
        index_bytes = (
            int(self._index_file_ids.nbytes)
            + int(self._index_offsets.nbytes)
            + int(self._index_pred_ids.nbytes)
            + int(self.sample_labels.nbytes)
            + int(self.sample_turns.nbytes)
        )
        logger.info(
            f"[IndexedRoundDataset] Built index with {sample_count} samples from {len(self.files)} files "
            f"(compact index ~{index_bytes / (1024 * 1024):.1f} MB)"
        )

    # ------------------------------------------------------------------
    def _get_mmap(self, fpath: str) -> mmap.mmap:
        """Return a read-only mmap for *fpath*, cached per worker process with LRU eviction."""
        mm = self._mmap_cache.get(fpath)
        if mm is not None:
            # Move to end (most recently used)
            self._mmap_cache.move_to_end(fpath)
            return mm

        # Evict oldest entries if cache is full
        while len(self._mmap_cache) >= self._max_open_files:
            oldest_path, oldest_mm = self._mmap_cache.popitem(last=False)
            try:
                oldest_mm.close()
            except Exception:
                pass
            oldest_fd = self._fd_cache.pop(oldest_path, None)
            if oldest_fd is not None:
                try:
                    oldest_fd.close()
                except Exception:
                    pass

        f = open(fpath, "rb")  # noqa: SIM115 — must keep alive
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._fd_cache[fpath] = f
        self._mmap_cache[fpath] = mm
        return mm

    # ------------------------------------------------------------------
    def get_indices_for_files(self, files: Sequence[str]) -> np.ndarray:
        """Return compact-index positions for files in the supplied order."""
        if self._index_file_ids.size == 0 or not files:
            return np.empty(0, dtype=np.int64)
        out: List[np.ndarray] = []
        for path in files:
            file_id = self._file_id_by_path.get(path)
            if file_id is None:
                continue
            hits = np.nonzero(self._index_file_ids == int(file_id))[0]
            if hits.size > 0:
                out.append(hits.astype(np.int64, copy=False))
        if not out:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(out).astype(np.int64, copy=False)

    # ------------------------------------------------------------------
    def set_active_indices(self, indices: Optional[np.ndarray]) -> None:
        """Store current epoch subset for diagnostics without changing sampling."""
        if indices is None:
            self._active_indices = None
            return
        arr = np.asarray(indices, dtype=np.int64)
        self._active_indices = arr[(arr >= 0) & (arr < len(self))]

    # ------------------------------------------------------------------
    def get_active_sample_labels(self) -> np.ndarray:
        """Return labels for current epoch subset when present."""
        active = getattr(self, "_active_indices", None)
        if active is None:
            return self.sample_labels
        return self.sample_labels[active]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self._index_pred_ids.size)

    # ------------------------------------------------------------------
    def get_sample_labels(self) -> np.ndarray:
        """Return label array aligned with index order."""
        return self.sample_labels

    # ------------------------------------------------------------------
    def get_sample_turns(self) -> np.ndarray:
        """Return turn array aligned with index order."""
        return self.sample_turns

    # ------------------------------------------------------------------
    def set_importance_sampling_loss_correction(
        self, correction_weights: np.ndarray
    ) -> None:
        """Attach per-index p/q correction weights for training loss."""
        corr = np.asarray(correction_weights, dtype=np.float32)
        if corr.ndim != 1:
            raise ValueError(
                "correction_weights must be 1D. "
                f"Got shape={corr.shape}."
            )
        if corr.size != len(self):
            raise ValueError(
                "correction_weights size must equal dataset length. "
                f"Got {corr.size} vs {len(self)}."
            )
        self._importance_sampling_loss_correction = corr

    # ------------------------------------------------------------------
    def clear_importance_sampling_loss_correction(self) -> None:
        """Disable per-index p/q correction export in __getitem__."""
        self._importance_sampling_loss_correction = None

    # ------------------------------------------------------------------
    def _cache_record(
        self,
        key: Tuple[int, int],
        value: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if self._record_cache_size <= 0:
            return value
        self._record_cache[key] = value
        self._record_cache.move_to_end(key)
        while len(self._record_cache) > self._record_cache_size:
            self._record_cache.popitem(last=False)
        return value

    # ------------------------------------------------------------------
    def _get_packed_record(
        self,
        file_id: int,
        byte_offset: int,
        rec_size: int,
        fpath: str,
    ) -> Optional[Dict[str, Any]]:
        """Parse and pack one binary record, cached per worker process."""
        cache_key = (int(file_id), int(byte_offset))
        cached = self._record_cache.get(cache_key)
        if cached is not None or cache_key in self._record_cache:
            self._record_cache.move_to_end(cache_key)
            return cached

        mm = self._get_mmap(fpath)
        rec_bytes = mm[byte_offset : byte_offset + rec_size]
        ridx = byte_offset // rec_size

        (
            token_states,
            labels_batch,
            line_number,
            _seq_len,
            player_id,
            dora_ids,
            action_type,
            _action_tile_count,
            action_tiles,
            opp_shanten,
        ) = parse_single_binary_record(
            rec_bytes, ridx, debug_output=False, parse_tokens=True
        )

        tile_id = next((int(t) for t in action_tiles if t >= 0), -1)
        riichi_seen = bool(np.any(np.isin(labels_batch, [5])))

        phase_id = _phase_id_from_state(
            token_states[-1] if token_states.size > 0 else None,
            riichi_seen=riichi_seen,
        )
        phase_name = PHASE_ID_TO_NAME.get(phase_id, "early")
        if self.allowed_phases and phase_name not in self.allowed_phases:
            return self._cache_record(cache_key, None)

        riichi_seats = tuple(
            (player_id + offset) % 4
            for offset, raw_label in enumerate(labels_batch, start=1)
            if int(raw_label) == 5
        )
        action_metadata = None
        if USE_COMPACT_BINARY_TOKEN_PARSER:
            action_metadata = _build_action_metadata_from_final_state(
                token_states[-1] if token_states.size > 0 else None,
                player_id,
            )
            if VERIFY_COMPACT_BINARY_TOKEN_PARSER:
                old_actions, old_rows = _extract_action_sequence_and_discard_rows(
                    token_states, player_id
                )
                if (
                    list(action_metadata[0]) != old_actions
                    or list(action_metadata[1]) != old_rows
                ):
                    raise RuntimeError(
                        "Compact binary token parser metadata mismatch "
                        f"at file_id={file_id}, offset={byte_offset}."
                    )

        packed_tokens, init_sidecar, valid_len = _pack_token_sequence(
            token_states.astype(np.float32),
            player_id,
            action_type,
            tile_id,
            riichi_seen=riichi_seen,
            riichi_seats=riichi_seats,
            dora_ids=tuple(int(d) for d in dora_ids),
            action_metadata=action_metadata,
        )

        if valid_len <= 0:
            return self._cache_record(cache_key, None)

        return self._cache_record(
            cache_key,
            {
                "packed_tokens": packed_tokens,
                "init_sidecar": init_sidecar,
                "valid_len": int(valid_len),
                "labels_batch": labels_batch,
                "line_number": int(line_number),
                "player_id": int(player_id),
                "phase_id": int(phase_id),
                "opp_shanten": opp_shanten,
                "ridx": int(ridx),
                "byte_offset": int(byte_offset),
            },
        )

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        i = int(idx)
        file_id = int(self._index_file_ids[i])
        byte_offset = int(self._index_offsets[i])
        pred_id = int(self._index_pred_ids[i])
        fpath = self._files_by_id[file_id]
        rec_size = (
            int(self._record_size_by_file_id[file_id])
            if 0 <= file_id < len(self._record_size_by_file_id)
            else 0
        )
        if rec_size is None or rec_size <= 0:
            # Return zero tensor as fallback
            return self._fallback_sample()

        record = self._get_packed_record(file_id, byte_offset, rec_size, fpath)
        if record is None:
            return self._fallback_sample()

        # Step 3: per-opponent pred_id encoding
        packed_tokens = record["packed_tokens"]
        init_sidecar = record["init_sidecar"]
        valid_len = int(record["valid_len"])
        labels_batch = record["labels_batch"]
        line_number = int(record["line_number"])
        player_id = int(record["player_id"])
        phase_id = int(record["phase_id"])
        turn_no = int(self.sample_turns[i]) if i < len(self.sample_turns) else 0
        _opp_shanten = record["opp_shanten"]
        ridx = int(record["ridx"])
        raw_label = int(labels_batch[pred_id])
        if PREDICT_SHANTEN:
            s_val = int(_opp_shanten[pred_id]) if _opp_shanten is not None else -1
            label_bin = max(0, min(s_val, 4)) if s_val >= 0 else 4
        else:
            label_bin = 1 if raw_label in (1, 3, 5) else 0

        per_pred_tokens = packed_tokens.copy()
        init_block = per_pred_tokens[0, :F_INIT_BASE].reshape(
            F_INIT_ROWS, FEAT_COLS
        )
        init_block[:3, :] = 0.0
        if 0 <= pred_id < 3:
            init_block[pred_id, :] = 1.0
        per_pred_tokens[0, :F_INIT_BASE] = init_block.reshape(-1)

        if self.is_training and not MTP_PRETRAINING:
            sample_rng = make_deterministic_sample_rng(
                line_number,
                pred_id,
                player_id,
                phase_id,
                valid_len,
                ridx,
                byte_offset,
            )
            per_pred_tokens, mask_arr, current_valid_len = (
                apply_length_perturbation(
                    per_pred_tokens,
                    valid_len,
                    rng=sample_rng,
                )
            )
        else:
            mask_arr = np.zeros(MAX_SEQ_LEN_FALLBACK, dtype=np.float32)
            mask_arr[:valid_len] = 1.0
            current_valid_len = valid_len

        per_pred_tokens, mask_arr, current_valid_len = apply_token_placement(
            per_pred_tokens, mask_arr
        )

        # MTP Pre-training: apply masking to action tokens
        mtp_mask_positions = np.zeros(MAX_SEQ_LEN_FALLBACK, dtype=bool)
        mtp_tile_labels = np.full(MAX_SEQ_LEN_FALLBACK, -1, dtype=np.int32)
        if MTP_PRETRAINING:
            sample_rng = make_deterministic_sample_rng(
                line_number,
                pred_id,
                player_id,
                phase_id,
                current_valid_len,
                ridx,
                byte_offset,
                "mtp",
            )
            per_pred_tokens, mtp_mask_positions, mtp_tile_labels = (
                apply_mtp_masking(
                    per_pred_tokens,
                    current_valid_len,
                    attention_mask=mask_arr,
                    rng=sample_rng,
                )
            )

        feat_tensor = torch.from_numpy(per_pred_tokens)
        mask_tensor = torch.from_numpy(mask_arr)
        meta_tensor = torch.tensor(
            [player_id, pred_id, phase_id, turn_no], dtype=torch.int32
        )
        sidecar_tensor = torch.from_numpy(init_sidecar)

        if MTP_PRETRAINING:
            sample = (
                feat_tensor,
                torch.tensor(float(label_bin), dtype=torch.float32),
                int(line_number),
                mask_tensor,
                meta_tensor,
                sidecar_tensor,
                torch.from_numpy(mtp_mask_positions),
                torch.from_numpy(mtp_tile_labels),
            )
        else:
            sample = (
                feat_tensor,
                torch.tensor(float(label_bin), dtype=torch.float32),
                int(line_number),
                mask_tensor,
                meta_tensor,
                sidecar_tensor,
            )

        if self._importance_sampling_loss_correction is not None:
            sample = sample + (
                torch.tensor(
                    float(self._importance_sampling_loss_correction[i]),
                    dtype=torch.float32,
                ),
            )
        return sample

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_sample():
        """Return a zero sample that the training loop can safely ignore."""
        base = (
            torch.zeros(
                MAX_SEQ_LEN_FALLBACK,
                PACKED_ACTION_FEATURE_DIM,
                dtype=torch.float32,
            ),
            torch.tensor(-1.0, dtype=torch.float32),  # sentinel label
            0,
            torch.zeros(MAX_SEQ_LEN_FALLBACK, dtype=torch.float32),
            torch.tensor([0, 0, 0, 0], dtype=torch.int32),
            torch.zeros(NEW_ADD_INIT_SIDECAR_DIM, dtype=torch.float32),
        )
        if MTP_PRETRAINING:
            return base + (
                torch.zeros(MAX_SEQ_LEN_FALLBACK, dtype=torch.bool),
                # mtp_mask_positions
                torch.full(
                    (MAX_SEQ_LEN_FALLBACK,), -1, dtype=torch.int32
                ),  # mtp_tile_labels
            )
        return base

    # ------------------------------------------------------------------
    @staticmethod
    def collate_filter(batch):
        """Custom collate that drops fallback samples (label == -1)."""
        filtered = [s for s in batch if s[1].item() >= 0.0]
        if not filtered:
            # Return a minimal batch to keep DataLoader happy
            return IndexedRoundDataset._fallback_sample()
        return torch.utils.data.dataloader.default_collate(filtered)


class RecordGroupedSampler(Sampler[int]):
    """Deterministic sampler that keeps pred_id samples from one record together."""

    def __init__(
        self,
        dataset: IndexedRoundDataset,
        generator: Optional[torch.Generator] = None,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.generator = generator
        self.shuffle = bool(shuffle)
        n = len(dataset)
        if n <= 0:
            self._group_starts = np.empty(0, dtype=np.int64)
            self._group_ends = np.empty(0, dtype=np.int64)
        else:
            file_ids = dataset._index_file_ids
            offsets = dataset._index_offsets
            changes = (file_ids[1:] != file_ids[:-1]) | (
                offsets[1:] != offsets[:-1]
            )
            self._group_starts = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.flatnonzero(changes).astype(np.int64) + 1,
                )
            )
            self._group_ends = np.concatenate(
                (
                    self._group_starts[1:],
                    np.array([n], dtype=np.int64),
                )
            )

    def __iter__(self):
        group_count = int(self._group_starts.size)
        if group_count <= 0:
            return

        if self.shuffle:
            order = torch.randperm(
                group_count, generator=self.generator
            ).tolist()
        else:
            order = range(group_count)

        for group_idx in order:
            start = int(self._group_starts[group_idx])
            end = int(self._group_ends[group_idx])
            for sample_idx in range(start, end):
                yield sample_idx

    def __len__(self) -> int:
        return len(self.dataset)


__all__ = [name for name in globals() if not name.startswith("__")]
