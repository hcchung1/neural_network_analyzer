"""Feature builders, token packing, placement, and augmentation."""

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

def _collect_discard_rows(
    base_matrix: np.ndarray, player_id: int
) -> Dict[int, List[int]]:
    """依照座位收集每位玩家的棄牌列索引（已轉置為 TOTAL_FEATURE_ROWS×34 後的列）。"""
    seat_row_map: Dict[int, List[int]] = {seat: [] for seat in range(4)}

    def collect_rows(
        discard_range: range, tsumogiri_range: range
    ) -> List[int]:
        paired_rows: List[int] = []
        for d_idx, t_idx in zip(discard_range, tsumogiri_range):
            discard_row = BITSET_OFFSET + d_idx
            tsumogiri_row = BITSET_OFFSET + t_idx
            if np.any(base_matrix[discard_row] != 0):
                paired_rows.append(discard_row)
                paired_rows.append(tsumogiri_row)
        return paired_rows

    seat_row_map[player_id] = collect_rows(MY_DISCARD_ROWS, MY_TSUMOGIRI_ROWS)

    for offset, (opp_discard_range, opp_tsumo_range) in enumerate(
        zip(OPP_DISCARD_ROWS, OPP_TSUMOGIRI_ROWS), start=1
    ):
        seat = (player_id + offset) % 4
        seat_row_map[seat] = collect_rows(opp_discard_range, opp_tsumo_range)

    return seat_row_map


def _reverse_discard_blocks(base_matrix: np.ndarray) -> None:
    """Reverse discard slots so the most recent entry sits at the front."""
    slot_specs = [(MY_DISCARD_ROWS, MY_TSUMOGIRI_ROWS)] + list(
        zip(OPP_DISCARD_ROWS, OPP_TSUMOGIRI_ROWS)
    )

    for discard_range, tsumogiri_range in slot_specs:
        _reverse_single_discard_block(
            base_matrix, discard_range, tsumogiri_range
        )


def _reverse_single_discard_block(
    base_matrix: np.ndarray,
    discard_range: range,
    tsumogiri_range: range,
) -> None:
    row_pairs = [
        (BITSET_OFFSET + d_idx, BITSET_OFFSET + t_idx)
        for d_idx, t_idx in zip(discard_range, tsumogiri_range)
    ]
    if not row_pairs:
        return

    slot_values = [
        (base_matrix[d_row].copy(), base_matrix[t_row].copy())
        for d_row, t_row in row_pairs
    ]

    filled_slots = [pair for pair in slot_values if np.any(pair[0] != 0)]
    reversed_slots = list(reversed(filled_slots))

    pad_needed = len(row_pairs) - len(reversed_slots)
    if pad_needed > 0:
        zeros_disc = np.zeros_like(base_matrix[row_pairs[0][0]])
        zeros_tsumo = np.zeros_like(base_matrix[row_pairs[0][1]])
        for _ in range(pad_needed):
            reversed_slots.append((zeros_disc.copy(), zeros_tsumo.copy()))

    for (d_row, t_row), (disc_vals, tsumo_vals) in zip(
        row_pairs, reversed_slots
    ):
        base_matrix[d_row] = disc_vals
        base_matrix[t_row] = tsumo_vals


def _recent_first_discard_blocks(
    base_matrix: np.ndarray, recent_count: int = 6
) -> None:
    """Reorder discard slots so most recent N entries come first, followed by the rest."""
    slot_specs = [(MY_DISCARD_ROWS, MY_TSUMOGIRI_ROWS)] + list(
        zip(OPP_DISCARD_ROWS, OPP_TSUMOGIRI_ROWS)
    )

    for discard_range, tsumogiri_range in slot_specs:
        _recent_first_single_discard_block(
            base_matrix, discard_range, tsumogiri_range, recent_count
        )


def _recent_first_single_discard_block(
    base_matrix: np.ndarray,
    discard_range: range,
    tsumogiri_range: range,
    recent_count: int = 6,
) -> None:
    """Reorder a single discard block so most recent N entries come first."""
    row_pairs = [
        (BITSET_OFFSET + d_idx, BITSET_OFFSET + t_idx)
        for d_idx, t_idx in zip(discard_range, tsumogiri_range)
    ]
    if not row_pairs:
        return

    slot_values = [
        (base_matrix[d_row].copy(), base_matrix[t_row].copy())
        for d_row, t_row in row_pairs
    ]

    filled_slots = [pair for pair in slot_values if np.any(pair[0] != 0)]
    n = len(filled_slots)
    if n <= recent_count:
        return

    split_idx = n - recent_count
    reordered_slots = filled_slots[split_idx:] + filled_slots[:split_idx]

    pad_needed = len(row_pairs) - len(reordered_slots)
    if pad_needed > 0:
        zeros_disc = np.zeros_like(base_matrix[row_pairs[0][0]])
        zeros_tsumo = np.zeros_like(base_matrix[row_pairs[0][1]])
        for _ in range(pad_needed):
            reordered_slots.append((zeros_disc.copy(), zeros_tsumo.copy()))

    for (d_row, t_row), (disc_vals, tsumo_vals) in zip(
        row_pairs, reordered_slots
    ):
        base_matrix[d_row] = disc_vals
        base_matrix[t_row] = tsumo_vals


def _recent_reverse_discard_blocks(
    base_matrix: np.ndarray, recent_count: int = 6
) -> None:
    """Reorder discard slots so most recent N entries (in reverse) come first."""
    slot_specs = [(MY_DISCARD_ROWS, MY_TSUMOGIRI_ROWS)] + list(
        zip(OPP_DISCARD_ROWS, OPP_TSUMOGIRI_ROWS)
    )

    for discard_range, tsumogiri_range in slot_specs:
        _recent_reverse_single_discard_block(
            base_matrix, discard_range, tsumogiri_range, recent_count
        )


def _recent_reverse_single_discard_block(
    base_matrix: np.ndarray,
    discard_range: range,
    tsumogiri_range: range,
    recent_count: int = 6,
) -> None:
    """Reorder one discard block to [latest..] + [older..] with padding at tail."""
    row_pairs = [
        (BITSET_OFFSET + d_idx, BITSET_OFFSET + t_idx)
        for d_idx, t_idx in zip(discard_range, tsumogiri_range)
    ]
    if not row_pairs:
        return

    slot_values = [
        (base_matrix[d_row].copy(), base_matrix[t_row].copy())
        for d_row, t_row in row_pairs
    ]

    filled_slots = [pair for pair in slot_values if np.any(pair[0] != 0)]
    if not filled_slots:
        return

    split_idx = max(0, len(filled_slots) - recent_count)
    recent_reversed = list(reversed(filled_slots[split_idx:]))
    reordered_slots = recent_reversed + filled_slots[:split_idx]

    pad_needed = len(row_pairs) - len(reordered_slots)
    if pad_needed > 0:
        zeros_disc = np.zeros_like(base_matrix[row_pairs[0][0]])
        zeros_tsumo = np.zeros_like(base_matrix[row_pairs[0][1]])
        for _ in range(pad_needed):
            reordered_slots.append((zeros_disc.copy(), zeros_tsumo.copy()))

    for (d_row, t_row), (disc_vals, tsumo_vals) in zip(
        row_pairs, reordered_slots
    ):
        base_matrix[d_row] = disc_vals
        base_matrix[t_row] = tsumo_vals


def _extract_discard_ids(
    base_matrix: np.ndarray, discard_range: range
) -> List[int]:
    """Return discard tile ids in row order for the given range."""
    tile_ids: List[int] = []
    for row_idx in discard_range:
        row = base_matrix[row_idx]
        if np.any(row != 0):
            tile_ids.append(int(np.argmax(row)))
    return tile_ids


def _build_discard_history_features(base_matrix: np.ndarray) -> np.ndarray:
    """Build last-N discard history features for 4 seats (self + 3 opps)."""
    out = np.zeros(
        (4, NEW_ADD_DISCARD_HISTORY_STEPS, FEAT_COLS), dtype=np.float32
    )
    seat_ranges = [MY_DISCARD_ROWS, *OPP_DISCARD_ROWS]

    for seat_idx, discard_range in enumerate(seat_ranges):
        tile_ids = _extract_discard_ids(base_matrix, discard_range)
        if not tile_ids:
            continue
        recent = tile_ids[-NEW_ADD_DISCARD_HISTORY_STEPS:]
        start = NEW_ADD_DISCARD_HISTORY_STEPS - len(recent)
        for offset, tile_id in enumerate(recent):
            if 0 <= tile_id < FEAT_COLS:
                out[seat_idx, start + offset, tile_id] = 1.0

    return out.reshape(-1)


def _build_discard_set_features(base_matrix: np.ndarray) -> np.ndarray:
    """Build per-seat discard set features (orderless)."""
    out = np.zeros((4, FEAT_COLS), dtype=np.float32)
    seat_ranges = [MY_DISCARD_ROWS, *OPP_DISCARD_ROWS]

    for seat_idx, discard_range in enumerate(seat_ranges):
        for row_idx in discard_range:
            row = base_matrix[row_idx]
            if np.any(row != 0):
                tile_id = int(np.argmax(row))
                if 0 <= tile_id < FEAT_COLS:
                    out[seat_idx, tile_id] = 1.0

    return out.reshape(-1)


def _build_meld_set_features(base_matrix: np.ndarray) -> np.ndarray:
    """Build per-seat meld set features (orderless, type-agnostic)."""
    out = np.zeros((4, FEAT_COLS), dtype=np.float32)
    seat_ranges = [MY_MELD_ROWS, *OPP_MELD_ROWS]

    for seat_idx, meld_range in enumerate(seat_ranges):
        for row_idx in meld_range:
            row = base_matrix[row_idx]
            if np.any(row != 0):
                tile_id = int(np.argmax(row))
                if 0 <= tile_id < FEAT_COLS:
                    out[seat_idx, tile_id] = 1.0

    return out.reshape(-1)


def _build_remaining_tile_features(base_matrix: np.ndarray) -> np.ndarray:
    """Build remaining tile counts (normalized to 0-1) per tile id."""
    visible = _visible_tile_counts(base_matrix, include_private_hand=True)
    remaining = np.clip(4 - visible, 0, 4).astype(np.float32)
    return remaining / 4.0


def _build_new_add_init_extra(base_matrix: np.ndarray) -> np.ndarray:
    """Build concatenated New_Add init extras from the supplied state."""
    if NEW_ADD_INIT_EXTRA_DIM == 0 or not USE_NEW_ADD_INIT_EXTRA:
        return np.zeros((0,), dtype=np.float32)

    discard_hist = _build_discard_history_features(base_matrix)
    discard_set = _build_discard_set_features(base_matrix)
    meld_set = _build_meld_set_features(base_matrix)
    remaining = _build_remaining_tile_features(base_matrix)
    return np.concatenate(
        [discard_hist, discard_set, meld_set, remaining], axis=0
    ).astype(np.float32)


def _build_new_add_init_old_single_recent_extra(
    states: np.ndarray,
    player_id: int,
    recent_count: int,
    action_rows: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Build old_single-style recent-action snapshot as New_Add init extra."""
    if NEW_ADD_INIT_OLD_SINGLE_RECENT_DIM == 0:
        return np.zeros((0,), dtype=np.float32)

    if not (0 <= int(player_id) < 4) or states.size == 0:
        return np.zeros((NEW_ADD_INIT_OLD_SINGLE_RECENT_DIM,), dtype=np.float32)

    recent_state = _build_old_single_recent_action_state(
        states,
        int(player_id),
        int(recent_count),
        action_rows=action_rows,
    )
    flat = recent_state.reshape(-1).astype(np.float32)

    if flat.size >= NEW_ADD_INIT_OLD_SINGLE_RECENT_DIM:
        return flat[:NEW_ADD_INIT_OLD_SINGLE_RECENT_DIM]

    out = np.zeros((NEW_ADD_INIT_OLD_SINGLE_RECENT_DIM,), dtype=np.float32)
    out[: flat.size] = flat
    return out


def _build_new_add_final_context_block(state_matrix: np.ndarray) -> np.ndarray:
    """Build New_Add final-context token payload.

    Included information:
    - all 4 seats discard sets (4 x 34)
    - all 4 seats meld sets (4 x 34)
    - dora slot block (5 x 34)
    - own hand layers (4 x 34)
    """

    if (
        NEW_ADD_FINAL_CONTEXT_DIM == 0
        or state_matrix is None
        or state_matrix.size == 0
    ):
        return np.zeros((0,), dtype=np.float32)

    discard_set = np.zeros((4, FEAT_COLS), dtype=np.float32)
    meld_set = np.zeros((4, FEAT_COLS), dtype=np.float32)

    seat_discard_ranges = [MY_DISCARD_ROWS, *OPP_DISCARD_ROWS]
    seat_meld_ranges = [MY_MELD_ROWS, *OPP_MELD_ROWS]

    for seat_idx, discard_range in enumerate(seat_discard_ranges):
        for row_idx in discard_range:
            abs_row = BITSET_OFFSET + row_idx
            if abs_row < 0 or abs_row >= state_matrix.shape[0]:
                continue
            row = state_matrix[abs_row]
            if np.any(row != 0):
                tile_id = int(np.argmax(row))
                if 0 <= tile_id < FEAT_COLS:
                    discard_set[seat_idx, tile_id] = 1.0

    for seat_idx, meld_range in enumerate(seat_meld_ranges):
        for row_idx in meld_range:
            abs_row = BITSET_OFFSET + row_idx
            if abs_row < 0 or abs_row >= state_matrix.shape[0]:
                continue
            row = state_matrix[abs_row]
            if np.any(row != 0):
                tile_id = int(np.argmax(row))
                if 0 <= tile_id < FEAT_COLS:
                    meld_set[seat_idx, tile_id] = 1.0

    dora_start = STATIC_FEATURE_ROWS - DORA_FEATURE_SLOTS
    dora_block = np.zeros((DORA_FEATURE_SLOTS, FEAT_COLS), dtype=np.float32)
    if 0 <= dora_start < state_matrix.shape[0]:
        dora_usable = min(
            DORA_FEATURE_SLOTS, state_matrix.shape[0] - dora_start
        )
        if dora_usable > 0:
            dora_block[:dora_usable, :] = state_matrix[
                dora_start : dora_start + dora_usable, :
            ]

    my_hand_block = np.zeros((4, FEAT_COLS), dtype=np.float32)
    hand_start = BITSET_OFFSET + MY_HAND_ROWS.start
    if 0 <= hand_start < state_matrix.shape[0]:
        hand_usable = min(4, state_matrix.shape[0] - hand_start)
        if hand_usable > 0:
            my_hand_block[:hand_usable, :] = state_matrix[
                hand_start : hand_start + hand_usable, :
            ]

    return np.concatenate(
        [
            discard_set.reshape(-1),
            meld_set.reshape(-1),
            dora_block.reshape(-1),
            my_hand_block.reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def _build_new_add_step_visible_block(state_matrix: np.ndarray) -> np.ndarray:
    """Build per-action visible block for New_Add action tokens.

    Included information:
    - all 4 seats discard sets (4 x 34)
    - all 4 seats meld sets (4 x 34)
    """

    if (
        NEW_ADD_STEP_VISIBLE_DIM == 0
        or state_matrix is None
        or state_matrix.size == 0
    ):
        return np.zeros((0,), dtype=np.float32)

    discard_set = np.zeros((4, FEAT_COLS), dtype=np.float32)
    meld_set = np.zeros((4, FEAT_COLS), dtype=np.float32)

    seat_discard_ranges = [MY_DISCARD_ROWS, *OPP_DISCARD_ROWS]
    seat_meld_ranges = [MY_MELD_ROWS, *OPP_MELD_ROWS]

    for seat_idx, discard_range in enumerate(seat_discard_ranges):
        for row_idx in discard_range:
            abs_row = BITSET_OFFSET + row_idx
            if abs_row < 0 or abs_row >= state_matrix.shape[0]:
                continue
            row = state_matrix[abs_row]
            if np.any(row != 0):
                tile_id = int(np.argmax(row))
                if 0 <= tile_id < FEAT_COLS:
                    discard_set[seat_idx, tile_id] = 1.0

    for seat_idx, meld_range in enumerate(seat_meld_ranges):
        for row_idx in meld_range:
            abs_row = BITSET_OFFSET + row_idx
            if abs_row < 0 or abs_row >= state_matrix.shape[0]:
                continue
            row = state_matrix[abs_row]
            if np.any(row != 0):
                tile_id = int(np.argmax(row))
                if 0 <= tile_id < FEAT_COLS:
                    meld_set[seat_idx, tile_id] = 1.0

    return np.concatenate(
        [discard_set.reshape(-1), meld_set.reshape(-1)],
        axis=0,
    ).astype(np.float32)


def _visible_tile_counts(
    state_matrix: np.ndarray,
    *,
    include_private_hand: bool = True,
) -> np.ndarray:
    """Count tiles visible from the record player's perspective."""

    bitset_slice = state_matrix[
        BITSET_OFFSET : BITSET_OFFSET + BITSET_ROWS_TOTAL
    ]
    counts = np.zeros(FEAT_COLS, dtype=np.int32)

    def accumulate(rows: range) -> None:
        if rows.stop <= rows.start:
            return
        subset = bitset_slice[rows.start : rows.stop]
        if subset.size == 0:
            return
        counts[:] += subset.sum(axis=0, dtype=np.int32)

    if include_private_hand:
        accumulate(MY_HAND_ROWS)

    accumulate(MY_MELD_ROWS)
    for rows in OPP_MELD_ROWS:
        accumulate(rows)

    accumulate(MY_DISCARD_ROWS)
    for rows in OPP_DISCARD_ROWS:
        accumulate(rows)

    return counts


def _count_discards(bitset_matrix: np.ndarray, discard_range: range) -> int:
    """Count non-empty discard rows for a given seat."""

    return sum(1 for r in discard_range if np.any(bitset_matrix[r] != 0))


def _estimate_turn_number(bitset_matrix: np.ndarray) -> int:
    """Estimate current turn (巡目) from total filled discard slots across seats."""

    seat_counts = [
        _count_discards(bitset_matrix, MY_DISCARD_ROWS),
        *[_count_discards(bitset_matrix, rng) for rng in OPP_DISCARD_ROWS],
    ]
    if not seat_counts:
        return 0

    total_discards = sum(seat_counts)
    return total_discards


def _classify_phase_from_turn(turn_no: int, riichi_seen: bool = False) -> int:
    """Map total discards to phase id (early/mid/late)."""

    mid_phase = (TURN_MID_START <= turn_no < TURN_LATE_START) or (
        turn_no < TURN_LATE_START and riichi_seen
    )
    late_phase = turn_no >= TURN_LATE_START

    if late_phase:
        return PHASE_ID_LATE
    if mid_phase:
        return PHASE_ID_MID
    return PHASE_ID_EARLY


def _phase_id_from_state(state: np.ndarray, riichi_seen: bool = False) -> int:
    """Derive phase id from a token state (TOTAL_FEATURE_ROWS x 34)."""

    if state is None or state.size == 0:
        return PHASE_ID_EARLY

    bitset_slice = state[BITSET_OFFSET : BITSET_OFFSET + BITSET_ROWS_TOTAL, :]
    turn_no = _estimate_turn_number(bitset_slice)
    return _classify_phase_from_turn(turn_no, riichi_seen=riichi_seen)


def _build_phase_feature(
    bitset_matrix: np.ndarray, riichi_seen: bool = False
) -> np.ndarray:
    """Return phase block shaped (34, TIME_SECTION_ROWS).

    Mode 1 (legacy): first three tile columns encode early/mid/late as 1-hot.
    Mode 3 (new): three full columns; only one column is all 1s (early/mid/late).
    """

    if TIME_SECTION_ROWS == 0:
        return np.zeros((FEAT_COLS, 0), dtype=np.float32)

    turn_no = _estimate_turn_number(bitset_matrix)
    mid_phase = (TURN_MID_START <= turn_no < TURN_LATE_START) or (
        turn_no < TURN_LATE_START and riichi_seen
    )
    late_phase = turn_no >= TURN_LATE_START
    early_phase = not mid_phase and not late_phase

    phase_block = np.zeros((FEAT_COLS, TIME_SECTION_ROWS), dtype=np.float32)

    if TIME_SECTION_IN_METADATA == 1:
        phase_vec = np.zeros(FEAT_COLS, dtype=np.float32)
        if late_phase:
            phase_vec[2] = 1.0
        elif mid_phase:
            phase_vec[1] = 1.0
        else:
            phase_vec[0] = 1.0
        phase_block[:, 0] = phase_vec
        return phase_block

    # TIME_SECTION_IN_METADATA == 3
    if early_phase:
        phase_block[:, 0] = 1.0
    elif mid_phase:
        phase_block[:, 1] = 1.0
    else:
        phase_block[:, 2] = 1.0

    return phase_block


def build_action_tokens(
    base_matrix: np.ndarray, player_id: int
) -> Tuple[np.ndarray, int]:
    """依照棄牌順序建立 token 序列，每個 token 是 TOTAL_FEATURE_ROWS×34 flatten 後的向量。

    1. 第一個 token 為所有棄牌列清空（全 0）。
     2. 之後依照座位 [0,1,2,3] 迴圈，一次揭示一位玩家的「一組棄牌事件」
         （discard row + tsumogiri row 同步揭示），每張棄牌只產生 1 個 token。

    Returns:
        tokens: (seq_len, TOKEN_FEATURE_DIM)
        seq_len: 實際長度
    """

    discard_rows_sets = _collect_discard_rows(base_matrix, player_id)
    all_discard_rows = [
        row for rows in discard_rows_sets.values() for row in rows
    ]

    zero_state = base_matrix.copy()
    if all_discard_rows:
        zero_state[all_discard_rows] = 0

    max_actions = sum((len(rows) + 1) // 2 for rows in discard_rows_sets.values())
    tokens = np.empty(
        (1 + max_actions, zero_state.shape[0], zero_state.shape[1]),
        dtype=np.float32,
    )
    tokens[0] = zero_state
    token_count = 1
    current_state = zero_state.copy()

    pointers = {seat: 0 for seat in range(4)}
    seat_order = [0, 1, 2, 3]

    while True:
        progressed = False
        for seat in seat_order:
            rows = discard_rows_sets.get(seat, [])
            idx = pointers[seat]
            if idx < len(rows):
                discard_row = rows[idx]
                current_state[discard_row] = base_matrix[discard_row]

                # Rows are paired as [discard_row, tsumogiri_row, ...].
                # Reveal both rows in the same step so one discard => one token.
                if idx + 1 < len(rows):
                    tsumogiri_row = rows[idx + 1]
                    current_state[tsumogiri_row] = base_matrix[tsumogiri_row]
                    pointers[seat] += 2
                else:
                    pointers[seat] += 1

                tokens[token_count] = current_state
                token_count += 1
                progressed = True
        if not progressed:
            break

    return tokens[:token_count], token_count


def tile_string_to_id(tile_str: str) -> int:
    """
    將牌面字串轉換為數字編碼 (配合C++的編碼)
    萬子0-8, 筒子9-17, 索子18-26, 字牌27-33

    支援格式:
    - 數字牌: "1m", "5P" (大寫P表示赤5筒), "9s" 等
    - 字牌: "1z"(東), "2z"(南), "3z"(西), "4z"(北), "5z"(白), "6z"(發), "7z"(中)
    """
    # 處理 NaN 或非字串值
    if (
        pd.isna(tile_str)
        or not isinstance(tile_str, str)
        or not tile_str
        or len(tile_str) < 2
    ):
        return -1

    # 處理數字+字母格式
    if len(tile_str) == 2:
        num_char = tile_str[0]
        suit_char = tile_str[1].lower()

        try:
            num = int(num_char)
        except ValueError:
            return -1

        if suit_char == "m":  # 萬子
            if 1 <= num <= 9:
                return num - 1  # 0-8
        elif suit_char == "p":  # 筒子
            if 1 <= num <= 9:
                return 9 + num - 1  # 9-17
        elif suit_char == "s":  # 索子
            if 1 <= num <= 9:
                return 18 + num - 1  # 18-26
        elif suit_char == "z":  # 字牌
            if 1 <= num <= 7:
                return 27 + num - 1  # 27-33

    # 處理大寫P (赤5筒)
    elif len(tile_str) == 2 and tile_str[1] == "P":
        if tile_str[0] == "5":
            return 13  # 5筒的位置 (9 + 5 - 1)

    # 處理大寫M (赤5萬)
    elif len(tile_str) == 2 and tile_str[1] == "M":
        if tile_str[0] == "5":
            return 4  # 5萬的位置 (5 - 1)

    # 處理大寫S (赤5索)
    elif len(tile_str) == 2 and tile_str[1] == "S":
        if tile_str[0] == "5":
            return 22  # 5索的位置 (18 + 5 - 1)

    # 兼容舊格式的中文字牌 (如果需要)
    chinese_tiles = {
        "東": 27,
        "南": 28,
        "西": 29,
        "北": 30,
        "白": 31,
        "發": 32,
        "発": 32,
        "中": 33,
    }

    if tile_str in chinese_tiles:
        return chinese_tiles[tile_str]

    return -1  # 無效牌


def _reverse_sequence_with_padding(
    values: np.ndarray, pad_value: int
) -> np.ndarray:
    """Reverse non-pad entries and keep padding at the tail."""
    valid_mask = values != pad_value
    valid_values = values[valid_mask]
    if valid_values.size == 0:
        return values

    values[:] = pad_value
    values[: valid_values.size] = valid_values[::-1]
    return values


def _recent_first_sequence_with_padding(
    values: np.ndarray, pad_value: int, recent_count: int = 6
) -> np.ndarray:
    """Reorder so most recent N entries come first, followed by the rest.

    Example: if 10 tiles and recent_count=6:
      Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      Result:   [5, 6, 7, 8, 9, 10, 1, 2, 3, 4]
    """
    valid_mask = values != pad_value
    valid_values = values[valid_mask]
    n = valid_values.size
    if n == 0:
        return values

    if n <= recent_count:
        return values

    split_idx = n - recent_count
    reordered = np.concatenate(
        [valid_values[split_idx:], valid_values[:split_idx]]
    )

    values[:] = pad_value
    values[:n] = reordered
    return values


def _recent_reverse_sequence_with_padding(
    values: np.ndarray, pad_value: int, recent_count: int = 6
) -> np.ndarray:
    """Reorder to [latest..] + [older..], where latest block is reversed.

    Example: if 10 tiles and recent_count=6:
      Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      Result:   [10, 9, 8, 7, 6, 5, 1, 2, 3, 4]
    """
    valid_mask = values != pad_value
    valid_values = values[valid_mask]
    n = valid_values.size
    if n == 0:
        return values

    split_idx = max(0, n - recent_count)
    reordered = np.concatenate(
        [valid_values[split_idx:][::-1], valid_values[:split_idx]]
    )

    values[:] = pad_value
    values[:n] = reordered
    return values


def _normalize_discard_order_mode(mode: str) -> str:
    """Normalize mode labels like 'recent first'/'recent-first' to 'recent_first'."""
    return str(mode).strip().lower().replace("-", "_").replace(" ", "_")


def parse_tile_sequence(
    tile_str: str,
    max_length: int = 30,
    *,
    reverse_recent: bool = False,
) -> np.ndarray:
    """
    解析牌序列，返回固定長度的ID數組，不足補-1
    """
    # 處理 NaN 或非字串值
    if (
        pd.isna(tile_str)
        or not isinstance(tile_str, str)
        or not tile_str
        or tile_str.strip() == ""
    ):
        return np.full(max_length, -1, dtype=np.int32)

    tiles = tile_str.strip().split()
    tile_ids = np.full(max_length, -1, dtype=np.int32)

    for i, tile in enumerate(tiles):
        if i >= max_length:
            break
        tile_ids[i] = tile_string_to_id(tile)

    order_mode = _normalize_discard_order_mode(DISCARD_ORDER_MODE)
    if reverse_recent and order_mode != "original" and INPUT_FORMAT_IS_OLD:
        if order_mode == "reverse":
            tile_ids = _reverse_sequence_with_padding(tile_ids, pad_value=-1)
        elif order_mode == "recent_first":
            tile_ids = _recent_first_sequence_with_padding(
                tile_ids, pad_value=-1
            )
        elif order_mode == "recent_reverse":
            tile_ids = _recent_reverse_sequence_with_padding(
                tile_ids, pad_value=-1
            )

    return tile_ids


def parse_meld_sequence(meld_str: str, max_groups: int = 5) -> np.ndarray:
    """
    解析吃碰槓, 每組用3個特徵: [類型, 主牌ID, 輔助ID]
    類型: N=1(碰), C=2(吃), K=3(槓)
    """
    if (
        pd.isna(meld_str)
        or not isinstance(meld_str, str)
        or not meld_str
        or meld_str.strip() == ""
    ):
        return np.full(max_groups * 3, -1, dtype=np.int32)

    melds = meld_str.strip().split()
    meld_features = np.full(max_groups * 3, -1, dtype=np.int32)

    feature_idx = 0
    for meld in melds:
        if not meld or feature_idx >= max_groups * 3 - 2:
            break

        meld_type = meld[0]
        meld_content = meld[1:]

        if meld_type == "N":  # 碰
            # 例如: N西西 -> 碰西
            if len(meld_content) >= 2:
                tile = meld_content[: len(meld_content) // 2]
                tile_id = tile_string_to_id(tile)
                meld_features[feature_idx : feature_idx + 3] = [
                    1,
                    tile_id,
                    tile_id,
                ]
            else:
                meld_features[feature_idx : feature_idx + 3] = [1, -1, -1]

        elif meld_type == "C":  # 吃
            # 例如: C5p6p -> 吃5p6p(+7p)
            if len(meld_content) >= 4:
                tile1 = meld_content[:2]
                tile2 = meld_content[2:4]
                tile1_id = tile_string_to_id(tile1)
                tile2_id = tile_string_to_id(tile2)
                meld_features[feature_idx : feature_idx + 3] = [
                    2,
                    tile1_id,
                    tile2_id,
                ]
            else:
                meld_features[feature_idx : feature_idx + 3] = [2, -1, -1]

        elif meld_type == "K":  # 槓
            # 例如: K1m1m1m1m -> 槓1m
            if len(meld_content) >= 2:
                tile = meld_content[:2]
                tile_id = tile_string_to_id(tile)
                meld_features[feature_idx : feature_idx + 3] = [
                    3,
                    tile_id,
                    tile_id,
                ]
            else:
                meld_features[feature_idx : feature_idx + 3] = [3, -1, -1]
        else:
            meld_features[feature_idx : feature_idx + 3] = [-1, -1, -1]

        feature_idx += 3

    return meld_features


def parse_row_to_features(row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse a single row from the DataFrame from labeled.csv into features and label tensors.
    """
    # parse label
    label = int(row["label"])

    # parse basic game state metadata
    player_id = int(row["player_id"])
    round_num = int(row["round"])
    honba = int(row["honba"])
    scores = [int(row[f"score{i+1}"]) for i in range(4)]

    # parse dora 寶牌
    dora_id = -1 if pd.isna(row["dora"]) else tile_string_to_id(row["dora"])

    # 解析所有牌序
    my_hand = parse_tile_sequence(row["my_hand"], 14)
    my_discards = parse_tile_sequence(
        row["my_discards"], 30, reverse_recent=True
    )
    my_melds = parse_meld_sequence(row["my_melds"], 5)
    opp1_discards = parse_tile_sequence(
        row["opp1_discards"], 30, reverse_recent=True
    )
    opp1_melds = parse_meld_sequence(row["opp1_melds"], 5)
    opp2_discards = parse_tile_sequence(
        row["opp2_discards"], 30, reverse_recent=True
    )
    opp2_melds = parse_meld_sequence(row["opp2_melds"], 5)
    opp3_discards = parse_tile_sequence(
        row["opp3_discards"], 30, reverse_recent=True
    )
    opp3_melds = parse_meld_sequence(row["opp3_melds"], 5)

    # 組裝所有特徵 - 使用 numpy concatenate 代替 list extend
    game_state_features = np.array(
        [
            player_id,
            round_num,
            honba,
            dora_id,
            *scores,
            np.sum(my_hand != -1),  # 計算有效牌數
        ],
        dtype=np.int32,
    )

    # 使用 numpy concatenate 組合所有特徵
    all_features = np.concatenate(
        [
            my_hand,
            my_discards,
            my_melds,
            opp1_discards,
            opp1_melds,
            opp2_discards,
            opp2_melds,
            opp3_discards,
            opp3_melds,
            game_state_features,
        ]
    )

    # Tensor conversion
    feature_tensor = torch.tensor(all_features, dtype=torch.float32)
    label_tensor = torch.tensor(float(label), dtype=torch.float32)

    return feature_tensor, label_tensor
def _build_action_type_block(
    player_id: int, action_type: int, tile_id: int
) -> np.ndarray:
    """
    Build a one-hot action block with shape (F_ACT_ROWS, FEAT_COLS).

    Layout:
        - Rows are grouped by player, with ACTION_TYPE_COUNT rows per player.
        - Global row index = player_id * ACTION_TYPE_COUNT + action_type.
        - If tile_id ∈ [0, 34), set block[row_index, tile_id] = 1.0.
        - All other entries remain 0.

    Args:
        player_id: Seat index in [0, 3]. Rows outside range leave the block zeroed.
        action_type: Action category id in [0, ACTION_TYPE_COUNT).
        tile_id: Mahjong tile id; only valid indices mark the column hot.

    Returns:
        np.ndarray: Dense float32 matrix encoding the single action event.
    """

    block = np.zeros((F_ACT_ROWS, FEAT_COLS), dtype=np.float32)
    if 0 <= player_id < 4 and 0 <= action_type < ACTION_TYPE_COUNT:
        if 0 <= tile_id < FEAT_COLS:
            row = player_id * ACTION_TYPE_COUNT + action_type
            if row < F_ACT_ROWS:
                block[row, tile_id] = 1.0
    return block


def _build_additive_action_vector(
    player_id: int,
    action_type: int,
    tile_id: int,
    phase_vec: Optional[np.ndarray] = None,
    *,
    riichi_safe_flag: bool = False,
    dora_flag: bool = False,
    danger_flag: bool = False,
    one_chance_flag: bool = False,
    no_chance_flag: bool = False,
    opp_tenpai_flag: bool = False,
    opp_new_tenpai_flag: bool = False,
    v_attack: float = 0.5,
    v_defense: float = 0.5,
    step_visible_block: Optional[np.ndarray] = None,
    oracle_block: Optional[np.ndarray] = None,
    is_final_token: bool = False,
    final_context_block: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Encode action metadata as concatenated one-hots plus discard flags."""

    vec = np.zeros(F_ACT_ADD_DIM, dtype=np.float32)

    offset = 0
    if 0 <= player_id < NEW_ADD_PLAYER_ROWS:
        vec[player_id] = 1.0
    offset += NEW_ADD_PLAYER_ROWS

    if 0 <= action_type < ACTION_TYPE_COUNT:
        vec[offset + action_type] = 1.0
    offset += ACTION_TYPE_COUNT

    if 0 <= tile_id < FEAT_COLS:
        vec[offset + tile_id] = 1.0
    offset += NEW_ADD_TILE_ROWS

    if TIME_SECTION_IN_ACTION and TIME_SECTION_ROWS > 0:
        usable = min(
            TIME_SECTION_ROWS, 0 if phase_vec is None else phase_vec.size
        )
        if usable > 0 and phase_vec is not None:
            vec[offset : offset + usable] = phase_vec[:usable]
        offset += TIME_SECTION_ROWS
    if riichi_safe_flag:
        vec[offset] = 1.0
    offset += NEW_ADD_RIICHI_FLAG_ROWS

    if dora_flag:
        vec[offset] = 1.0
    offset += NEW_ADD_DORA_FLAG_ROWS

    if danger_flag:
        vec[offset] = 1.0
    offset += NEW_ADD_DANGER_FLAG_ROWS

    if one_chance_flag:
        vec[offset] = 1.0
    offset += NEW_ADD_ONE_CHANCE_FLAG_ROWS

    if no_chance_flag:
        vec[offset] = 1.0
    offset += NEW_ADD_NO_CHANCE_FLAG_ROWS

    vec[offset] = float(np.clip(v_attack, 0.0, 1.0))
    offset += NEW_ADD_ATTACK_EMBED_ROWS

    vec[offset] = float(np.clip(v_defense, 0.0, 1.0))
    offset += NEW_ADD_DEFENSE_EMBED_ROWS

    if NEW_ADD_STEP_VISIBLE_DIM > 0:
        if step_visible_block is not None and step_visible_block.size > 0:
            flat_visible = step_visible_block.reshape(-1).astype(
                np.float32, copy=False
            )
            usable = min(NEW_ADD_STEP_VISIBLE_DIM, flat_visible.size)
            vec[offset : offset + usable] = flat_visible[:usable]
        offset += NEW_ADD_STEP_VISIBLE_DIM

    if NEW_ADD_ORACLE_BLOCK_DIM > 0:
        if oracle_block is not None and oracle_block.size > 0:
            flat_oracle = oracle_block.reshape(-1).astype(
                np.float32, copy=False
            )
            usable = min(NEW_ADD_ORACLE_BLOCK_DIM, flat_oracle.size)
            vec[offset : offset + usable] = flat_oracle[:usable]
        offset += NEW_ADD_ORACLE_BLOCK_DIM

    if NEW_ADD_FINAL_DIM > 0:
        if is_final_token:
            vec[offset : offset + NEW_ADD_FINAL_DIM] = 1.0
        offset += NEW_ADD_FINAL_DIM

    if NEW_ADD_FINAL_CONTEXT_DIM > 0:
        if final_context_block is not None and final_context_block.size > 0:
            flat_ctx = final_context_block.reshape(-1).astype(
                np.float32, copy=False
            )
            usable = min(NEW_ADD_FINAL_CONTEXT_DIM, flat_ctx.size)
            vec[offset : offset + usable] = flat_ctx[:usable]
        offset += NEW_ADD_FINAL_CONTEXT_DIM

    if NEW_ADD_OG_TENPAI_FLAG_ROWS > 0:
        if opp_tenpai_flag:
            # 第一个值：1.0 表示OG已打开
            vec[offset] = 1.0
            # 第二个值：1.0 表示听牌
            vec[offset + 1] = 1.0
        else:
            # OG未打开时，第二个值设为0.5（未知状态）
            vec[offset + 1] = 0.5
        offset += NEW_ADD_OG_TENPAI_FLAG_ROWS

    if NEW_ADD_OG_NEW_TENPAI_FLAG_ROWS > 0:
        if opp_new_tenpai_flag:
            vec[offset] = 1.0
        offset += NEW_ADD_OG_NEW_TENPAI_FLAG_ROWS

    return vec


def _build_og_step_flags(
    valid_len: int,
    action_seq: Sequence[Tuple[int, ...]],
    target_seat: int,
    target_is_tenpai: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-step OG flags (after each action) for a target opponent."""

    tenpai_flags = np.zeros((valid_len,), dtype=np.float32)
    new_tenpai_flags = np.zeros((valid_len,), dtype=np.float32)

    if valid_len <= 1 or not target_is_tenpai:
        return tenpai_flags, new_tenpai_flags

    became_step = -1
    step_limit = min(len(action_seq), valid_len - 1)
    for step_idx in range(1, step_limit + 1):
        action_entry = action_seq[step_idx - 1]
        acting_player = int(action_entry[0]) if len(action_entry) > 0 else -1
        if acting_player == target_seat:
            became_step = step_idx
            break

    if became_step < 0:
        became_step = valid_len - 1

    tenpai_flags[became_step:valid_len] = 1.0
    new_tenpai_flags[became_step] = 1.0
    return tenpai_flags, new_tenpai_flags


def _encode_phase_additive(
    bitset_matrix: np.ndarray, riichi_seen: bool = False
) -> np.ndarray:
    """Encode early/mid/late into a small vector for additive action tokens."""

    if TIME_SECTION_ROWS == 0:
        return np.zeros((0,), dtype=np.float32)

    turn_no = _estimate_turn_number(bitset_matrix)
    mid_phase = (TURN_MID_START <= turn_no < TURN_LATE_START) or (
        turn_no < TURN_LATE_START and riichi_seen
    )
    late_phase = turn_no >= TURN_LATE_START
    early_phase = not mid_phase and not late_phase

    if TIME_SECTION_ROWS == 1:
        if late_phase:
            return np.array([1.0], dtype=np.float32)
        if mid_phase:
            return np.array([0.5], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    phase_vec = np.zeros(TIME_SECTION_ROWS, dtype=np.float32)
    if early_phase:
        phase_vec[0] = 1.0
    elif mid_phase:
        phase_vec[1] = 1.0
    else:
        phase_vec[2] = 1.0

    return phase_vec


# Pre-compute tile suit/rank lookup table for faster access
_TILE_SUIT_RANK_LUT = np.zeros((34, 2), dtype=np.int32)
for _tid in range(34):
    if _tid >= 27:
        _TILE_SUIT_RANK_LUT[_tid] = [3, -1]
    else:
        _TILE_SUIT_RANK_LUT[_tid] = [_tid // 9, (_tid % 9) + 1]


def _tile_suit_and_rank(tile_id: int) -> Tuple[int, int]:
    """Return (suit, rank) for tile_id. Optimized with lookup table.

    suit: 0=man, 1=pin, 2=sou, 3=honor; rank: 1-9 or -1 for honors/invalid.
    """
    if not (0 <= tile_id < 34):
        return -1, -1
    return int(_TILE_SUIT_RANK_LUT[tile_id, 0]), int(
        _TILE_SUIT_RANK_LUT[tile_id, 1]
    )


def _is_honor_tile(tile_id: int) -> bool:
    """Return True if tile is a honor tile (字牌)."""
    return 27 <= tile_id < 34


_ONE_CHANCE_TRIGGER_TO_TARGETS: Dict[int, Tuple[int, ...]] = {
    2: (1,),
    3: (1, 2),
    4: (2, 3),
    5: (3, 7),
    6: (7, 8),
    7: (8, 9),
    8: (9,),
}
_one_chance_target_sets: Dict[int, Set[int]] = {}
for blocker_rank, unlocked_ranks in _ONE_CHANCE_TRIGGER_TO_TARGETS.items():
    for target_rank in unlocked_ranks:
        bucket = _one_chance_target_sets.setdefault(target_rank, set())
        bucket.add(blocker_rank)
_ONE_CHANCE_TARGET_TO_TRIGGERS: Dict[int, Tuple[int, ...]] = {
    rank: tuple(sorted(blockers))
    for rank, blockers in _one_chance_target_sets.items()
}


def _is_one_chance_tile(tile_id: int, visible_counts: np.ndarray) -> bool:
    """Return True when the discard qualifies as a one-chance tile per kabe mapping."""

    if _is_honor_tile(tile_id):
        return False

    suit, rank = _tile_suit_and_rank(tile_id)
    if suit not in (0, 1, 2) or not (1 <= rank <= 9):
        return False

    triggers = _ONE_CHANCE_TARGET_TO_TRIGGERS.get(rank)
    if not triggers:
        return False

    base = suit * 9
    for blocker_rank in triggers:
        idx = base + (blocker_rank - 1)
        if 0 <= idx < FEAT_COLS and visible_counts[idx] >= 3:
            return True

    return False


def _build_action_metadata_from_final_state(
    final_state: np.ndarray, player_id: int
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    """Build action metadata directly from final discard rows, no dense diff pass."""
    if final_state is None or final_state.size == 0:
        return [], []

    seat_rows = _collect_discard_rows(final_state, player_id)
    pointers = {seat: 0 for seat in range(4)}
    seat_order = [0, 1, 2, 3]
    actions: List[Tuple[int, int, int]] = []
    action_rows: List[Tuple[int, int]] = []

    while True:
        progressed = False
        for seat in seat_order:
            rows = seat_rows.get(seat, [])
            idx = pointers[seat]
            if idx >= len(rows):
                continue

            discard_row = int(rows[idx])
            tsumogiri_row = int(rows[idx + 1]) if idx + 1 < len(rows) else -1
            tile_vec = final_state[discard_row]
            tile_id = int(np.argmax(tile_vec)) if np.any(tile_vec > 0) else -1
            action_type_id = ACTION_TYPE_TO_ID[8]
            if 0 <= tsumogiri_row < final_state.shape[0]:
                is_tedashi = bool(np.any(final_state[tsumogiri_row] != 0))
                action_type_id = (
                    ACTION_TYPE_TO_ID[8] if is_tedashi else ACTION_TYPE_TO_ID[9]
                )

            actions.append((int(seat), tile_id, action_type_id))
            action_rows.append((discard_row, tsumogiri_row))
            pointers[seat] = idx + (2 if idx + 1 < len(rows) else 1)
            progressed = True
        if not progressed:
            break

    return actions, action_rows


def _extract_action_sequence_and_discard_rows(
    states: np.ndarray, player_id: int
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    """Infer action tuples and discard-row metadata in one diff pass."""
    if states.shape[0] <= 1:
        return [], []

    final_state = states[-1]
    seat_rows = _collect_discard_rows(final_state, player_id)
    actions: List[Tuple[int, int, int]] = []
    action_rows: List[Tuple[int, int]] = []

    seat_discard_row_sets: Dict[int, Set[int]] = {
        seat: set(rows[0::2]) for seat, rows in seat_rows.items()
    }
    discard_to_tsumogiri_row: Dict[int, int] = {}
    for rows in seat_rows.values():
        for discard_row, tsumogiri_row in zip(rows[0::2], rows[1::2]):
            discard_to_tsumogiri_row[int(discard_row)] = int(tsumogiri_row)

    # Vectorized diff computation for all steps at once
    # shape: (seq_len-1, TOTAL_FEATURE_ROWS, FEAT_COLS)
    diffs = np.diff(states, axis=0)

    for step in range(1, states.shape[0]):
        diff = diffs[step - 1]
        curr = states[step]

        # Find changed rows (vectorized)
        changed_mask = np.any(diff != 0, axis=1)
        changed_rows = np.where(changed_mask)[0]

        acting_player = -1
        discard_row = -1
        for seat, discard_row_set in seat_discard_row_sets.items():
            for idx in changed_rows:
                if int(idx) in discard_row_set:
                    acting_player = int(seat)
                    discard_row = int(idx)
                    break
            if discard_row != -1:
                break

        if discard_row == -1:
            for idx in changed_rows:
                if np.any(curr[idx] != 0):
                    discard_row = int(idx)
                    break

        if discard_row == -1:
            actions.append((-1, -1, -1))
            action_rows.append((-1, -1))
            continue

        tile_vec = curr[discard_row]
        tile_id = int(np.argmax(tile_vec)) if np.any(tile_vec > 0) else -1

        action_type_id = ACTION_TYPE_TO_ID[8]
        tsumogiri_row = discard_to_tsumogiri_row.get(discard_row)
        if tsumogiri_row is not None and 0 <= tsumogiri_row < curr.shape[0]:
            # tsumogiri rows are encoded as all-1 for tedashi, all-0 for tsumogiri.
            is_tedashi = bool(np.any(curr[tsumogiri_row] != 0))
            action_type_id = (
                ACTION_TYPE_TO_ID[8] if is_tedashi else ACTION_TYPE_TO_ID[9]
            )

        actions.append((acting_player, tile_id, action_type_id))
        action_rows.append((discard_row, int(tsumogiri_row) if tsumogiri_row is not None else -1))

    return actions, action_rows


def _extract_action_sequence(
    states: np.ndarray, player_id: int
) -> List[Tuple[int, int, int]]:
    """Infer (player_id, tile_id, action_type_id) for each revealed action step.
    Optimized with vectorized diff detection.

    Args:
        states: Token states with shape (seq_len, TOTAL_FEATURE_ROWS, FEAT_COLS).
        player_id: Seat id used to anchor `_collect_discard_rows` alignment.

    Returns:
        List of tuples, one per action step (t ≥ 1):
        (acting_player, tile_id, action_type_id).
    """
    actions, _ = _extract_action_sequence_and_discard_rows(states, player_id)
    return actions


def _extract_action_discard_rows(
    states: np.ndarray, player_id: int
) -> List[Tuple[int, int]]:
    """Return (discard_row, tsumogiri_row) per action step."""
    _, action_rows = _extract_action_sequence_and_discard_rows(states, player_id)
    return action_rows


def _build_old_single_recent_action_state(
    states: np.ndarray,
    player_id: int,
    recent_count: int,
    action_rows: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Build old_single state with only the most recent N actions kept."""
    if states.size == 0:
        return states

    final_state = states[-1].copy()
    if recent_count <= 0:
        return final_state

    if action_rows is None:
        action_rows = _extract_action_discard_rows(states, player_id)
    if not action_rows:
        return final_state

    keep_rows: Set[int] = set()
    for discard_row, tsumogiri_row in action_rows[-recent_count:]:
        if discard_row >= 0:
            keep_rows.add(discard_row)
        if tsumogiri_row >= 0:
            keep_rows.add(tsumogiri_row)

    seat_rows = _collect_discard_rows(final_state, player_id)
    for rows in seat_rows.values():
        for discard_row, tsumogiri_row in zip(rows[0::2], rows[1::2]):
            if int(discard_row) not in keep_rows:
                final_state[discard_row] = 0
                final_state[tsumogiri_row] = 0

    return final_state


def _reserved_final_token_count() -> int:
    """Return the number of non-action New_Add tail tokens."""

    append_final_token = INPUT_FORMAT_IS_NEW_ADD and NEW_ADD_FINAL_DIM > 0
    append_final_context_token = (
        INPUT_FORMAT_IS_NEW_ADD and NEW_ADD_FINAL_CONTEXT_DIM > 0
    )
    return (1 if append_final_context_token else 0) + (
        1 if append_final_token else 0
    )


def get_effective_max_valid_tokens(mode: Optional[str] = None) -> int:
    """Return the maximum non-padding token count after token placement."""

    placement_mode = TOKEN_PLACEMENT_MODE if mode is None else mode
    placement_mode = _TOKEN_PLACEMENT_MODE_ALIASES.get(
        placement_mode, placement_mode
    )
    if placement_mode == "last_n_actions":
        reserved_final_tokens = _reserved_final_token_count()
        action_slots = max(
            0, MAX_SEQ_LEN_FALLBACK - 1 - reserved_final_tokens
        )
        kept_actions = min(
            int(TOKEN_PLACEMENT_LAST_ACTION_COUNT), action_slots
        )
        return min(
            MAX_SEQ_LEN_FALLBACK,
            1 + kept_actions + reserved_final_tokens,
        )
    return MAX_SEQ_LEN_FALLBACK


def _select_action_window(
    actions: Sequence[Tuple[int, int, int]], max_action_tokens: int
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Select action tokens for packing and return their source offset."""

    safe_max = max(0, int(max_action_tokens))
    if safe_max <= 0 or not actions:
        return (
            [],
            len(actions) if TOKEN_PLACEMENT_MODE == "last_n_actions" else 0,
        )

    if TOKEN_PLACEMENT_MODE == "last_n_actions":
        keep_count = min(
            int(TOKEN_PLACEMENT_LAST_ACTION_COUNT), safe_max, len(actions)
        )
        start_idx = max(0, len(actions) - keep_count)
        return list(actions[start_idx : start_idx + keep_count]), start_idx

    return list(actions[:safe_max]), 0


def _select_action_window_for_valid_tokens(
    actions: Sequence[Tuple[int, int, int]], valid_len: int
) -> List[Tuple[int, int, int]]:
    """Select the action sequence portion represented by a packed sample."""

    reserved = _reserved_final_token_count()
    action_slots = max(0, int(valid_len) - 1 - reserved)
    selected, _ = _select_action_window(actions, action_slots)
    return selected


def _reverse_new_add_action_segment(
    values: np.ndarray,
    valid_len: int,
) -> np.ndarray:
    """Reverse only New_Add action positions, preserving special tokens."""

    if not (INPUT_FORMAT_IS_NEW_ADD and REVERSE_NEW_ADD_ACTION_TOKENS):
        return values

    safe_valid_len = max(0, min(int(valid_len), int(values.shape[0])))
    reserved = min(
        _reserved_final_token_count(), max(0, safe_valid_len - 1)
    )
    action_end = max(1, safe_valid_len - reserved)
    if action_end <= 2:
        return values

    reversed_values = values.copy()
    reversed_values[1:action_end] = values[1:action_end][::-1]
    return reversed_values


def _build_init_token_vector(
    states: np.ndarray,
    init_state_idx: int,
    *,
    riichi_seen: bool = False,
    refresh_phase_from_state: bool = False,
    player_id: Optional[int] = None,
    action_rows: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Build the flattened heterogeneous init token."""

    init_feature_len = F_INIT
    init_vec = np.zeros((init_feature_len,), dtype=np.float32)
    if states.size == 0:
        return init_vec

    safe_idx = max(0, min(int(init_state_idx), states.shape[0] - 1))
    init_state = states[safe_idx]
    init_block = np.zeros((F_INIT_ROWS, FEAT_COLS), dtype=np.float32)

    static_rows = min(STATIC_FEATURE_ROWS, init_state.shape[0], F_INIT_ROWS)
    if static_rows > 0:
        init_block[:static_rows, :] = init_state[:static_rows, :]

    if (
        refresh_phase_from_state
        and TIME_SECTION_IN_METADATA
        and not TIME_SECTION_IN_ACTION
        and TIME_SECTION_ROWS > 0
    ):
        bitset_start = BITSET_OFFSET
        bitset_stop = min(
            bitset_start + BITSET_ROWS_TOTAL, init_state.shape[0]
        )
        phase_row_start = STATIC_PHASE_ROW_OFFSET
        if bitset_stop > bitset_start and phase_row_start < F_INIT_ROWS:
            phase_rows = _build_phase_feature(
                init_state[bitset_start:bitset_stop, :],
                riichi_seen=riichi_seen,
            ).T
            usable_phase_rows = min(
                TIME_SECTION_ROWS,
                phase_rows.shape[0],
                F_INIT_ROWS - phase_row_start,
            )
            if usable_phase_rows > 0:
                init_block[
                    phase_row_start : phase_row_start + usable_phase_rows, :
                ] = phase_rows[:usable_phase_rows, :]

    if NEW_ADD_INIT_MY_HAND_ROWS > 0:
        hand_src_start = BITSET_OFFSET + MY_HAND_ROWS.start
        hand_src_stop = min(
            BITSET_OFFSET + MY_HAND_ROWS.stop, init_state.shape[0]
        )
        hand_dst_start = STATIC_FEATURE_ROWS
        hand_rows = min(
            NEW_ADD_INIT_MY_HAND_ROWS,
            max(0, hand_src_stop - hand_src_start),
            max(0, F_INIT_ROWS - hand_dst_start),
        )
        if hand_rows > 0:
            init_block[
                hand_dst_start : hand_dst_start + hand_rows, :
            ] = init_state[hand_src_start : hand_src_start + hand_rows, :]

    if ORACLE_FEATURE_ROWS > 0:
        start = ORACLE_FEATURE_OFFSET
        stop = min(start + ORACLE_FEATURE_ROWS, init_state.shape[0])
        usable = min(
            max(0, stop - start),
            max(0, F_INIT_ROWS - NEW_ADD_INIT_ORACLE_OFFSET_ROWS),
        )
        if usable > 0:
            init_block[
                NEW_ADD_INIT_ORACLE_OFFSET_ROWS : NEW_ADD_INIT_ORACLE_OFFSET_ROWS
                + usable,
                :,
            ] = init_state[start : start + usable, :]

    init_flat = init_block.reshape(-1)
    init_vec[: min(init_flat.size, init_feature_len)] = init_flat[
        :init_feature_len
    ]

    if NEW_ADD_INIT_EXTRA_DIM > 0 and init_flat.size < init_feature_len:
        extra_state = (
            init_state
            if TOKEN_PLACEMENT_MODE == "last_n_actions"
            else states[-1]
        )
        extra_parts: List[np.ndarray] = []
        if USE_NEW_ADD_INIT_EXTRA:
            extra_parts.append(_build_new_add_init_extra(extra_state))
        if NEW_ADD_INIT_OLD_SINGLE_RECENT_DIM > 0:
            extra_parts.append(
                _build_new_add_init_old_single_recent_extra(
                    states,
                    int(player_id) if player_id is not None else -1,
                    TOKEN_PLACEMENT_LAST_ACTION_COUNT,
                    action_rows=action_rows,
                )
            )

        if extra_parts:
            init_extra = np.concatenate(extra_parts, axis=0)
            usable = min(init_extra.size, init_feature_len - init_flat.size)
            if usable > 0:
                init_vec[
                    init_flat.size : init_flat.size + usable
                ] = init_extra[:usable]

    return init_vec


def _pack_token_sequence(
    states: np.ndarray,
    player_id: int,
    action_type: int,
    fallback_tile_id: int,
    riichi_seen: bool = False,
    *,
    riichi_seats: Optional[Sequence[int]] = None,
    dora_ids: Optional[Sequence[int]] = None,
    action_metadata: Optional[
        Tuple[Sequence[Tuple[int, int, int]], Sequence[Tuple[int, int]]]
    ] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pack heterogeneous tokens plus an optional init-only sidecar.

    Returns the compact padded sequence, init sidecar, and valid-token count.
    """

    packed = np.zeros(
        (MAX_SEQ_LEN_FALLBACK, PACKED_ACTION_FEATURE_DIM), dtype=np.float32
    )
    init_sidecar = np.zeros((NEW_ADD_INIT_SIDECAR_DIM,), dtype=np.float32)
    if states.size == 0:
        return packed, init_sidecar, 0

    if INPUT_FORMAT_IS_OLD_SINGLE:
        final_state = states[-1]
        flat = final_state.reshape(-1).astype(np.float32)
        usable = min(flat.size, PACKED_ACTION_FEATURE_DIM)
        packed[0, :usable] = flat[:usable]
        return packed, init_sidecar, 1

    if INPUT_FORMAT_IS_OLD_SEQUENCE:
        seq_len = min(states.shape[0], MAX_SEQ_LEN_FALLBACK)
        for idx in range(seq_len):
            flat = states[idx].reshape(-1).astype(np.float32)
            usable = min(flat.size, PACKED_ACTION_FEATURE_DIM)
            packed[idx, :usable] = flat[:usable]
        return packed, init_sidecar, seq_len

    if action_metadata is None:
        actions, action_rows = _extract_action_sequence_and_discard_rows(
            states, player_id
        )
    else:
        actions = list(action_metadata[0])
        action_rows = list(action_metadata[1])
    append_final_token = INPUT_FORMAT_IS_NEW_ADD and NEW_ADD_FINAL_DIM > 0
    append_final_context_token = (
        INPUT_FORMAT_IS_NEW_ADD and NEW_ADD_FINAL_CONTEXT_DIM > 0
    )
    reserved_final_tokens = _reserved_final_token_count()

    max_action_tokens = MAX_SEQ_LEN_FALLBACK - 1 - reserved_final_tokens
    if max_action_tokens < 0:
        max_action_tokens = 0
    truncated_actions, action_state_offset = _select_action_window(
        actions, max_action_tokens
    )
    truncated_action_rows = action_rows[
        action_state_offset : action_state_offset + len(truncated_actions)
    ]

    init_state_idx = action_state_offset
    if (
        TOKEN_PLACEMENT_MODE == "last_n_actions"
        and TOKEN_PLACEMENT_LAST_ACTION_INIT_STATE == "final"
    ):
        init_state_idx = states.shape[0] - 1
    refresh_init_phase = TOKEN_PLACEMENT_MODE == "last_n_actions"
    init_vec = _build_init_token_vector(
        states,
        init_state_idx,
        riichi_seen=riichi_seen,
        refresh_phase_from_state=refresh_init_phase,
        player_id=player_id,
        action_rows=truncated_action_rows,
    )
    compact_init = init_vec[:F_INIT_COMPACT]
    packed[0, : compact_init.size] = compact_init
    if USE_NEW_ADD_INIT_SIDECAR:
        sidecar_start = F_INIT_COMPACT
        sidecar_stop = sidecar_start + NEW_ADD_INIT_SIDECAR_DIM
        init_sidecar[:] = init_vec[sidecar_start:sidecar_stop]

    if not actions and reserved_final_tokens == 0:
        return packed, init_sidecar, 1

    if truncated_actions and fallback_tile_id >= 0:
        last_player, _, last_action_type = truncated_actions[-1]
        truncated_actions[-1] = (
            last_player,
            fallback_tile_id,
            last_action_type,
        )

    if truncated_actions and 0 <= action_type < ACTION_TYPE_COUNT:
        last_player, last_tile, _ = truncated_actions[-1]
        truncated_actions[-1] = (last_player, last_tile, action_type)

    dora_set: Set[int] = set()
    if dora_ids is not None:
        dora_set = {
            int(tile)
            for tile in dora_ids
            if isinstance(tile, (int, np.integer))
            and 0 <= int(tile) < FEAT_COLS
        }

    riichi_target_seats: Set[int] = set(riichi_seats or [])
    riichi_tile_bank: Set[int] = set()
    visibility_cache: Dict[int, np.ndarray] = {}

    for idx, (player_step, tile_step, action_type_step) in enumerate(
        truncated_actions, start=1
    ):
        phase_block_rows: Optional[np.ndarray] = None
        phase_vec_add: Optional[np.ndarray] = None
        state_idx = min(action_state_offset + idx, states.shape[0] - 1)

        if TIME_SECTION_IN_ACTION and TIME_SECTION_ROWS > 0:
            bitset_slice = states[
                state_idx,
                BITSET_OFFSET : BITSET_OFFSET + BITSET_ROWS_TOTAL,
                :,
            ]
            if INPUT_FORMAT_IS_NEW_MULTIPLY:
                phase_block_rows = _build_phase_feature(
                    bitset_slice, riichi_seen=riichi_seen
                ).T
            elif INPUT_FORMAT_IS_NEW_ADD:
                phase_vec_add = _encode_phase_additive(
                    bitset_slice, riichi_seen=riichi_seen
                )
        oracle_block = None
        if NEW_ADD_ORACLE_BLOCK_DIM > 0:
            start = ORACLE_FEATURE_OFFSET
            stop = min(start + ORACLE_FEATURE_ROWS, states.shape[1])
            if stop > start:
                oracle_block = states[state_idx, start:stop, :]
        is_discard_action = action_type_step in DISCARD_ACTION_IDS
        valid_tile = 0 <= tile_step < FEAT_COLS

        if INPUT_FORMAT_IS_NEW_MULTIPLY:
            block = _build_action_type_block(
                player_step, action_type_step, tile_step
            )

            if (
                TIME_SECTION_IN_ACTION
                and TIME_SECTION_ROWS > 0
                and phase_block_rows is not None
            ):
                start_row = block.shape[0] - TIME_SECTION_ROWS
                if start_row >= 0 and phase_block_rows.size > 0:
                    block[
                        start_row : start_row + phase_block_rows.shape[0], :
                    ] = phase_block_rows

            packed[idx, :F_ACT] = block.reshape(-1)
        elif INPUT_FORMAT_IS_NEW_ADD:
            step_visible_block = None
            if NEW_ADD_STEP_VISIBLE_DIM > 0:
                step_visible_block = _build_new_add_step_visible_block(
                    states[state_idx]
                )

            riichi_flag = bool(
                is_discard_action
                and valid_tile
                and tile_step in riichi_tile_bank
            )
            dora_flag = bool(
                is_discard_action and valid_tile and tile_step in dora_set
            )
            prev_counts: Optional[np.ndarray] = None
            prev_seen: Optional[int] = None
            if valid_tile and states.shape[0] > 0:
                state_before_idx = max(
                    0,
                    min(
                        action_state_offset + idx - 1,
                        states.shape[0] - 1,
                    ),
                )
                cached_counts = visibility_cache.get(state_before_idx)
                if cached_counts is None:
                    cached_counts = _visible_tile_counts(
                        states[state_before_idx],
                        include_private_hand=True,
                    )
                    visibility_cache[state_before_idx] = cached_counts
                prev_counts = cached_counts
                prev_seen = int(cached_counts[tile_step])
            one_chance_flag = bool(
                is_discard_action
                and valid_tile
                and prev_counts is not None
                and _is_one_chance_tile(tile_step, prev_counts)
            )
            no_chance_flag = bool(
                is_discard_action
                and valid_tile
                and prev_seen is not None
                and prev_seen >= 4
            )

            danger_flag = bool(
                is_discard_action
                and valid_tile
                and bool(riichi_target_seats)
                and not riichi_flag
                and not one_chance_flag
                and not no_chance_flag
            )

            if danger_flag:
                v_at_embed, v_df_embed = 1.0, 0.0
            else:
                v_at_embed, v_df_embed = 0.0, 1.0

            packed[idx, :F_ACT_ADD_DIM] = _build_additive_action_vector(
                player_step,
                action_type_step,
                tile_step,
                phase_vec=phase_vec_add,
                riichi_safe_flag=riichi_flag,
                dora_flag=dora_flag,
                danger_flag=danger_flag,
                one_chance_flag=one_chance_flag,
                no_chance_flag=no_chance_flag,
                v_attack=v_at_embed,
                v_defense=v_df_embed,
                step_visible_block=step_visible_block,
                oracle_block=oracle_block,
            )

            if (
                is_discard_action
                and valid_tile
                and player_step in riichi_target_seats
            ):
                riichi_tile_bank.add(tile_step)
        else:
            raise RuntimeError(
                "Unsupported input format while packing action tokens."
            )

    valid_tokens = 1 + len(truncated_actions)
    if append_final_context_token and valid_tokens < MAX_SEQ_LEN_FALLBACK:
        final_context_block = _build_new_add_final_context_block(states[-1])
        packed[valid_tokens, :F_ACT_ADD_DIM] = _build_additive_action_vector(
            -1,
            -1,
            -1,
            final_context_block=final_context_block,
        )
        valid_tokens += 1

    if append_final_token and valid_tokens < MAX_SEQ_LEN_FALLBACK:
        packed[valid_tokens, :F_ACT_ADD_DIM] = _build_additive_action_vector(
            -1,
            -1,
            -1,
            is_final_token=True,
        )
        valid_tokens += 1

    packed = _reverse_new_add_action_segment(packed, valid_tokens)
    return packed, init_sidecar, valid_tokens


# ==============================================================================
# MTP (Masked Tile Prediction) Masking Functions
# ==============================================================================
def apply_mtp_masking(
    packed_tokens: np.ndarray,
    valid_len: int,
    attention_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply BERT-style masking to action tokens for MTP pre-training.

    Args:
        packed_tokens: Token sequence of shape (seq_len, feature_dim).
        valid_len: Number of valid tokens (including init token at position 0).
        attention_mask: Optional mask with 1.0 on valid token positions. Use
            this when valid tokens are not packed contiguously.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        masked_tokens: Token sequence with masking applied.
        mask_positions: Boolean array of shape (seq_len,) indicating masked positions.
        tile_labels: Integer array of shape (seq_len,) with original tile IDs
                     (-1 for non-masked positions).

    Note:
        - Position 0 (init token) is never masked.
        - Only valid action-token positions are candidates for masking.
        - The tile ID is extracted from the NEW_ADD_TILE_ROWS portion of each action token.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq_len, feature_dim = packed_tokens.shape
    masked_tokens = packed_tokens.copy()
    mask_positions = np.zeros(seq_len, dtype=bool)
    tile_labels = np.full(seq_len, -1, dtype=np.int32)

    # Only mask action tokens (skip init token at position 0).
    if attention_mask is not None:
        valid_positions = np.flatnonzero(np.asarray(attention_mask) > 0)
        action_indices = valid_positions[valid_positions != 0]
    else:
        safe_valid_len = max(0, min(int(valid_len), seq_len))
        action_indices = np.arange(1, safe_valid_len)

    num_action_tokens = int(action_indices.size)
    if num_action_tokens < MTP_MIN_ACTIONS:
        return masked_tokens, mask_positions, tile_labels

    # Calculate number of tokens to mask
    num_to_mask = max(1, int(num_action_tokens * MTP_MASK_RATIO))

    # Randomly select valid action positions to mask.
    mask_indices = rng.choice(
        action_indices,
        size=min(num_to_mask, len(action_indices)),
        replace=False,
    )

    # Tile feature offset within action token (for New_Add format)
    tile_offset = NEW_ADD_PLAYER_ROWS + NEW_ADD_ACTION_ROWS
    tile_end = tile_offset + NEW_ADD_TILE_ROWS

    for idx in mask_indices:
        # Extract original tile ID from one-hot encoding
        tile_vec = packed_tokens[idx, tile_offset:tile_end]
        original_tile_id = (
            int(np.argmax(tile_vec)) if np.any(tile_vec > 0) else -1
        )

        if original_tile_id < 0 or original_tile_id >= MTP_NUM_TILE_CLASSES:
            continue

        mask_positions[idx] = True
        tile_labels[idx] = original_tile_id

        # Determine replacement strategy
        rand_val = rng.random()
        if rand_val < MTP_REPLACE_WITH_MASK:
            # 80%: Replace tile features with zeros (mask token)
            masked_tokens[idx, tile_offset:tile_end] = 0.0
        elif rand_val < MTP_REPLACE_WITH_MASK + MTP_REPLACE_WITH_RANDOM:
            # 10%: Replace with random tile
            random_tile = rng.integers(0, MTP_NUM_TILE_CLASSES)
            masked_tokens[idx, tile_offset:tile_end] = 0.0
            masked_tokens[idx, tile_offset + random_tile] = 1.0
        # else: 10%: Keep original (no change)

    return masked_tokens, mask_positions, tile_labels


def _rng_random(rng: Any) -> float:
    return float(rng.random())


def _rng_randint_inclusive(rng: Any, low: int, high: int) -> int:
    if high <= low:
        return int(low)
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))
    return int(rng.randint(low, high + 1))


def _seed_mix32(seed: int, value: int) -> int:
    """32-bit mix function for deterministic seed composition."""
    return (
        seed
        ^ (
            value
            + 0x9E3779B9
            + ((seed << 6) & 0xFFFFFFFF)
            + (seed >> 2)
        )
    ) & 0xFFFFFFFF


def _seed_from_part(part: Any) -> int:
    """Convert supported objects into a stable 32-bit integer seed part."""
    if isinstance(part, np.generic):
        part = part.item()

    if isinstance(part, (int, np.integer, bool)):
        return int(part) & 0xFFFFFFFF

    if isinstance(part, (float, np.floating)):
        return int(round(float(part) * 1_000_000.0)) & 0xFFFFFFFF

    if isinstance(part, bytes):
        h = 0
        for b in part:
            h = ((h * 131) + int(b)) & 0xFFFFFFFF
        return h

    if isinstance(part, str):
        return _seed_from_part(part.encode("utf-8", errors="ignore"))

    return _seed_from_part(str(part))


def _get_global_seed_base(default_seed: int = 42) -> int:
    """Use run-global seed from environment when available."""
    raw = os.environ.get("PYTHONHASHSEED", "").strip()
    if not raw:
        return int(default_seed) & 0xFFFFFFFF
    try:
        return int(raw) & 0xFFFFFFFF
    except ValueError:
        return int(default_seed) & 0xFFFFFFFF


def make_deterministic_sample_rng(*parts: Any) -> np.random.Generator:
    """Create a deterministic RNG keyed by global seed + sample parts."""
    seed = _get_global_seed_base()
    for part in parts:
        seed = _seed_mix32(seed, _seed_from_part(part))
    return np.random.default_rng(seed)


def _even_token_positions(valid_count: int, seq_len: int) -> np.ndarray:
    """Return target positions for evenly spreading valid tokens."""
    safe_count = max(0, min(int(valid_count), int(seq_len)))
    if safe_count <= 0:
        return np.empty((0,), dtype=np.int64)
    if safe_count == 1:
        return np.array([0], dtype=np.int64)

    span = int(seq_len) - 1
    action_count = safe_count - 1
    action_steps = np.arange(1, action_count + 1, dtype=np.int64)
    action_positions = (
        (action_steps * span + (action_count // 2)) // action_count
    )
    return np.concatenate(
        [np.array([0], dtype=np.int64), action_positions.astype(np.int64)]
    )


def _head_tail_even_token_positions(
    valid_count: int, seq_len: int
) -> np.ndarray:
    """Return positions with init first and action head/tail anchored."""
    safe_count = max(0, min(int(valid_count), int(seq_len)))
    if safe_count <= 0:
        return np.empty((0,), dtype=np.int64)
    if safe_count == 1:
        return np.array([0], dtype=np.int64)

    action_count = safe_count - 1
    if action_count == 1:
        return np.array([0, int(seq_len) - 1], dtype=np.int64)

    span = int(seq_len) - 2
    action_steps = np.arange(action_count, dtype=np.int64)
    action_positions = 1 + (
        (action_steps * span) // (action_count - 1)
    )
    return np.concatenate(
        [np.array([0], dtype=np.int64), action_positions.astype(np.int64)]
    )


def _last_six_tail_token_positions(
    valid_count: int, seq_len: int
) -> np.ndarray:
    """Return positions with the last six action tokens anchored at the tail."""
    safe_count = max(0, min(int(valid_count), int(seq_len)))
    if safe_count <= 0:
        return np.empty((0,), dtype=np.int64)
    if safe_count == 1:
        return np.array([0], dtype=np.int64)

    action_count = safe_count - 1
    tail_count = min(6, action_count, max(int(seq_len) - 1, 0))
    front_count = action_count - tail_count

    front_positions = np.arange(1, front_count + 1, dtype=np.int64)
    tail_positions = np.arange(
        int(seq_len) - tail_count, int(seq_len), dtype=np.int64
    )
    return np.concatenate(
        [
            np.array([0], dtype=np.int64),
            front_positions,
            tail_positions,
        ]
    )


def apply_token_placement(
    packed_tokens: np.ndarray,
    mask_arr: np.ndarray,
    *,
    mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Apply the configured token placement mode to tokens and mask."""
    placement_mode = TOKEN_PLACEMENT_MODE if mode is None else mode
    placement_mode = _TOKEN_PLACEMENT_MODE_ALIASES.get(
        placement_mode, placement_mode
    )
    if placement_mode in ("contiguous", "last_n_actions"):
        return packed_tokens, mask_arr, int(np.count_nonzero(mask_arr > 0))
    if placement_mode not in _TOKEN_PLACEMENT_MODE_CHOICES:
        raise ValueError(f"Unsupported token placement mode: {placement_mode}")

    seq_len = int(packed_tokens.shape[0])
    valid_indices = np.flatnonzero(mask_arr > 0)
    if valid_indices.size <= 0 or seq_len <= 0:
        return packed_tokens, mask_arr, 0

    valid_indices = valid_indices[:seq_len]
    if placement_mode == "even":
        target_positions = _even_token_positions(valid_indices.size, seq_len)
    elif placement_mode == "head_tail_even":
        target_positions = _head_tail_even_token_positions(
            valid_indices.size, seq_len
        )
    else:
        target_positions = _last_six_tail_token_positions(
            valid_indices.size, seq_len
        )
    out_tokens = np.zeros_like(packed_tokens)
    out_mask = np.zeros_like(mask_arr)

    token_count = min(valid_indices.size, target_positions.size)
    if token_count > 0:
        src = valid_indices[:token_count]
        dst = target_positions[:token_count]
        out_tokens[dst] = packed_tokens[src]
        out_mask[dst] = 1.0

    return out_tokens, out_mask, token_count


def apply_length_perturbation(
    packed_tokens: np.ndarray,
    valid_len: int,
    *,
    rng: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Apply train-time length perturbation to one sample.

    Techniques:
      (a) random cropping (on action-token segment)
      (b) random padding via masked dummy tokens
      (c) token dropout (on action-token segment)

    Returns:
      perturbed_tokens: (MAX_SEQ_LEN_FALLBACK, ACTION_FEATURE_DIM)
      mask_arr: float mask with 1.0 on valid positions (not necessarily contiguous)
      perturbed_valid_len: int(mask_arr.sum())
    """
    if rng is None:
        rng = np.random

    seq = packed_tokens.copy()
    mask_arr = np.zeros(MAX_SEQ_LEN_FALLBACK, dtype=np.float32)
    safe_valid_len = max(0, min(int(valid_len), MAX_SEQ_LEN_FALLBACK))
    if safe_valid_len <= 0:
        return seq, mask_arr, 0

    # Short-circuit when disabled or not selected this time.
    if (
        (not LENGTH_PERTURBATION)
        or safe_valid_len <= 1
        or _rng_random(rng) > LENGTH_PERTURBATION_PROB
    ):
        mask_arr[:safe_valid_len] = 1.0
        return seq, mask_arr, safe_valid_len

    append_final_token = INPUT_FORMAT_IS_NEW_ADD and NEW_ADD_FINAL_DIM > 0
    append_final_context_token = (
        INPUT_FORMAT_IS_NEW_ADD and NEW_ADD_FINAL_CONTEXT_DIM > 0
    )
    reserved_final_tokens = (1 if append_final_context_token else 0) + (
        1 if append_final_token else 0
    )
    usable_reserved = min(reserved_final_tokens, max(0, safe_valid_len - 1))

    action_start = 1
    action_end = max(action_start, safe_valid_len - usable_reserved)
    action_tokens = seq[action_start:action_end].copy()
    tail_tokens = seq[action_end:safe_valid_len].copy()  # final-context/final

    # Token dropout on action tokens
    if (
        LENGTH_PERTURB_TOKEN_DROPOUT
        and action_tokens.shape[0] > 1
        and _rng_random(rng) <= LENGTH_PERTURB_TOKEN_DROPOUT_PROB
    ):
        keep_mask = (
            rng.random(action_tokens.shape[0])
            >= float(LENGTH_PERTURB_TOKEN_DROPOUT_RATE)
        )
        if not np.any(keep_mask):
            keep_mask[
                _rng_randint_inclusive(rng, 0, action_tokens.shape[0] - 1)
            ] = True
        action_tokens = action_tokens[keep_mask]

    # Random cropping on action tokens (contiguous crop)
    if (
        LENGTH_PERTURB_RANDOM_CROP
        and action_tokens.shape[0] > 1
        and _rng_random(rng) <= LENGTH_PERTURB_RANDOM_CROP_PROB
    ):
        curr_n = int(action_tokens.shape[0])
        min_keep = max(1, min(int(LENGTH_PERTURB_MIN_ACTION_KEEP), curr_n))
        crop_keep = _rng_randint_inclusive(rng, min_keep, curr_n)
        if crop_keep < curr_n:
            crop_start = _rng_randint_inclusive(rng, 0, curr_n - crop_keep)
            action_tokens = action_tokens[crop_start : crop_start + crop_keep]

    # Rebuild sequence (keep init token fixed at index 0)
    out = np.zeros_like(seq)
    out[0] = seq[0]

    base_after_init = action_tokens.shape[0] + tail_tokens.shape[0]
    max_action_offset = max(0, MAX_SEQ_LEN_FALLBACK - 1 - base_after_init)
    action_offset = 0
    if (
        LENGTH_PERTURB_RANDOM_PADDING
        and max_action_offset > 0
        and _rng_random(rng) <= LENGTH_PERTURB_RANDOM_PADDING_PROB
    ):
        action_offset = _rng_randint_inclusive(rng, 0, max_action_offset)

    action_pos_start = 1 + action_offset
    action_pos_end = action_pos_start + action_tokens.shape[0]
    tail_pos_end = action_pos_end + tail_tokens.shape[0]

    if action_tokens.shape[0] > 0:
        out[action_pos_start:action_pos_end] = action_tokens
    if tail_tokens.shape[0] > 0:
        out[action_pos_end:tail_pos_end] = tail_tokens

    mask_arr[0] = 1.0
    if action_tokens.shape[0] > 0:
        mask_arr[action_pos_start:action_pos_end] = 1.0
    if tail_tokens.shape[0] > 0:
        mask_arr[action_pos_end:tail_pos_end] = 1.0

    return out, mask_arr, int(mask_arr.sum())


def _simulate_perturbed_length_for_turn(
    valid_len: int, rng: np.random.Generator
) -> int:
    """Simulate perturbed sequence length for ratio matching (no token changes)."""
    safe_valid_len = max(0, min(int(valid_len), int(MAX_SEQ_LEN_FALLBACK)))
    if safe_valid_len <= 1:
        return safe_valid_len
    if (not LENGTH_PERTURBATION) or rng.random() > float(LENGTH_PERTURBATION_PROB):
        return safe_valid_len

    append_final_token = INPUT_FORMAT_IS_NEW_ADD and NEW_ADD_FINAL_DIM > 0
    append_final_context_token = (
        INPUT_FORMAT_IS_NEW_ADD and NEW_ADD_FINAL_CONTEXT_DIM > 0
    )
    reserved_final_tokens = (1 if append_final_context_token else 0) + (
        1 if append_final_token else 0
    )
    usable_reserved = min(reserved_final_tokens, max(0, safe_valid_len - 1))

    action_len = max(0, safe_valid_len - 1 - usable_reserved)

    if (
        LENGTH_PERTURB_TOKEN_DROPOUT
        and action_len > 1
        and rng.random() <= float(LENGTH_PERTURB_TOKEN_DROPOUT_PROB)
    ):
        keep_mask = rng.random(action_len) >= float(LENGTH_PERTURB_TOKEN_DROPOUT_RATE)
        if not np.any(keep_mask):
            keep_mask[rng.integers(0, action_len)] = True
        action_len = int(np.sum(keep_mask))

    if (
        LENGTH_PERTURB_RANDOM_CROP
        and action_len > 1
        and rng.random() <= float(LENGTH_PERTURB_RANDOM_CROP_PROB)
    ):
        min_keep = max(1, min(int(LENGTH_PERTURB_MIN_ACTION_KEEP), action_len))
        crop_keep = int(rng.integers(min_keep, action_len + 1))
        action_len = crop_keep

    return int(min(MAX_SEQ_LEN_FALLBACK, 1 + action_len + usable_reserved))


def _match_turn_label_ratios_with_perturbation(
    turns: np.ndarray,
    labels: np.ndarray,
    seed: int,
    min_total: int,
) -> np.ndarray:
    """Return a boolean mask to drop samples to match perturbed ratios."""
    turns_arr = np.asarray(turns, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int8)
    if turns_arr.size == 0 or labels_arr.size == 0:
        return np.ones_like(labels_arr, dtype=bool)

    rng = np.random.default_rng(int(seed))
    pert_turns = np.array(
        [_simulate_perturbed_length_for_turn(t, rng) for t in turns_arr],
        dtype=np.int64,
    )

    orig_df = pd.DataFrame({"turn": turns_arr, "label": labels_arr})
    pert_df = pd.DataFrame({"turn": pert_turns, "label": labels_arr})

    orig_pivot = orig_df.groupby(["turn", "label"]).size().unstack(fill_value=0)
    pert_pivot = pert_df.groupby(["turn", "label"]).size().unstack(fill_value=0)

    all_turns = np.union1d(orig_pivot.index.values, pert_pivot.index.values)
    orig_pivot = orig_pivot.reindex(all_turns, fill_value=0)
    pert_pivot = pert_pivot.reindex(all_turns, fill_value=0)

    pert_totals = pert_pivot.sum(axis=1).replace(0, np.nan)
    pert_ratios = pert_pivot.div(pert_totals, axis=0).fillna(0.0)

    keep_mask = np.zeros(labels_arr.shape[0], dtype=bool)
    index_arr = np.arange(labels_arr.shape[0], dtype=np.int64)

    pos_ratio_series = (
        pert_ratios[1] if 1 in pert_ratios.columns else pd.Series(0.0, index=all_turns)
    )

    for turn in all_turns:
        turn_int = int(turn)
        target_pos_ratio = float(pos_ratio_series.get(turn, 0.0))
        if target_pos_ratio < 0.0:
            target_pos_ratio = 0.0
        if target_pos_ratio > 1.0:
            target_pos_ratio = 1.0

        turn_mask = turns_arr == turn_int
        if not np.any(turn_mask):
            continue

        idx_turn = index_arr[turn_mask]
        labels_turn = labels_arr[turn_mask]
        pos_idx = idx_turn[labels_turn == 1]
        neg_idx = idx_turn[labels_turn == 0]

        orig_pos = int(pos_idx.size)
        orig_neg = int(neg_idx.size)
        orig_total = orig_pos + orig_neg

        if orig_total <= 0:
            continue
        if orig_total < int(min_total):
            keep_mask[idx_turn] = True
            continue

        if target_pos_ratio <= 0.0:
            keep_pos = 0
            keep_neg = orig_neg
        elif target_pos_ratio >= 1.0:
            keep_pos = orig_pos
            keep_neg = 0
        else:
            max_total_by_pos = orig_pos / target_pos_ratio
            max_total_by_neg = orig_neg / (1.0 - target_pos_ratio)
            keep_total = int(np.floor(min(max_total_by_pos, max_total_by_neg)))
            keep_total = max(0, min(orig_total, keep_total))
            keep_pos = int(np.round(keep_total * target_pos_ratio))
            keep_pos = max(0, min(orig_pos, keep_pos))
            keep_neg = max(0, min(orig_neg, keep_total - keep_pos))

        if keep_pos > 0:
            keep_pos_idx = rng.choice(pos_idx, size=keep_pos, replace=False)
            keep_mask[keep_pos_idx] = True
        if keep_neg > 0:
            keep_neg_idx = rng.choice(neg_idx, size=keep_neg, replace=False)
            keep_mask[keep_neg_idx] = True

    return keep_mask


__all__ = [name for name in globals() if not name.startswith("__")]
