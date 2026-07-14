"""Read individual Transformer samples from round/chunk ``.bin`` files."""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()
_PACK_CONFIG_LOCK = threading.Lock()


class SearchRequest(BaseModel):
    directory: str
    file_name: str = ""
    line_number: Optional[int] = Field(default=None, ge=0)
    record_index: Optional[int] = Field(default=None, ge=0)
    pred_id: int = Field(default=0, ge=0, le=2)
    seq_len: Optional[int] = Field(default=None, ge=1)
    feature_dim: Optional[int] = Field(default=None, ge=1)


def _utils_module():
    repo_root = Path(__file__).resolve().parents[3]
    transformer_root = repo_root / "Transformer"
    if str(transformer_root) not in sys.path:
        sys.path.insert(0, str(transformer_root))
    return importlib.import_module("utils")


def _safe_directory(raw_path: str) -> Path:
    directory = Path(os.path.abspath(os.path.expanduser(raw_path)))
    if not directory.is_dir() and not Path(raw_path).is_absolute():
        repo_candidate = Path(__file__).resolve().parents[3] / raw_path
        directory = repo_candidate.resolve()
    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    return directory


def _bin_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.bin"), key=lambda p: p.name)


def _read_info_line(info_path: Path, line_number: int) -> str:
    """Read a zero-based info.txt line without retaining the huge file in memory."""
    if not info_path.is_file():
        return ""
    with info_path.open("rb") as stream:
        for current, raw_line in enumerate(stream):
            if current == line_number:
                return raw_line.decode("utf-8", errors="replace").strip()
            if current > line_number:
                break
    return ""


def _tenhou_url(raw_json: str) -> str:
    if not raw_json:
        return ""
    try:
        payload = json.loads(raw_json)
        raw_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        pass
    return "https://tenhou.net/6/#json=" + urllib.parse.quote(raw_json, safe="") + "&ts=0"


def _pack_for_model_schema(
    utils: Any,
    states: np.ndarray,
    player_id: int,
    action_type: int,
    tile_id: int,
    doras: tuple[int, ...],
    target_seq_len: Optional[int],
    target_feature_dim: Optional[int],
) -> tuple[np.ndarray, int, str]:
    """Pack with the loaded model's physical length and restore dense New_Add init data."""
    with _PACK_CONFIG_LOCK:
        original_seq_len = utils.MAX_SEQ_LEN_FALLBACK
        original_action_count = utils.TOKEN_PLACEMENT_LAST_ACTION_COUNT
        try:
            if target_seq_len is not None:
                utils.MAX_SEQ_LEN_FALLBACK = int(target_seq_len)
                if utils.TOKEN_PLACEMENT_MODE == "last_n_actions":
                    utils.TOKEN_PLACEMENT_LAST_ACTION_COUNT = max(1, int(target_seq_len) - 1)
            packed, sidecar, valid_len = utils._pack_token_sequence(
                states.astype(np.float32), player_id, action_type, tile_id,
                dora_ids=doras,
            )
        finally:
            utils.MAX_SEQ_LEN_FALLBACK = original_seq_len
            utils.TOKEN_PLACEMENT_LAST_ACTION_COUNT = original_action_count

    conversion = "native"
    if target_feature_dim is not None and packed.shape[1] != target_feature_dim:
        can_restore_dense_new_add = (
            bool(utils.INPUT_FORMAT_IS_NEW_ADD)
            and int(target_feature_dim) == int(utils.F_INIT)
            and packed.shape[1] == int(utils.F_INIT_COMPACT)
            and sidecar.size == int(utils.NEW_ADD_INIT_SIDECAR_DIM)
        )
        if can_restore_dense_new_add:
            dense = np.zeros((packed.shape[0], int(target_feature_dim)), dtype=np.float32)
            dense[:, : packed.shape[1]] = packed
            start = int(utils.F_INIT_COMPACT)
            dense[0, start : start + sidecar.size] = sidecar
            packed = dense
            conversion = "new_add_compact_to_dense"

    return packed, int(valid_len), conversion


@router.get("/files")
async def list_files(directory: str) -> dict[str, Any]:
    try:
        root = _safe_directory(directory)
        files = _bin_files(root)
        return {
            "success": True,
            "directory": str(root),
            "files": [{"name": p.name, "size": p.stat().st_size} for p in files],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "files": []}


@router.post("/search")
async def search_sample(request: SearchRequest) -> dict[str, Any]:
    try:
        root = _safe_directory(request.directory)
        files = _bin_files(root)
        if request.file_name:
            safe_name = Path(request.file_name).name
            files = [p for p in files if p.name == safe_name]
        if not files:
            raise ValueError("No matching .bin file found")

        utils = _utils_module()
        match: tuple[Path, int, int, bytes] | None = None
        for path in files:
            raw = path.read_bytes()
            record_size = utils.detect_record_size_from_length(len(raw))
            if not record_size:
                continue
            count = len(raw) // record_size
            indices = range(count)
            if request.line_number is None and request.record_index is not None:
                indices = [request.record_index] if request.record_index < count else []
            for index in indices:
                start = index * record_size
                record = raw[start : start + record_size]
                line_number = int.from_bytes(record[:4], "little", signed=False)
                if request.line_number is None or line_number == request.line_number:
                    match = (path, index, record_size, record)
                    break
            if match:
                break
        if match is None:
            raise ValueError("No sample matched the requested line/record")

        path, record_index, record_size, record = match
        parsed = utils.parse_single_binary_record(record, record_index)
        states, labels, line_number, _, player_id, doras, action_type, _, action_tiles, _ = parsed
        tile_id = next((int(tile) for tile in action_tiles if int(tile) >= 0), -1)
        packed, valid_len, conversion = _pack_for_model_schema(
            utils, states, int(player_id), int(action_type), tile_id,
            tuple(int(tile) for tile in doras), request.seq_len, request.feature_dim,
        )
        init_base = int(utils.F_INIT_BASE)
        sample = packed.copy()
        init = sample[0, :init_base].reshape(int(utils.F_INIT_ROWS), int(utils.FEAT_COLS))
        init[:3, :] = 0.0
        init[request.pred_id, :] = 1.0
        sample[0, :init_base] = init.reshape(-1)

        expected = (request.seq_len, request.feature_dim)
        actual = tuple(int(v) for v in sample.shape)
        compatible = expected[0] is None or expected[1] is None or actual == expected
        raw_json = _read_info_line(root / "info.txt", int(line_number))
        target_seat = (int(player_id) + request.pred_id + 1) % 4
        return {
            "success": True,
            "file_name": path.name,
            "record_index": record_index,
            "record_size": record_size,
            "line_number": int(line_number),
            "player_id": int(player_id),
            "pred_id": request.pred_id,
            "target_seat": target_seat,
            "seat_encoding": "absolute_with_relative_target_index",
            "label": int(labels[request.pred_id]),
            "valid_len": int(valid_len),
            "conversion": conversion,
            "shape": list(actual),
            "compatible": compatible,
            "expected_shape": list(expected),
            "feature": sample.tolist() if compatible else None,
            "tenhou_url": _tenhou_url(raw_json),
            "warning": "" if compatible else f"Parsed shape {actual} does not match loaded model shape {expected}.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
