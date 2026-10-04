"""Private Python Sidecar API router (/internal/v1/*) for PyTorch model inference and session management.
"""

from __future__ import annotations

import os
import json
import uuid
import time
import threading
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status, Header
import torch

from model_engine.compatibility import (
    load_checkpoint_strictly,
    describe_analysis_capabilities,
)
from model_engine.transformer_model import TransformerModelEngine
from model_engine.resources import ModelStore, compute_gate, resource_error, dispose_engine

sidecar_router = APIRouter(prefix="/internal/v1", tags=["private_sidecar"])


# --- Schemas ---
class SidecarErrorBody(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    recovery: List[str] = Field(default_factory=list)
    evidence: List[Any] = Field(default_factory=list)

class SidecarErrorEnvelope(BaseModel):
    error: SidecarErrorBody

class LoadModelRequest(BaseModel):
    checkpoint_path: str
    manifest_spec: Optional[Dict[str, Any]] = None
    requested_phase: Optional[str] = None
    device: str = "cpu"

class LoadModelResponse(BaseModel):
    model_handle: str
    input_schema: Optional[Dict[str, Any]] = None
    layer_count: int
    status: str = "ok"
    model_name: str
    phase: Optional[str] = None
    device: str
    capabilities: Dict[str, Any]

class CreateSessionRequest(BaseModel):
    input_data: List[Any]
    requested_probes: List[str] = Field(default_factory=list)
    token_idx: int = 0

class SessionSummary(BaseModel):
    session_id: str
    created_at: float
    model_name: str
    output_shape: List[int]
    output_sample: List[float]
    output_kind: Literal["raw"] = "raw"
    probes_captured: List[str]

class ActivationsResponse(BaseModel):
    session_id: str
    activations: Dict[str, Any]

class AnalyzeActivationsRequest(BaseModel):
    input_data: Optional[List[Any]] = None
    use_dummy_input: bool = True

class AnalyzeActivationsResponse(BaseModel):
    status: str
    torchlens_history: Optional[Dict[str, Any]] = None



# Results are owned by a model handle and bounded without evicting other users.
class SessionStore:
    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 100,
                 max_bytes: int = 256 * 1024 * 1024):
        if min(ttl_seconds, max_sessions, max_bytes) <= 0:
            raise ValueError("Session limits must be positive")
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self.max_bytes = max_bytes
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()

    def cleanup_expired(self):
        with self.lock:
            now = time.monotonic()
            for sid, sess in list(self.sessions.items()):
                if now - sess["touched"] >= self.ttl:
                    del self.sessions[sid]

    def create(self, model_handle: str, model_name: str, output_data: Dict[str, Any],
               activations: Dict[str, Any], probes: List[str]) -> str:
        payload = {"model_name": model_name, "output": output_data,
                   "activations": activations, "probes": probes}
        size = len(json.dumps(payload).encode("utf-8"))
        with self.lock:
            self.cleanup_expired()
            if len(self.sessions) >= self.max_sessions or sum(s["size"] for s in self.sessions.values()) + size > self.max_bytes:
                raise resource_error(503, "RESULT_CAPACITY", "分析結果額度已滿，請釋放舊結果後重試。")
            sid = f"sess_{uuid.uuid4().hex}"
            self.sessions[sid] = {**payload, "session_id": sid, "model_handle": model_handle,
                                  "created_at": time.time(), "touched": time.monotonic(), "size": size}
            return sid

    def get(self, session_id: str, model_handle: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            self.cleanup_expired()
            sess = self.sessions.get(session_id)
            if sess and sess["model_handle"] == model_handle:
                sess["touched"] = time.monotonic()
                return sess
            return None

    def delete(self, session_id: str, model_handle: str) -> bool:
        with self.lock:
            sess = self.sessions.get(session_id)
            if sess and sess["model_handle"] == model_handle:
                del self.sessions[session_id]
                return True
            return False

    def delete_owner(self, model_handle: str):
        with self.lock:
            for sid, sess in list(self.sessions.items()):
                if sess["model_handle"] == model_handle:
                    del self.sessions[sid]

    def prune_owners(self, active_handles):
        # Snapshot owners while holding the result lock so a newly published
        # model's result cannot race with a stale owner snapshot.
        with self.lock:
            valid = active_handles()
            for sid, sess in list(self.sessions.items()):
                if sess["model_handle"] not in valid:
                    del self.sessions[sid]


_MODEL_STORE = ModelStore(
    ttl_seconds=float(os.getenv("ANALYZER_MODEL_TTL_SECONDS", "1800")),
    max_models=int(os.getenv("ANALYZER_MAX_MODELS", "4")),
    max_bytes=int(os.getenv("ANALYZER_MODEL_MEMORY_MB", "2048")) * 1024 * 1024,
)
_SESSION_STORE = SessionStore(
    ttl_seconds=int(os.getenv("ANALYZER_RESULT_TTL_SECONDS", "3600")),
    max_sessions=int(os.getenv("ANALYZER_MAX_RESULTS", "100")),
    max_bytes=int(os.getenv("ANALYZER_RESULT_MEMORY_MB", "256")) * 1024 * 1024,
)


def cleanup_resources():
    _MODEL_STORE.cleanup_expired()
    _SESSION_STORE.cleanup_expired()
    _SESSION_STORE.prune_owners(_MODEL_STORE.handles)


def _summary(sess):
    return SessionSummary(session_id=sess["session_id"], created_at=sess["created_at"],
        model_name=sess["model_name"], output_shape=sess["output"].get("output_shape", []),
        output_sample=sess["output"].get("output_sample", []), probes_captured=sess["probes"])


def _session(session_id, model_handle):
    sess = _SESSION_STORE.get(session_id, model_handle)
    if sess is None:
        raise resource_error(404, "SESSION_EXPIRED", "分析結果已到期或不屬於此模型，請重新分析。")
    return sess


@sidecar_router.get("/health")
def internal_health(model_handle: Optional[str] = Header(default=None, alias="X-Model-Handle")):
    info = {"status": "ok", "model_loaded": False, "model_name": None,
            "device": None, "cuda_available": torch.cuda.is_available()}
    if model_handle:
        with _MODEL_STORE.lease(model_handle) as engine:
            info.update(model_loaded=engine.is_loaded, model_name=engine.model_class_name, device=engine.device)
    return info


@sidecar_router.post("/models/load", response_model=LoadModelResponse)
def internal_load_model(req: LoadModelRequest,
                        model_handle: Optional[str] = Header(default=None, alias="X-Model-Handle")):
    metadata = {}
    def build():
        engine = TransformerModelEngine(isolated=True)
        try:
            model, spec, phase = load_checkpoint_strictly(
                checkpoint_path=req.checkpoint_path, manifest_spec=req.manifest_spec,
                requested_phase=req.requested_phase, device=req.device)
            engine._model = model
            engine._model_class_name = spec.name
            engine._device = req.device
            engine._is_loaded = True
            model.eval()
            metadata.update(model_name=spec.name, phase=phase, device=req.device,
                capabilities=describe_analysis_capabilities(spec.name), input_schema=engine.get_input_schema(),
                layer_count=sum(type(m).__name__ in ("MultiHeadAttention", "MultiheadAttention") for m in model.modules()))
            return engine
        except Exception:
            dispose_engine(engine)
            raise
    try:
        with compute_gate.slot():
            handle = _MODEL_STORE.load(build, model_handle)
            if model_handle:
                _SESSION_STORE.delete_owner(model_handle)
            return LoadModelResponse(model_handle=handle, **metadata)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise resource_error(404, "CHECKPOINT_NOT_FOUND", str(exc)) from exc
    except ValueError as exc:
        raise resource_error(400, "MODEL_BUILD_FAILED", str(exc)) from exc
    except Exception as exc:
        raise resource_error(500, "MODEL_BUILD_FAILED", str(exc)) from exc


@sidecar_router.delete("/models/current")
def internal_delete_model(model_handle: Optional[str] = Header(default=None, alias="X-Model-Handle")):
    _MODEL_STORE.delete(model_handle)
    _SESSION_STORE.delete_owner(model_handle)
    return {"status": "deleted"}


@sidecar_router.post("/sessions", response_model=SessionSummary)
def internal_create_session(req: CreateSessionRequest,
                            model_handle: Optional[str] = Header(default=None, alias="X-Model-Handle")):
    with _MODEL_STORE.lease(model_handle) as engine, compute_gate.slot():
        try:
            out = engine.forward(req.input_data)
            if "error" in out:
                raise ValueError(out["error"])
            activations = {}
            if "attention" in req.requested_probes or not req.requested_probes:
                for i in range(len(engine._captured_attention)):
                    attention = engine.get_attention_tensor(i)
                    if attention is not None:
                        # Same two-dimensional head/batch average as shared workspaces.
                        matrix = attention
                        while matrix.ndim > 2:
                            matrix = matrix.mean(dim=0)
                        activations[f"attention_layer_{i}"] = matrix.tolist()
            probes = ["attention"] if activations else []
            sid = _SESSION_STORE.create(model_handle, engine.model_class_name, out, activations, probes)
            return _summary(_session(sid, model_handle))
        except HTTPException:
            raise
        except Exception as exc:
            raise resource_error(400, "INFERENCE_FAILED", str(exc)) from exc
        finally:
            engine.clear_attention_cache()


@sidecar_router.get("/sessions/{session_id}", response_model=SessionSummary)
def internal_get_session(session_id: str,
                         model_handle: Optional[str] = Header(default=None, alias="X-Model-Handle")):
    with _MODEL_STORE.lease(model_handle):
        return _summary(_session(session_id, model_handle))


@sidecar_router.get("/sessions/{session_id}/activations", response_model=ActivationsResponse)
def internal_get_activations(session_id: str,
                             model_handle: Optional[str] = Header(default=None, alias="X-Model-Handle")):
    with _MODEL_STORE.lease(model_handle):
        return ActivationsResponse(session_id=session_id, activations=_session(session_id, model_handle)["activations"])


@sidecar_router.delete("/sessions/{session_id}")
def internal_delete_session(session_id: str,
                            model_handle: Optional[str] = Header(default=None, alias="X-Model-Handle")):
    with _MODEL_STORE.lease(model_handle):
        if not _SESSION_STORE.delete(session_id, model_handle):
            raise resource_error(404, "SESSION_EXPIRED", "分析結果已到期或不屬於此模型。")
    return {"status": "deleted", "session_id": session_id}


@sidecar_router.post("/analyze_activations", response_model=AnalyzeActivationsResponse)
def analyze_activations(req: AnalyzeActivationsRequest,
                        model_handle: Optional[str] = Header(default=None, alias="X-Model-Handle")):
    with _MODEL_STORE.lease(model_handle) as engine, compute_gate.slot():
        try:
            input_tensor = None
            if not req.use_dummy_input and req.input_data is not None:
                input_tensor = torch.tensor(req.input_data, dtype=torch.float32, device=engine.device)
                if input_tensor.ndim == 2:
                    input_tensor = input_tensor.unsqueeze(0)
            result = engine.analyze_with_torchlens(input_tensor=input_tensor)
            if "error" in result:
                raise ValueError(result["error"])
            return AnalyzeActivationsResponse(status=result["status"], torchlens_history=result.get("torchlens_history"))
        except Exception as exc:
            raise resource_error(400, "TORCHLENS_ANALYSIS_FAILED", str(exc)) from exc
        finally:
            engine.clear_attention_cache()
