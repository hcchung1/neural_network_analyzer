"""Private Python Sidecar API router (/internal/v1/*) for PyTorch model inference and session management.
"""

from __future__ import annotations

import os
import uuid
import time
import threading
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status, Depends
import torch

from model_engine.compatibility import (
    load_checkpoint_strictly,
    describe_capabilities,
    ModelBuildSpec,
    ModelCapabilities,
)
from model_engine.transformer_model import get_model_engine

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
    probes_captured: List[str]

class ActivationsResponse(BaseModel):
    session_id: str
    activations: Dict[str, Any]


# --- Session Store (In-Memory with TTL & Eviction) ---
class SessionStore:
    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 100):
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()

    def cleanup_expired(self):
        now = time.time()
        with self.lock:
            expired = [sid for sid, sess in self.sessions.items() if now - sess["created_at"] > self.ttl]
            for sid in expired:
                del self.sessions[sid]

    def create(self, model_name: str, output_data: Dict[str, Any], activations: Dict[str, Any], probes: List[str]) -> str:
        self.cleanup_expired()
        with self.lock:
            if len(self.sessions) >= self.max_sessions:
                # Evict oldest
                oldest_sid = min(self.sessions.keys(), key=lambda k: self.sessions[k]["created_at"])
                del self.sessions[oldest_sid]

            session_id = f"sess_{uuid.uuid4().hex[:12]}"
            self.sessions[session_id] = {
                "session_id": session_id,
                "created_at": time.time(),
                "model_name": model_name,
                "output": output_data,
                "activations": activations,
                "probes": probes,
            }
            return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        self.cleanup_expired()
        with self.lock:
            sess = self.sessions.get(session_id)
            if sess:
                sess["created_at"] = time.time()  # Touch LRU
            return sess

    def delete(self, session_id: str) -> bool:
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False

_SESSION_STORE = SessionStore()


# --- Endpoints ---

@sidecar_router.get("/health")
async def internal_health():
    engine = get_model_engine()
    return {
        "status": "ok",
        "model_loaded": engine.is_loaded,
        "model_name": engine.model_class_name,
        "device": engine.device,
        "cuda_available": torch.cuda.is_available(),
    }


@sidecar_router.post("/models/load", response_model=LoadModelResponse)
async def internal_load_model(req: LoadModelRequest):
    engine = get_model_engine()
    try:
        model, spec, phase = load_checkpoint_strictly(
            checkpoint_path=req.checkpoint_path,
            manifest_spec=req.manifest_spec,
            requested_phase=req.requested_phase,
            device=req.device,
        )
        engine._model = model
        engine._model_class_name = spec.name
        engine._device = req.device
        engine._is_loaded = True

        capabilities = describe_capabilities(spec.name)

        return LoadModelResponse(
            status="ok",
            model_name=spec.name,
            phase=phase,
            device=req.device,
            capabilities=capabilities.to_dict(),
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=SidecarErrorEnvelope(error=SidecarErrorBody(
                code="ARTIFACT_OUTSIDE_ALLOWED_ROOT",
                message=str(e),
            )).model_dump()
        )
    except ValueError as e:
        err_str = str(e)
        code = "STATE_DICT_SHAPE_MISMATCH"
        if ":" in err_str:
            code = err_str.split(":")[0].strip()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=SidecarErrorEnvelope(error=SidecarErrorBody(
                code=code,
                message=err_str,
            )).model_dump()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=SidecarErrorEnvelope(error=SidecarErrorBody(
                code="MODEL_BUILD_FAILED",
                message=str(e),
            )).model_dump()
        )


@sidecar_router.post("/sessions", response_model=SessionSummary)
async def internal_create_session(req: CreateSessionRequest):
    engine = get_model_engine()
    if not engine.is_loaded or engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=SidecarErrorEnvelope(error=SidecarErrorBody(
                code="MODEL_NOT_LOADED",
                message="No model is loaded in sidecar",
            )).model_dump()
        )

    try:
        out = engine.forward(req.input_data)
        if "error" in out:
            raise ValueError(out["error"])

        # Capture basic probes
        activations: Dict[str, Any] = {}
        if "attention" in req.requested_probes or not req.requested_probes:
            attn_res = engine.get_attention_weights(0)
            if "attention" in attn_res:
                activations["attention_layer_0"] = attn_res["attention"]

        session_id = _SESSION_STORE.create(
            model_name=engine.model_class_name,
            output_data=out,
            activations=activations,
            probes=req.requested_probes,
        )

        return SessionSummary(
            session_id=session_id,
            created_at=time.time(),
            model_name=engine.model_class_name,
            output_shape=out.get("output_shape", []),
            output_sample=out.get("output_sample", []),
            probes_captured=req.requested_probes,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=SidecarErrorEnvelope(error=SidecarErrorBody(
                code="INFERENCE_FAILED",
                message=str(e),
            )).model_dump()
        )


@sidecar_router.get("/sessions/{session_id}", response_model=SessionSummary)
async def internal_get_session(session_id: str):
    sess = _SESSION_STORE.get(session_id)
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=SidecarErrorEnvelope(error=SidecarErrorBody(
                code="SESSION_EXPIRED",
                message=f"Inference session '{session_id}' expired or not found",
            )).model_dump()
        )
    return SessionSummary(
        session_id=sess["session_id"],
        created_at=sess["created_at"],
        model_name=sess["model_name"],
        output_shape=sess["output"].get("output_shape", []),
        output_sample=sess["output"].get("output_sample", []),
        probes_captured=sess["probes"],
    )


@sidecar_router.get("/sessions/{session_id}/activations", response_model=ActivationsResponse)
async def internal_get_activations(session_id: str):
    sess = _SESSION_STORE.get(session_id)
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=SidecarErrorEnvelope(error=SidecarErrorBody(
                code="SESSION_EXPIRED",
                message=f"Inference session '{session_id}' expired or not found",
            )).model_dump()
        )
    return ActivationsResponse(
        session_id=session_id,
        activations=sess["activations"],
    )


@sidecar_router.delete("/sessions/{session_id}")
async def internal_delete_session(session_id: str):
    deleted = _SESSION_STORE.delete(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=SidecarErrorEnvelope(error=SidecarErrorBody(
                code="SESSION_EXPIRED",
                message=f"Inference session '{session_id}' not found",
            )).model_dump()
        )
    return {"status": "deleted", "session_id": session_id}
