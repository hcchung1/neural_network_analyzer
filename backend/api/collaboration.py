"""Shared resident collaboration engines with bounded serialized computation."""
from collections.abc import Mapping
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model_engine.compatibility import load_checkpoint_strictly, describe_analysis_capabilities
from model_engine.transformer_model import TransformerModelEngine
from model_engine.resources import compute_gate, dispose_engine, shared_model_cache
from api.sidecar import LoadModelRequest

router = APIRouter(prefix="/internal/v1/collaboration", tags=["collaboration"])
_SHARED_MODEL_CACHE = shared_model_cache


class AnalysisRequest(BaseModel):
    model: LoadModelRequest
    input_data: Optional[List[List[float]]] = None


def _mutable(value):
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


def _build(selection):
    engine = TransformerModelEngine(isolated=True)
    try:
        model, spec, _ = load_checkpoint_strictly(
            checkpoint_path=selection.checkpoint_path, manifest_spec=selection.manifest_spec,
            requested_phase=selection.requested_phase, device=selection.device)
        engine._model = model
        engine._model_class_name = spec.name
        engine._device = selection.device
        engine._is_loaded = True
        model.eval()
        layers = [m for m in model.modules() if type(m).__name__ in ("MultiHeadAttention", "MultiheadAttention")]
        return engine, {
            "model_name": spec.name, "input_schema": engine.get_input_schema(),
            "capabilities": describe_analysis_capabilities(spec.name),
            "layer_count": len(layers),
        }
    except BaseException:
        dispose_engine(engine)
        raise


@router.post("/analyze")
def analyze(request: AnalysisRequest):
    with compute_gate.slot():
        return _analyze(request)


def _analyze(request: AnalysisRequest):
    try:
        with _SHARED_MODEL_CACHE.lease(request.model, lambda: _build(request.model)) as entry:
            engine = entry.engine
            try:
                result: Dict[str, Any] = _mutable(entry.metadata)
                result.update(output_shape=[], output_kind="raw", attention={}, output=[])
                if request.input_data is not None:
                    output = engine.forward(request.input_data)
                    if "error" in output:
                        raise ValueError(output["error"])
                    values = output.get("output_sample", [])
                    while values and isinstance(values[0], list):
                        values = values[0]
                    result["output"] = values
                    result["output_shape"] = output["output_shape"]
                    for i in range(result["layer_count"]):
                        matrix = engine.get_attention_tensor(i)
                        if matrix is not None:
                            while matrix.ndim > 2:
                                matrix = matrix.mean(dim=0)
                            result["attention"][str(i)] = matrix.tolist()
                return result
            finally:
                engine.clear_attention_cache()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
