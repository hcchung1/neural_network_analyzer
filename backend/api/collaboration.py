"""Isolated, bounded collaboration inference; never replaces the legacy engine."""
from typing import Any, Dict, List, Optional
import threading
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model_engine.compatibility import load_checkpoint_strictly, describe_capabilities
from model_engine.transformer_model import TransformerModelEngine
from api.sidecar import LoadModelRequest

router = APIRouter(prefix="/internal/v1/collaboration", tags=["collaboration"])
_compute_lock = threading.Lock()

class AnalysisRequest(BaseModel):
    model: LoadModelRequest
    input_data: Optional[List[List[float]]] = None

@router.post("/analyze")
def analyze(request: AnalysisRequest):
    # Sync handler runs in FastAPI's thread pool; only one expensive job per process.
    # Each request owns its model and hooks, including when other rooms use the legacy API.
    if not _compute_lock.acquire(timeout=120):
        raise HTTPException(503, "推論服務忙碌，請稍後再試。")
    engine = None
    try:
        selection = request.model
        model, spec, _ = load_checkpoint_strictly(
            checkpoint_path=selection.checkpoint_path, manifest_spec=selection.manifest_spec,
            requested_phase=selection.requested_phase, device=selection.device)
        engine = TransformerModelEngine(isolated=True)
        engine._model = model
        engine._model_class_name = spec.name
        engine._device = selection.device
        engine._is_loaded = True
        model.eval()
        layers = [m for m in model.modules() if type(m).__name__ in ("MultiHeadAttention", "MultiheadAttention")]
        result: Dict[str, Any] = {
            "model_name": spec.name, "input_schema": engine.get_input_schema(),
            "capabilities": describe_capabilities(spec.name).to_dict(),
            "layer_count": len(layers), "attention": {}, "output": [],
        }
        if request.input_data is not None:
            output = engine.forward(request.input_data)
            if "error" in output:
                raise ValueError(output["error"])
            values = output.get("output_sample", [])
            while values and isinstance(values[0], list):
                values = values[0]
            result["output"] = values
            for i in range(len(layers)):
                attention = engine.get_attention_weights(i).get("attention")
                if attention is not None:
                    # The view displays a batch/head average, a two-dimensional matrix.
                    matrix = torch.tensor(attention)
                    while matrix.ndim > 2:
                        matrix = matrix.mean(dim=0)
                    result["attention"][str(i)] = matrix.tolist()
        return result
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        if engine is not None:
            engine.clear_attention_cache()
        _compute_lock.release()
