"""模型載入與推論 API."""

from __future__ import annotations

from fastapi import APIRouter
from model_engine.transformer_model import get_model_engine

router = APIRouter()


@router.get("/status")
async def model_status():
    engine = get_model_engine()
    return {
        "loaded": engine.is_loaded,
        "device": engine.device,
        "model_class": engine.model_class_name,
    }


@router.post("/load")
async def load_model(path: str, device: str = "auto"):
    engine = get_model_engine()
    result = engine.load_model(path, device=device)
    return {"success": result}
