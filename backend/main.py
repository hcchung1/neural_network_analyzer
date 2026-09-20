"""FastAPI Sidecar Entry Point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.sidecar import sidecar_router
from api.visualize_chat import router as visualize_chat_router
from api.binary_samples import router as binary_samples_router
from api.csv_reader import router as csv_reader_router
from api.training import router as training_router
from api.compare import router as compare_router
from api.collaboration import router as collaboration_router
from model_engine.transformer_model import get_model_engine

app = FastAPI(
    title="ArchAnalyzer Python Sidecar",
    description="Python backend sidecar for ArchAnalyzer Next.js app",
    version="0.2.0",
)

# CORS for internal communication if needed, though Next.js proxy avoids direct client calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Internal Sidecar APIs only
app.include_router(sidecar_router)
app.include_router(visualize_chat_router, prefix="/internal/v1/assistant")
app.include_router(binary_samples_router, prefix="/internal/v1/legacy/binary_samples")
app.include_router(csv_reader_router, prefix="/internal/v1/legacy/csv_reader")
app.include_router(training_router, prefix="/internal/v1/legacy/training")
app.include_router(compare_router, prefix="/internal/v1")
app.include_router(collaboration_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "sidecar": True, "model_loaded": get_model_engine().is_loaded}
