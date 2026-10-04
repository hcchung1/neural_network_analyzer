"""FastAPI Sidecar Entry Point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.sidecar import sidecar_router, cleanup_resources
from api.visualize_chat import router as visualize_chat_router
from api.binary_samples import router as binary_samples_router
from api.csv_reader import router as csv_reader_router
from api.training import router as training_router
from api.compare import router as compare_router
from api.collaboration import router as collaboration_router
from model_engine.resources import shared_model_cache


@asynccontextmanager
async def lifespan(app):
    async def cleanup_loop():
        while True:
            await asyncio.sleep(60)
            await asyncio.to_thread(cleanup_resources)
            await asyncio.to_thread(shared_model_cache.cleanup_expired)
    task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await asyncio.to_thread(shared_model_cache.close)

app = FastAPI(
    lifespan=lifespan,
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
    return {"status": "ok", "sidecar": True}
