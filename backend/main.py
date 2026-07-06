"""FastAPI 入口與 WebSocket handler."""

from __future__ import annotations

import json
import os
import asyncio
import traceback
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.model import router as model_router
from api.visualize import router as viz_router
from api.training import router as training_router
from model_engine.transformer_model import get_model_engine
from hooks.registry import HookRegistry


# Global hook registry shared across WebSocket connections
_hook_registry: HookRegistry | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _hook_registry
    _hook_registry = HookRegistry()
    print("[lifespan] HookRegistry initialized")
    yield
    print("[lifespan] Shutting down...")


app = FastAPI(
    title="ArchAnalyzer Backend",
    description="Transformer single-token visualization backend",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routers
app.include_router(model_router, prefix="/api/model", tags=["model"])
app.include_router(viz_router, prefix="/api/visualize", tags=["visualize"])
app.include_router(training_router, prefix="/api/training", tags=["training"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": get_model_engine().is_loaded}


# WebSocket endpoint for real-time feature streaming
@app.websocket("/ws/visualize")
async def websocket_visualize(websocket: WebSocket):
    await websocket.accept()
    engine = get_model_engine()
    registry = _hook_registry

    async def _send_error(action: str, message: str, extra: dict | None = None) -> None:
        payload = {"action": action, "error": message}
        if extra:
            payload.update(extra)
        await websocket.send_json(payload)

    async def _handle_load_model(msg: dict) -> None:
        path = msg.get("path")
        device = msg.get("device", "cuda" if engine.is_cuda_available() else "cpu")
        
        if path:
            expanded_path = os.path.expanduser(path)
            if not os.path.exists(expanded_path):
                await _send_error(
                    "load_model",
                    f"Model path not found: {path} (expanded: {expanded_path})"
                )
                return
        
        try:
            result = engine.load_model(path, device=device)
            # Get model metadata for frontend
            model_meta = {}
            if engine.model:
                model = engine.model
                model_meta['seq_len'] = getattr(model, 'seq_len', 49)
                model_meta['input_features'] = getattr(model, 'input_features', None)
                model_meta['n_layers'] = getattr(model, 'n_layers', None)
                model_meta['d_model'] = getattr(model, 'd_model', None)
            await websocket.send_json({"action": "load_model", "result": result, "model_meta": model_meta})
        except Exception as e:
            traceback.print_exc()
            await _send_error("load_model", f"Failed to load model: {str(e)}")

    async def _handle_register_hooks(msg: dict) -> None:
        if registry is None:
            await _send_error("register_hooks", "HookRegistry not initialized")
            return
        layers = msg.get("layers", ["embedding", "attention", "ffn", "output"])
        try:
            registry.register_all(engine.model, layers)
            await websocket.send_json({
                "action": "register_hooks",
                "status": "ok",
                "layers": layers
            })
        except Exception as e:
            traceback.print_exc()
            await _send_error("register_hooks", f"Failed to register hooks: {str(e)}")

    async def _handle_run_forward(msg: dict) -> None:
        input_data = msg.get("input")
        
        # Auto-generate dummy input if not provided, based on model dimensions
        if input_data is None:
            if not engine.model:
                await _send_error("run_forward", "Model not loaded")
                return
            try:
                import numpy as np
                import torch
                
                # Determine expected input dimensions from model
                model = engine.model
                seq_len = getattr(model, 'seq_len', 49)  # Default to 49 (common for Mahjong)
                input_features = getattr(model, 'input_features', None)
                
                if input_features is None:
                    # Try to infer from input_projection or proj_init/proj_act
                    if hasattr(model, 'input_projection') and model.input_projection is not None:
                        input_features = model.input_projection.in_features
                    elif hasattr(model, 'proj_init') and model.proj_init is not None:
                        # Heterogeneous tokens mode
                        f_init = model.proj_init.in_features
                        input_features = f_init  # Use F_INIT as base
                    elif hasattr(model, 'proj_act') and model.proj_act is not None:
                        input_features = model.proj_act.in_features
                    else:
                        input_features = 850  # Sensible default for Mahjong Transformer
                
                print(f"[run_forward] Auto-generating dummy input: batch=1, seq_len={seq_len}, features={input_features}")
                input_data = np.random.randn(1, seq_len, input_features).tolist()
            except Exception as e:
                await _send_error("run_forward", f"Failed to auto-generate input: {str(e)}")
                return
        
        if not registry:
            await _send_error("run_forward", "HookRegistry not initialized")
            return

        try:
            result = engine.forward(input_data)
            # After forward, also return cached features from hooks
            token_idx = msg.get("token_idx", 0)
            features = registry.get_features(0, token_idx)
            await websocket.send_json({
                "action": "run_forward",
                "result": result,
                "features": features
            })
        except Exception as e:
            traceback.print_exc()
            await _send_error("run_forward", f"Forward pass failed: {str(e)}")

    try:
        while True:
            raw = await websocket.receive_text()
            print(f"[DEBUG] Raw message received: {raw[:200]}...")
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error("unknown", "Invalid JSON")
                continue

            action = msg.get("action")
            print(f"[DEBUG] Processing action: {action}")

            if action == "clear_cache":
                if registry is None:
                    await _send_error("clear_cache", "HookRegistry not initialized")
                    continue
                registry.clear_cache()
                await websocket.send_json({"action": "clear_cache", "status": "ok"})

            elif action == "load_model":
                await _handle_load_model(msg)

            elif action == "register_hooks":
                await _handle_register_hooks(msg)

            elif action == "get_token_features":
                if registry is None:
                    await _send_error("get_token_features", "HookRegistry not initialized")
                    continue
                data = registry.get_features(
                    msg.get("batch_idx", 0),
                    msg.get("token_idx"),
                    msg.get("layer_idx")
                )
                await websocket.send_json({"action": "get_token_features", "data": data})

            elif action == "get_token_embedding":
                if registry is None:
                    await _send_error("get_token_embedding", "HookRegistry not initialized")
                    continue
                try:
                    print(f"[DEBUG] Processing get_token_embedding request...")
                    data = registry.get_embedding(
                        msg.get("batch_idx", 0),
                        msg.get("token_idx")
                    )
                    print(f"[DEBUG] get_token_embedding result: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                    await websocket.send_json({"action": "get_token_embedding", "data": data})
                    print(f"[DEBUG] get_token_embedding response sent")
                except Exception as e:
                    print(f"[ERROR] get_token_embedding failed: {e}")
                    traceback.print_exc()
                    await _send_error("get_token_embedding", str(e))

            elif action == "get_attention":
                if registry is None:
                    await _send_error("get_attention", "HookRegistry not initialized")
                    continue
                try:
                    print(f"[DEBUG] Processing get_attention request...")
                    data = registry.get_attention(
                        msg.get("layer_idx", 0),
                        msg.get("head_idx")
                    )
                    print(f"[DEBUG] get_attention result: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                    await websocket.send_json({"action": "get_attention", "data": data})
                    print(f"[DEBUG] get_attention response sent")
                except Exception as e:
                    print(f"[ERROR] get_attention failed: {e}")
                    traceback.print_exc()
                    await _send_error("get_attention", str(e))

            elif action == "get_output":
                if registry is None:
                    await _send_error("get_output", "HookRegistry not initialized")
                    continue
                print(f"[DEBUG] Processing get_output request...")
                data = registry.get_output()
                print(f"[DEBUG] get_output result: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                await websocket.send_json({"action": "get_output", "data": data})
                print(f"[DEBUG] get_output response sent")

            elif action == "run_forward":
                await _handle_run_forward(msg)

            else:
                await _send_error("unknown", f"Unknown action: {action}")

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Unexpected error: {e}")
        traceback.print_exc()
        try:
            await _send_error("server_error", "Internal server error")
        except Exception:
            pass
    finally:
        print("[WebSocket] Connection closed")
