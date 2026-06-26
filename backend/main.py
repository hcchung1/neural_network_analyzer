"""FastAPI 入口與 WebSocket handler."""

from __future__ import annotations

import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.model import router as model_router
from api.visualize import router as viz_router
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


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": get_model_engine().is_loaded}


# WebSocket endpoint for real-time feature streaming
@app.websocket("/ws/visualize")
async def websocket_visualize(websocket: WebSocket):
    await websocket.accept()
    engine = get_model_engine()
    registry = _hook_registry

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            action = msg.get("action")

            if action == "clear_cache":
                if registry is None:
                    await websocket.send_json({"error": "HookRegistry not initialized"})
                    continue
                registry.clear_cache()
                await websocket.send_json({"action": "clear_cache", "status": "ok"})

            elif action == "load_model":
                path = msg.get("path")
                device = msg.get("device", "cuda" if engine.is_cuda_available() else "cpu")
                
                # Expand user path for better error reporting
                if path:
                    expanded_path = os.path.expanduser(path)
                    if not os.path.exists(expanded_path):
                        await websocket.send_json({
                            "action": "load_model", 
                            "result": False,
                            "error": f"Model path not found: {path} (expanded: {expanded_path})"
                        })
                        continue
                
                result = engine.load_model(path, device=device)
                await websocket.send_json({"action": "load_model", "result": result})

            elif action == "register_hooks":
                if registry is None:
                    await websocket.send_json({"error": "HookRegistry not initialized"})
                    continue
                # Register forward hooks on current model
                layers = msg.get("layers", ["embedding", "attention", "ffn", "output"])
                registry.register_all(engine.model, layers)
                await websocket.send_json({"action": "register_hooks", "status": "ok", "layers": layers})

            elif action == "get_token_features":
                batch_idx = msg.get("batch_idx", 0)
                token_idx = msg.get("token_idx")
                layer_idx = msg.get("layer_idx")
                if registry is None:
                    await websocket.send_json({"error": "HookRegistry not initialized"})
                    continue
                data = registry.get_features(batch_idx, token_idx, layer_idx)
                await websocket.send_json({"action": "get_token_features", "data": data})

            elif action == "get_token_embedding":
                batch_idx = msg.get("batch_idx", 0)
                token_idx = msg.get("token_idx")
                if registry is None:
                    await websocket.send_json({"error": "HookRegistry not initialized"})
                    continue
                data = registry.get_embedding(batch_idx, token_idx)
                await websocket.send_json({"action": "get_token_embedding", "data": data})

            elif action == "get_attention":
                layer_idx = msg.get("layer_idx", 0)
                head_idx = msg.get("head_idx")
                if registry is None:
                    await websocket.send_json({"error": "HookRegistry not initialized"})
                    continue
                data = registry.get_attention(layer_idx, head_idx)
                await websocket.send_json({"action": "get_attention", "data": data})

            elif action == "get_output":
                if registry is None:
                    await websocket.send_json({"error": "HookRegistry not initialized"})
                    continue
                data = registry.get_output()
                await websocket.send_json({"action": "get_output", "data": data})

            elif action == "run_forward":
                # Expect input tensor serialized as list
                input_data = msg.get("input")
                if input_data is None:
                    await websocket.send_json({"error": "Missing 'input' field"})
                    continue
                result = engine.forward(input_data)
                # After forward, also return cached features from hooks
                features = registry.get_features(0, msg.get("token_idx", 0)) if registry else {}
                await websocket.send_json({"action": "run_forward", "result": result, "features": features})

            else:
                await websocket.send_json({"error": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
