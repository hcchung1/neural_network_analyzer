"""PyTorch forward hooks registry and management."""

from __future__ import annotations

import os
import uuid
import time
import weakref
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# PyTorch availability guard
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


class HookRegistry:
    """Manages forward hooks on a PyTorch model to capture intermediate features.
    
    Features:
      - LRU-style cache with max size limit
      - Memory usage tracking and automatic cleanup
      - Thread-safe operations
      - Support for named hook targets and selective registration
    """

    def __init__(self, max_cache_size: int = 100) -> None:
        self._handles: List[Any] = []
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._model: Any = None
        self._max_cache_size = max_cache_size
        self._session_id: str = str(uuid.uuid4())[:8]
        self._lock = threading.RLock()
        self._memory_usage_mb: float = 0.0
        self._max_memory_mb: float = 1024.0  # 1 GB max

    # ------------------------------------------------------------------
    # Register / unregister
    # ------------------------------------------------------------------
    def register_all(self, model: Any, layers: Optional[List[str]] = None) -> None:
        """Register forward hooks on specified layers."""
        if not _TORCH_AVAILABLE:
            print("[HookRegistry] PyTorch unavailable, skipping hook registration.")
            return

        self.unregister_all()
        self._model = model
        self._cache.clear()

        target_layers = layers or ["embedding", "attention", "ffn", "output"]
        
        # Map of target keywords to possible module names/patterns
        layer_patterns = {
            "embedding": ["embed", "proj_init", "proj_act", "input_projection", "token_emb"],
            "attention": ["attention", "attn", "multihead"],
            "ffn": ["ffn", "feedforward", "feed_forward", "fc"],
            "output": ["output", "fc_out", "final", "classifier", "logits"]
        }

        # Recursively register hooks with layer name-based keys
        for name, module in model.named_modules():
            module_type = type(module).__name__.lower()
            for target in target_layers:
                # Check if this module matches any pattern for this target
                patterns = layer_patterns.get(target.lower(), [target.lower()])
                matched = False
                
                # Check module name against patterns
                for pattern in patterns:
                    if pattern.lower() in name.lower():
                        matched = True
                        break
                
                # Also check module type
                if not matched:
                    for pattern in patterns:
                        if pattern.lower() in module_type:
                            matched = True
                            break
                
                if matched:
                    handle = module.register_forward_hook(self._make_hook(name))
                    self._handles.append(handle)
                    print(f"[HookRegistry] Hook registered on {name} ({module_type})")
                    break

    def unregister_all(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cache.clear()
        self._memory_usage_mb = 0.0

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Clear cached features to free memory."""
        with self._lock:
            self._cache.clear()
            self._memory_usage_mb = 0.0
        print("[HookRegistry] Cache cleared")

    def _check_memory_limit(self) -> None:
        """Check if cache exceeds memory limit and evict oldest entries."""
        if self._memory_usage_mb > self._max_memory_mb:
            with self._lock:
                while self._memory_usage_mb > self._max_memory_mb * 0.8 and self._cache:
                    self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Evict the oldest cached entry (LRU)."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            removed = self._cache.pop(oldest_key, None)
            if removed is not None and hasattr(removed, 'element_size'):
                # Rough estimation of freed memory
                self._memory_usage_mb -= self._estimate_tensor_size(removed)
            print(f"[HookRegistry] Evicted oldest cache entry: {oldest_key}")

    def _estimate_tensor_size(self, tensor: Any) -> float:
        """Estimate tensor size in MB."""
        if hasattr(tensor, 'element_size') and hasattr(tensor, 'numel'):
            return (tensor.element_size() * tensor.numel()) / (1024 * 1024)
        return 0.0

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------
    def get_features(self, batch_idx: int, token_idx: Optional[int], layer_idx: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve cached features for a specific token."""
        if not self._cache:
            return {"error": "No cached features. Run a forward pass first."}

        with self._lock:
            results: Dict[str, Any] = {}
            for key, tensor in self._cache.items():
                # Filter by layer index if specified
                if layer_idx is not None:
                    # Match patterns like "transformer_layers.0.attention" or "layer_0"
                    # Check if the key contains the layer index number
                    import re
                    # Pattern to match layer index in various formats: .0., layer_0, layers.0, etc.
                    layer_patterns = [
                        f"\\.{layer_idx}\\.",  # matches .0.
                        f"layer_{layer_idx}",   # matches layer_0
                        f"layers\\.{layer_idx}",  # matches layers.0
                        f"layer{layer_idx}",    # matches layer0
                    ]
                    if not any(re.search(pattern, key) for pattern in layer_patterns):
                        continue
                # Handle different data types
                if isinstance(tensor, list) and len(tensor) > token_idx:
                    results[key] = tensor[token_idx]
                elif hasattr(tensor, "shape"):
                    # Handle different tensor shapes
                    if len(tensor.shape) == 3:
                        # 3D tensor: [batch, seq, ...]
                        if token_idx is not None:
                            results[key] = tensor[batch_idx, token_idx, :].flatten().cpu().numpy().tolist()
                        else:
                            results[key] = tensor[batch_idx].cpu().numpy().tolist()
                    elif len(tensor.shape) == 2:
                        # 2D tensor: [batch, features]
                        results[key] = tensor[batch_idx].cpu().numpy().tolist()
                    elif len(tensor.shape) == 1:
                        # 1D tensor: [features]
                        results[key] = tensor.cpu().numpy().tolist()
                    else:
                        results[key] = tensor.cpu().numpy().tolist()
            return results

    def get_embedding(self, batch_idx: int, token_idx: int) -> Dict[str, Any]:
        """Get embedding vector for a specific token."""
        with self._lock:
            for key, tensor in self._cache.items():
                # Match various embedding-related keys
                if any(k in key.lower() for k in ["embedding", "embed", "proj_init", "proj_act", "input_projection"]):
                    print(f"[DEBUG] Found embedding key: {key}, type: {type(tensor)}, shape: {getattr(tensor, 'shape', 'N/A')}")
                    # Handle PyTorch tensors
                    if hasattr(tensor, "shape"):
                        # Handle different tensor shapes
                        if len(tensor.shape) == 3:
                            # 3D tensor: [batch, seq, features]
                            result = tensor[batch_idx, token_idx, :].flatten().cpu().numpy().tolist()
                            print(f"[DEBUG] 3D tensor result type: {type(result)}, len: {len(result) if isinstance(result, list) else 'N/A'}")
                            return {"embedding": result}
                        elif len(tensor.shape) == 2:
                            # 2D tensor: [batch, features] - return the whole vector for this batch
                            result = tensor[batch_idx, :].flatten().cpu().numpy().tolist()
                            print(f"[DEBUG] 2D tensor result type: {type(result)}, len: {len(result) if isinstance(result, list) else 'N/A'}")
                            return {"embedding": result}
                        elif len(tensor.shape) <= 1:
                            # 1D tensor or scalar: [features] - return as is
                            result = tensor.cpu().numpy().tolist()
                            print(f"[DEBUG] 1D/0D tensor result type: {type(result)}")
                            return {"embedding": result}
                    # Handle lists (already converted tensors)
                    elif isinstance(tensor, list):
                        print(f"[DEBUG] List result len: {len(tensor)}")
                        return {"embedding": tensor}
                    else:
                        print(f"[DEBUG] Unknown tensor type: {type(tensor)}")
            return {"error": "Embedding not found in cache"}

    def get_attention(self, layer_idx: int, head_idx: Optional[int] = None) -> Dict[str, Any]:
        """Get attention scores for a specific layer."""
        with self._lock:
            # Priority 1: Check model's last_attention_weights attribute (for custom models)
            if self._model is not None:
                model = self._model
                # Check engine's _captured_attention first (from registered hooks)
                from model_engine.transformer_model import get_model_engine
                engine = get_model_engine()
                if engine._captured_attention and any(w is not None for w in engine._captured_attention):
                    if 0 <= layer_idx < len(engine._captured_attention):
                        weights = engine._captured_attention[layer_idx]
                        if weights is not None:
                            if hasattr(weights, 'cpu'):
                                weights_data = weights.cpu().detach().numpy().tolist()
                            else:
                                weights_data = weights
                            if head_idx is not None and len(weights_data) > head_idx:
                                weights_data = weights_data[head_idx]
                            return {"attention": weights_data}
                        else:
                            return {"error": f"Attention for layer {layer_idx} is None (hook may not have captured)"}
                    else:
                        actual_layers = len(engine._captured_attention)
                        return {"error": f"Layer index {layer_idx} out of range. Engine has {actual_layers} layers."}
                # Check engine's get_attention_weights method
                elif hasattr(engine, 'get_attention_weights'):
                    return engine.get_attention_weights(layer_idx, head_idx)
                # Check model's _last_attention_weights (set by forward)
                elif hasattr(model, '_last_attention_weights') and model._last_attention_weights is not None:
                    if 0 <= layer_idx < len(model._last_attention_weights):
                        weights = model._last_attention_weights[layer_idx]
                        if hasattr(weights, 'cpu'):
                            weights_data = weights.cpu().detach().numpy().tolist()
                        else:
                            weights_data = weights
                        if head_idx is not None and len(weights_data) > head_idx:
                            weights_data = weights_data[head_idx]
                        return {"attention": weights_data}
                    else:
                        return {"error": f"Layer index {layer_idx} out of range. Model has {len(model._last_attention_weights) if model._last_attention_weights else 0} layers."}
                # Check model's last_attention_weights (for MahjongTransformer)
                elif hasattr(model, 'last_attention_weights') and model.last_attention_weights is not None:
                    if 0 <= layer_idx < len(model.last_attention_weights):
                        weights = model.last_attention_weights[layer_idx]
                        if hasattr(weights, 'cpu'):
                            weights_data = weights.cpu().detach().numpy().tolist()
                        else:
                            weights_data = weights
                        if head_idx is not None and len(weights_data) > head_idx:
                            weights_data = weights_data[head_idx]
                        return {"attention": weights_data}
                    else:
                        return {"error": f"Layer index {layer_idx} out of range. Model has {len(model.last_attention_weights) if model.last_attention_weights else 0} layers."}
                else:
                    return {"error": "Model does not have attention weights cached. Please restart the backend server."}
            
            # Priority 2: Check cache for standard models
            attn_key = f"attention_layer_{layer_idx}"
            for key, tensor in self._cache.items():
                if attn_key in key.lower() or ("attention" in key.lower() and str(layer_idx) in key):
                    # Handle PyTorch tensors
                    if hasattr(tensor, "shape"):
                        data = tensor
                        if head_idx is not None and len(data.shape) >= 3:
                            data = data[:, head_idx, :, :]
                        return {"attention": data.cpu().numpy().tolist()}
                    # Handle lists
                    elif isinstance(tensor, list):
                        return {"attention": tensor}
            return {"error": f"Attention for layer {layer_idx} not found"}

    # get_embedding is already defined above (lines 178-209)

    def get_output(self) -> Dict[str, Any]:
        """Get final model output including logits and softmax probabilities."""
        with self._lock:
            for key, tensor in self._cache.items():
                # Match various output-related keys (including fc, output, final)
                if any(k in key.lower() for k in ["output", "fc_out", "final", "fc"]):
                    if hasattr(tensor, "cpu"):
                        logits = tensor.cpu().numpy()
                        probs = self._softmax(logits)
                        return {
                            "logits": logits.tolist(),
                            "probabilities": probs.tolist()
                        }
            # If no output found, return the last cached tensor as fallback
            if self._cache:
                last_key = list(self._cache.keys())[-1]
                last_tensor = self._cache[last_key]
                if hasattr(last_tensor, "cpu"):
                    logits = last_tensor.cpu().numpy()
                    probs = self._softmax(logits)
                    return {
                        "logits": logits.tolist(),
                        "probabilities": probs.tolist()
                    }
            return {"error": "Output not found in cache"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_hook(self, layer_name: str):
        def hook(module, input, output):
            if hasattr(output, "detach"):
                cached = output.detach()
                self._cache[layer_name] = cached
                self._memory_usage_mb += self._estimate_tensor_size(cached)
                self._check_memory_limit()
            else:
                self._cache[layer_name] = output
            # Auto-clear cache if too large
            if len(self._cache) > self._max_cache_size:
                self._evict_oldest()
        return hook

    @staticmethod
    def _softmax(x: Any) -> Any:
        import numpy as np
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unregister_all()
        return False

    def __del__(self):
        # Ensure hooks are removed on garbage collection
        try:
            self.unregister_all()
        except Exception:
            pass
