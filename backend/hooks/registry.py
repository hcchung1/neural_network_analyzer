"""PyTorch forward hooks registry and management."""

from __future__ import annotations

import os
import uuid
import time
from typing import Any, Dict, List, Optional

# PyTorch availability guard
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


class HookRegistry:
    """Manages forward hooks on a PyTorch model to capture intermediate features."""

    def __init__(self, max_cache_size: int = 100) -> None:
        self._handles: List[Any] = []
        self._cache: Dict[str, Any] = {}
        self._model: Any = None
        self._max_cache_size = max_cache_size
        self._session_id: str = str(uuid.uuid4())[:8]

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

        # Recursively register hooks with layer name-based keys
        for name, module in model.named_modules():
            module_type = type(module).__name__.lower()
            for target in target_layers:
                if target.lower() in name.lower() or target.lower() in module_type:
                    handle = module.register_forward_hook(self._make_hook(name))
                    self._handles.append(handle)
                    print(f"[HookRegistry] Hook registered on {name} ({module_type})")
                    break

    def unregister_all(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cache.clear()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Clear cached features to free memory."""
        self._cache.clear()
        print("[HookRegistry] Cache cleared")

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------
    def get_features(self, batch_idx: int, token_idx: Optional[int], layer_idx: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve cached features for a specific token."""
        if not self._cache:
            return {"error": "No cached features. Run a forward pass first."}

        results: Dict[str, Any] = {}
        for key, tensor in self._cache.items():
            if layer_idx is not None and f"layer_{layer_idx}" not in key:
                continue
            if isinstance(tensor, list) and len(tensor) > token_idx:
                results[key] = tensor[token_idx]
            elif hasattr(tensor, "shape"):
                # Assumes tensor shape [batch, seq, ...]
                if token_idx is not None:
                    results[key] = tensor[batch_idx, token_idx].cpu().numpy().tolist()
                else:
                    results[key] = tensor[batch_idx].cpu().numpy().tolist()
        return results

    def get_embedding(self, batch_idx: int, token_idx: int) -> Dict[str, Any]:
        """Get embedding vector for a specific token."""
        for key, tensor in self._cache.items():
            if "embedding" in key.lower() or "embed" in key.lower():
                if hasattr(tensor, "shape"):
                    return {"embedding": tensor[batch_idx, token_idx].cpu().numpy().tolist()}
        return {"error": "Embedding not found in cache"}

    def get_attention(self, layer_idx: int, head_idx: Optional[int] = None) -> Dict[str, Any]:
        """Get attention scores for a specific layer."""
        attn_key = f"attention_layer_{layer_idx}"
        for key, tensor in self._cache.items():
            if attn_key in key.lower() or ("attention" in key.lower() and str(layer_idx) in key):
                data = tensor
                if head_idx is not None and len(data.shape) >= 3:
                    data = data[:, head_idx, :, :]
                return {"attention": data.cpu().numpy().tolist()}
        return {"error": f"Attention for layer {layer_idx} not found"}

    def get_output(self) -> Dict[str, Any]:
        """Get final model output including logits and softmax probabilities."""
        for key, tensor in self._cache.items():
            if "output" in key.lower() or "fc_out" in key.lower() or "final" in key.lower():
                if hasattr(tensor, "cpu"):
                    logits = tensor.cpu().numpy()
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
            self._cache[layer_name] = output.detach() if hasattr(output, "detach") else output
            # Auto-clear cache if too large
            if len(self._cache) > self._max_cache_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
        return hook

    @staticmethod
    def _softmax(x: Any) -> Any:
        import numpy as np
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
