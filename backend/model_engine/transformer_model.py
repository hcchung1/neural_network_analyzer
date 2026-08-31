"""Transformer model definition and loading logic."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None

# PyTorch guard for environments without torch
_EXECUTION_ENV = os.environ.get("EXECUTION_ENV", "auto")

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

# Try to import the real MahjongTransformer
_MAHJONG_TRANSFORMER = None

def _try_import_mahjong_transformer():
    """Try to import MahjongTransformer from internal backend models."""
    global _MAHJONG_TRANSFORMER
    if _MAHJONG_TRANSFORMER is not None:
        return _MAHJONG_TRANSFORMER

    try:
        from models.sequence import MahjongTransformer
        _MAHJONG_TRANSFORMER = MahjongTransformer
        print("[TransformerModelEngine] Successfully imported MahjongTransformer from models.sequence")
        return _MAHJONG_TRANSFORMER
    except Exception as e:
        print(f"[TransformerModelEngine] Failed to import MahjongTransformer: {e}")

    print("[TransformerModelEngine] Could not import MahjongTransformer, using dummy model")
    return None


class DummyModule:
    """Placeholder when torch is unavailable."""
    pass


class DummyModel:
    """Placeholder model when torch is unavailable."""
    pass

class TransformerModelEngine:
    """Singleton engine to load and run the MahjongTransformer model."""

    _instance: "TransformerModelEngine | None" = None

    def __new__(cls) -> "TransformerModelEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._model: Any = None
        self._device: str = "cpu"
        self._is_loaded: bool = False
        self._model_class_name: str = "Unknown"
        self._initialized = True
        # Store attention weights captured from any model type
        self._captured_attention: List[Any] = []
        self._attention_hooks: List[Any] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def model(self) -> Any:
        return self._model

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return _TORCH_AVAILABLE and self._is_loaded

    @property
    def model_class_name(self) -> str:
        return self._model_class_name

    def get_input_schema(self) -> Dict[str, Any] | None:
        """Return the dense feature tensor contract exposed by the loaded model."""
        if self._model is None:
            return None

        seq_len = getattr(self._model, "seq_len", None)
        feature_dim = getattr(self._model, "_expected_feature_dim", None)
        if feature_dim is None:
            feature_dim = getattr(self._model, "input_features", None)
        if not isinstance(seq_len, int) or not isinstance(feature_dim, int):
            return None
        if seq_len <= 0 or feature_dim <= 0:
            return None

        return {
            "dtype": "float32",
            "rank": 3,
            "batch_size": 1,
            "seq_len": seq_len,
            "feature_dim": feature_dim,
            "shape": [1, seq_len, feature_dim],
            "accepted_json_shapes": [
                [seq_len, feature_dim],
                [1, seq_len, feature_dim],
            ],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def is_cuda_available() -> bool:
        if not _TORCH_AVAILABLE:
            return False
        import torch
        return torch.cuda.is_available()

    @staticmethod
    def _configure_input_projection_from_state_dict(model: Any, state_dict: Any) -> None:
        """Align the model instance's input projection with checkpoint weight shapes."""
        if not isinstance(state_dict, dict):
            return

        proj_init_weight = state_dict.get("proj_init.weight")
        proj_act_weight = state_dict.get("proj_act.weight")
        input_projection_weight = state_dict.get("input_projection.weight")

        if proj_init_weight is not None and proj_act_weight is not None:
            import torch.nn as nn

            d_model = int(proj_init_weight.shape[0])
            init_dim = int(proj_init_weight.shape[1])
            act_dim = int(proj_act_weight.shape[1])
            feature_dim = max(init_dim, act_dim)
            cols = int(getattr(model, "cols", 34))
            token_mode = "multiply" if cols > 0 and act_dim % cols == 0 else "add"

            if hasattr(model, "input_projection"):
                delattr(model, "input_projection")
            model.proj_init = nn.Linear(init_dim, d_model, bias=False)
            model.proj_act = nn.Linear(act_dim, d_model, bias=False)
            model.F_INIT = init_dim
            if token_mode == "multiply":
                model.F_ACT = act_dim
                model.F_PAD = feature_dim
            else:
                model.F_ACT_ADD = act_dim
                model.F_PAD_ADD = feature_dim
            model.token_mode = token_mode
            model.use_heter_tokens = True
            model._act_feature_dim = act_dim
            model._expected_feature_dim = feature_dim
            model.input_features = feature_dim
            model.input_shape = (int(model.seq_len), feature_dim)
            print(
                "[TransformerModelEngine] Configured checkpoint input projection: "
                f"mode={token_mode}, init_dim={init_dim}, act_dim={act_dim}, "
                f"feature_dim={feature_dim}"
            )
            return

        if input_projection_weight is not None:
            import torch.nn as nn

            d_model = int(input_projection_weight.shape[0])
            feature_dim = int(input_projection_weight.shape[1])
            for attribute in ("proj_init", "proj_act"):
                if hasattr(model, attribute):
                    delattr(model, attribute)
            model.input_projection = nn.Linear(feature_dim, d_model, bias=False)
            model.token_mode = None
            model.use_heter_tokens = False
            model.input_features = feature_dim
            model.input_shape = (int(model.seq_len), feature_dim)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self, path: str | None, *, device: str = "auto") -> bool:
        if not _TORCH_AVAILABLE:
            print("[TransformerModelEngine] PyTorch not available, operating in dummy mode.")
            self._is_loaded = False
            return False

        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = device

        # If no path provided, instantiate a dummy model for development
        if path is not None:
            path = os.path.expanduser(path)
        if path is None or not os.path.exists(path):
            print("[TransformerModelEngine] No model file provided; creating dummy transformer.")
            # Use default hyperparameters
            self._model = self._create_dummy_model()
            self._is_loaded = True
            self._model_class_name = "DummyMahjongTransformer"
            return True

        try:
            from model_engine.compatibility import load_checkpoint_strictly
            model, spec, loaded_phase = load_checkpoint_strictly(path, device=device)
            self._model = model
            self._model_class_name = spec.name
            self._is_loaded = True
            print(f"[TransformerModelEngine] Model '{spec.name}' strictly loaded to {device} (phase: {loaded_phase})")
            return True
        except Exception as e:
            print(f"[TransformerModelEngine] Strict loading via compatibility layer failed: {e}")
            # Fallback to direct load or dummy model if explicit path failed
            self._is_loaded = False
            return False

    # ------------------------------------------------------------------
    # Attention hook management
    # ------------------------------------------------------------------
    def _register_attention_hooks(self) -> None:
        """Register hooks on attention modules to capture attention weights.
        
        For PyTorch's MultiheadAttention, uses register_forward_hook.
        For custom MultiHeadAttention (e.g., MahjongTransformer), monkey-patches
        scaled_dot_product_attention to capture weights.
        """
        if self._model is None:
            return
        
        # Clear any existing hooks
        self._remove_attention_hooks()
        self._captured_attention = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # For PyTorch MultiheadAttention, output is (attn_output, attn_weights)
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        while len(self._captured_attention) <= layer_idx:
                            self._captured_attention.append(None)
                        self._captured_attention[layer_idx] = attn_weights.detach().cpu() if hasattr(attn_weights, 'detach') else attn_weights
            return hook
        
        # Find attention modules and register hooks / patches
        layer_idx = 0
        for name, module in self._model.named_modules():
            module_type = type(module).__name__
            # Match PyTorch MultiheadAttention (has forward hook with tuple output)
            if module_type == 'MultiheadAttention':
                handle = module.register_forward_hook(make_hook(layer_idx))
                self._attention_hooks.append(handle)
                print(f"[TransformerModelEngine] Registered forward hook on {name} ({module_type})")
                layer_idx += 1
            # Match custom MultiHeadAttention (e.g., MahjongTransformer)
            # forward() returns (output, attention_weights) tuple
            elif module_type == 'MultiHeadAttention':
                handle = module.register_forward_hook(make_hook(layer_idx))
                self._attention_hooks.append(handle)
                print(f"[TransformerModelEngine] Registered forward hook on {name} ({module_type})")
                layer_idx += 1
            # Also match modules with 'attention' in name that are not already handled
            elif 'attention' in name.lower() and module_type not in ['MultiheadAttention', 'MultiHeadAttention']:
                # Generic fallback: try forward hook
                handle = module.register_forward_hook(make_hook(layer_idx))
                self._attention_hooks.append(handle)
                print(f"[TransformerModelEngine] Registered fallback hook on {name} ({module_type})")
                layer_idx += 1
        
        if layer_idx == 0:
            print("[TransformerModelEngine] Warning: No attention modules found for hook registration")

    def _patch_multihead_attention(self, module: Any, layer_idx: int) -> None:
        """Monkey-patch a custom MultiHeadAttention module to capture attention weights."""
        import torch
        import torch.nn.functional as F
        
        # Store reference to the original method
        if not hasattr(module, '_original_scaled_dot_product_attention'):
            module._original_scaled_dot_product_attention = module.scaled_dot_product_attention
        
        # Create a closure that captures layer_idx and self
        _self = self
        _layer_idx = layer_idx
        
        def patched_sdpa(Q, K, V, mask=None):
            # Call original method
            output, attn_weights = module._original_scaled_dot_product_attention(Q, K, V, mask)
            
            # Capture attention weights if available
            if attn_weights is not None:
                while len(_self._captured_attention) <= _layer_idx:
                    _self._captured_attention.append(None)
                _self._captured_attention[_layer_idx] = attn_weights.detach().cpu() if hasattr(attn_weights, 'detach') else attn_weights
            
            return output, attn_weights
        
        # Apply the patch
        module.scaled_dot_product_attention = patched_sdpa

    # ------------------------------------------------------------------
    # Dummy model creation for dev / when real weights missing
    # ------------------------------------------------------------------
    def _create_dummy_model(self) -> Any:
        if not _TORCH_AVAILABLE:
            return DummyModel()

        import torch
        import torch.nn as nn

        class _DummyTransformer(nn.Module):
            def __init__(self, vocab_size: int = 128, d_model: int = 256, n_layers: int = 4, n_heads: int = 8):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
                self.d_model = d_model
                self.n_heads = n_heads
                self.n_layers = n_layers
                
                # Manual attention layers to capture weights
                self.transformer_layers = nn.ModuleList()
                for _ in range(n_layers):
                    self.transformer_layers.append(nn.ModuleDict({
                        'norm1': nn.LayerNorm(d_model),
                        'attention': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                        'dropout': nn.Dropout(0.0),
                        'norm2': nn.LayerNorm(d_model),
                        'ffn': nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.ReLU(),
                            nn.Linear(d_model * 4, d_model)
                        )
                    }))
                
                # Store attention weights during forward pass
                self.last_attention_weights = None
                self.fc_out = nn.Linear(d_model, vocab_size)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                seq_len = x.size(1)
                x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
                
                # Manual transformer forward to capture attention weights
                self.last_attention_weights = []
                for layer in self.transformer_layers:
                    # LN + Attention
                    norm_x = layer['norm1'](x)
                    attn_out, attn_weights = layer['attention'](norm_x, norm_x, norm_x)
                    self.last_attention_weights.append(attn_weights)
                    x = x + layer['dropout'](attn_out)
                    
                    # LN + FFN
                    x = x + layer['ffn'](layer['norm2'](x))
                
                return self.fc_out(x)

        return _DummyTransformer()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, input_data: List[List[int]] | np.ndarray | Any) -> Dict[str, Any]:
        if not _TORCH_AVAILABLE or not self._is_loaded or self._model is None:
            return {"error": "Model not loaded or torch unavailable"}

        import torch

        try:
            if isinstance(input_data, list):
                # MahjongTransformer expects float features, not integer token indices
                tensor = torch.tensor(input_data, dtype=torch.float32, device=self._device)
            elif isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data).float().to(self._device)
            else:
                # Assume it's already a tensor
                tensor = input_data.to(self._device)
                if not torch.is_floating_point(tensor):
                    tensor = tensor.float()

            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 3:
                return {
                    "error": (
                        "Input feature tensor must have rank 3 [batch, seq_len, feature_dim] "
                        f"or rank 2 [seq_len, feature_dim]; received shape {list(tensor.shape)}"
                    )
                }

            input_schema = self.get_input_schema()
            if input_schema is not None:
                expected_seq_len = input_schema["seq_len"]
                expected_feature_dim = input_schema["feature_dim"]
                if tensor.shape[0] != 1:
                    return {
                        "error": f"Visualization accepts batch size 1; received {tensor.shape[0]}"
                    }
                if tensor.shape[1] != expected_seq_len or tensor.shape[2] != expected_feature_dim:
                    return {
                        "error": (
                            f"Input feature shape mismatch: expected [1, {expected_seq_len}, "
                            f"{expected_feature_dim}], received {list(tensor.shape)}"
                        )
                    }

            # Register attention hooks before forward pass
            self._register_attention_hooks()

            with torch.no_grad():
                output = self._model(tensor)
            
            # Collect attention weights from attention modules (for real MahjongTransformer)
            self._captured_attention = []
            for name, module in self._model.named_modules():
                module_type = type(module).__name__
                if module_type == 'MultiHeadAttention':
                    if hasattr(module, 'last_attention_weights') and module.last_attention_weights is not None:
                        self._captured_attention.append(module.last_attention_weights.detach().cpu())
                        print(f"[TransformerModelEngine] Collected attention weights from {name}: shape={module.last_attention_weights.shape}")
                    else:
                        self._captured_attention.append(None)
                elif module_type == 'MultiheadAttention':
                    # PyTorch native MHA - weights captured by hooks
                    pass
            
            # Capture attention weights from model attributes (for dummy model compatibility)
            if hasattr(self._model, 'last_attention_weights') and self._model.last_attention_weights is not None:
                self._model._last_attention_weights = [
                    w.detach().cpu() if hasattr(w, 'detach') else w 
                    for w in self._model.last_attention_weights
                ]
                print(f"[TransformerModelEngine] Captured attention weights from model attribute: {len(self._model._last_attention_weights)} layers")

            return {
                "output_shape": list(output.shape),
                "output_sample": output[0, :5].cpu().numpy().tolist(),
            }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Public API for attention retrieval
    # ------------------------------------------------------------------
    def get_attention_weights(self, layer_idx: int, head_idx: Optional[int] = None) -> Dict[str, Any]:
        """Get attention weights for a specific layer.
        
        Returns:
            Dict with 'attention' key containing the weights, or 'error' key.
        """
        # Check engine's captured attention first (from hooks)
        if self._captured_attention and any(w is not None for w in self._captured_attention):
            if 0 <= layer_idx < len(self._captured_attention):
                weights = self._captured_attention[layer_idx]
                if weights is not None:
                    if hasattr(weights, 'cpu'):
                        weights_data = weights.cpu().detach().numpy().tolist()
                    else:
                        weights_data = weights
                    if head_idx is not None and isinstance(weights_data, list) and len(weights_data) > head_idx:
                        weights_data = weights_data[head_idx]
                    return {"attention": weights_data}
                else:
                    return {"error": f"Attention for layer {layer_idx} is None (hook may not have captured)"}
            else:
                actual_layers = len(self._captured_attention)
                return {"error": f"Layer index {layer_idx} out of range. Engine has {actual_layers} layers."}
        
        # Check model's _last_attention_weights (set by forward())
        if self._model is not None:
            if hasattr(self._model, '_last_attention_weights') and self._model._last_attention_weights is not None:
                if 0 <= layer_idx < len(self._model._last_attention_weights):
                    weights = self._model._last_attention_weights[layer_idx]
                    if hasattr(weights, 'cpu'):
                        weights_data = weights.cpu().detach().numpy().tolist()
                    else:
                        weights_data = weights
                    if head_idx is not None and isinstance(weights_data, list) and len(weights_data) > head_idx:
                        weights_data = weights_data[head_idx]
                    return {"attention": weights_data}
                else:
                    actual_layers = len(self._model._last_attention_weights) if self._model._last_attention_weights else 0
                    return {"error": f"Layer index {layer_idx} out of range. Model has {actual_layers} layers."}
            
            # Check model's last_attention_weights (for dummy model)
            if hasattr(self._model, 'last_attention_weights') and self._model.last_attention_weights is not None:
                if 0 <= layer_idx < len(self._model.last_attention_weights):
                    weights = self._model.last_attention_weights[layer_idx]
                    if hasattr(weights, 'cpu'):
                        weights_data = weights.cpu().detach().numpy().tolist()
                    else:
                        weights_data = weights
                    if head_idx is not None and isinstance(weights_data, list) and len(weights_data) > head_idx:
                        weights_data = weights_data[head_idx]
                    return {"attention": weights_data}
                else:
                    actual_layers = len(self._model.last_attention_weights) if self._model.last_attention_weights else 0
                    return {"error": f"Layer index {layer_idx} out of range. Model has {actual_layers} layers."}
        
        return {"error": "No attention weights available. Please run a forward pass first."}

    def _remove_attention_hooks(self) -> None:
        """Remove all registered attention hooks."""
        for handle in self._attention_hooks:
            handle.remove()
        self._attention_hooks.clear()

    def clear_attention_cache(self) -> None:
        """Clear captured attention weights."""
        self._captured_attention = []
        self._remove_attention_hooks()

    def __del__(self):
        """Cleanup hooks on deletion."""
        try:
            self._remove_attention_hooks()
        except Exception:
            pass


# Global accessor
# Singleton instance
_engine_instance = None

def get_model_engine() -> TransformerModelEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TransformerModelEngine()
    return _engine_instance
