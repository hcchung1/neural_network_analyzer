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
    """Try to import MahjongTransformer from the project."""
    global _MAHJONG_TRANSFORMER
    if _MAHJONG_TRANSFORMER is not None:
        return _MAHJONG_TRANSFORMER
    
    # Try multiple possible paths
    possible_roots = [
        # From ArchAnalyzer root (go up 4 levels from backend/model_engine/)
        Path(__file__).parent.parent.parent.parent / "Transformer",
        # From current working directory
        Path.cwd() / "Transformer",
        # Absolute path (adjust if needed for your container)
        Path("/workspace/Study/Mahjong-Hidden-Information-Forecast/Transformer"),
    ]
    
    for root in possible_roots:
        if root.exists():
            try:
                import importlib.util
                # Add the Transformer directory to sys.path so transformer.py can find utils
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                
                spec = importlib.util.spec_from_file_location(
                    "transformer_module", str(root / "transformer.py")
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["transformer_module"] = module
                    spec.loader.exec_module(module)
                    _MAHJONG_TRANSFORMER = module.MahjongTransformer
                    print(f"[TransformerModelEngine] Successfully imported MahjongTransformer from {root}")
                    return _MAHJONG_TRANSFORMER
            except Exception as e:
                print(f"[TransformerModelEngine] Failed to import from {root}: {e}")
                continue
    
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def is_cuda_available() -> bool:
        if not _TORCH_AVAILABLE:
            return False
        import torch
        return torch.cuda.is_available()

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
            # Try to import the real MahjongTransformer first
            MahjongTransformer = _try_import_mahjong_transformer()
            
            if MahjongTransformer is not None:
                print("[TransformerModelEngine] Using real MahjongTransformer")
                # Inspect checkpoint first to infer correct hyperparameters
                ckpt = torch.load(path, map_location="cpu")
                state_dict = ckpt
                if isinstance(ckpt, dict):
                    if "model_state_dict" in ckpt:
                        state_dict = ckpt["model_state_dict"]
                    elif "state_dict" in ckpt:
                        state_dict = ckpt["state_dict"]
                # Infer shapes from checkpoint keys
                d_model = None
                seq_len = None
                try:
                    pe_shape = state_dict.get("pos_encoding.pe")
                    if pe_shape is not None and hasattr(pe_shape, "shape"):
                        seq_len = pe_shape.shape[1]  # e.g. 33
                        d_model = pe_shape.shape[2]  # e.g. 512
                    else:
                        # fallback: inspect proj_init.weight
                        proj_w = state_dict.get("proj_init.weight")
                        if proj_w is not None and hasattr(proj_w, "shape"):
                            d_model = proj_w.shape[0]
                except Exception:
                    pass
                # Build kwargs; allow passing different defaults without changing signature of TransformerModelEngine.load_model
                kwargs = {}
                if d_model is not None:
                    kwargs["d_model"] = d_model
                if seq_len is not None:
                    feature_dim = None
                    # Check for Heterogeneous Tokens (proj_init + proj_act)
                    proj_init_w = state_dict.get("proj_init.weight")
                    proj_act_w = state_dict.get("proj_act.weight")
                    if proj_init_w is not None and proj_act_w is not None:
                        # Heterogeneous Tokens mode
                        # proj_init.weight shape: (d_model, F_INIT)
                        # proj_act.weight shape: (d_model, act_dim)
                        f_init = proj_init_w.shape[1] if hasattr(proj_init_w, "shape") else 850
                        act_dim = proj_act_w.shape[1] if hasattr(proj_act_w, "shape") else 58
                        # input_features should be max(F_INIT, act_dim) to match F_PAD_ADD or F_PAD
                        feature_dim = max(f_init, act_dim)
                        print(f"[TransformerModelEngine] Detected Heterogenous Tokens: F_INIT={f_init}, act_dim={act_dim}, input_features={feature_dim}")
                    else:
                        # Single projection mode: try to infer from input_projection.weight
                        proj_w = state_dict.get("input_projection.weight")
                        if proj_w is not None and hasattr(proj_w, "shape"):
                            feature_dim = proj_w.shape[1]
                    if feature_dim is not None:
                        kwargs["input_shape"] = (seq_len, feature_dim)
                    else:
                        kwargs["input_shape"] = (seq_len, None)
                # Infer n_heads: d_model must be divisible by n_heads.
                n_heads = None
                try:
                    w_q = state_dict.get("transformer_layers.0.attention.w_q.weight")
                    if w_q is not None and hasattr(w_q, "shape"):
                        out_dim = w_q.shape[0]  # d_model
                        in_dim = w_q.shape[1]   # d_model
                        # For standard MHA, qkv weight is (d_model, d_model)
                        # head_dim = d_model // n_heads, so n_heads must divide d_model evenly.
                        # Try common n_heads values: 16, 8, 12, 4, 2, 1
                        for candidate in [16, 8, 12, 4, 2, 1]:
                            if d_model is not None and d_model % candidate == 0:
                                n_heads = candidate
                                break
                except Exception:
                    pass
                if n_heads is not None:
                    kwargs["n_heads"] = n_heads

                # Infer n_layers from the number of transformer_layers
                try:
                    layer_indices = set()
                    for k in state_dict.keys():
                        if k.startswith("transformer_layers."):
                            parts = k.split(".")
                            if len(parts) > 1 and parts[1].isdigit():
                                layer_indices.add(int(parts[1]))
                    if layer_indices:
                        kwargs["n_layers"] = max(layer_indices) + 1
                except Exception:
                    pass

                # Infer d_ff from feed_forward weight dimensions
                try:
                    ff_w = state_dict.get("transformer_layers.0.feed_forward.0.weight")
                    if ff_w is not None and hasattr(ff_w, "shape"):
                        d_ff = ff_w.shape[0]
                        kwargs["d_ff"] = d_ff
                except Exception:
                    pass
                if kwargs:
                    print(f"[TransformerModelEngine] Inferred model architecture: {kwargs}")
                    self._model = MahjongTransformer(**kwargs)
                else:
                    self._model = MahjongTransformer()
                
                # Load state dict with strict=False and handle mismatches
                state_dict_to_load = None
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    state_dict_to_load = ckpt["model_state_dict"]
                elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                    state_dict_to_load = ckpt["state_dict"]
                elif isinstance(ckpt, dict):
                    state_dict_to_load = ckpt
                else:
                    self._model = ckpt
                    self._model_class_name = "MahjongTransformer"
                    return True
                
                # Try strict loading first, fallback to non-strict
                try:
                    self._model.load_state_dict(state_dict_to_load, strict=True)
                    print("[TransformerModelEngine] Model loaded with strict=True")
                except RuntimeError as e:
                    print(f"[TransformerModelEngine] Strict loading failed: {e}")
                    print("[TransformerModelEngine] Attempting non-strict loading...")
                    missing, unexpected = self._model.load_state_dict(state_dict_to_load, strict=False)
                    if missing:
                        print(f"[TransformerModelEngine] Missing keys: {missing}")
                    if unexpected:
                        print(f"[TransformerModelEngine] Unexpected keys: {unexpected}")
                    print("[TransformerModelEngine] Non-strict loading completed")
                self._model_class_name = "MahjongTransformer"
            else:
                # Fall back to dummy model
                print("[TransformerModelEngine] Using dummy model (MahjongTransformer not found)")
                self._model = self._create_dummy_model()
                checkpoint = torch.load(path, map_location=device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["model_state_dict"])
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict) and "model" in checkpoint:
                    self._model = checkpoint["model"]
                elif isinstance(checkpoint, dict):
                    self._model.load_state_dict(checkpoint)
                else:
                    self._model = checkpoint
                self._model_class_name = type(self._model).__name__
            self._model.to(device)
            self._model.eval()
            self._is_loaded = True
            print(f"[TransformerModelEngine] Model loaded to {device}")
            return True

        except Exception as e:
            print(f"[TransformerModelEngine] Failed to load model: {e}")
            self._is_loaded = False
            return False

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
                # Convert list to float tensor (not long!)
                tensor = torch.tensor(input_data, dtype=torch.float32, device=self._device)
            elif isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data).float().to(self._device)
            else:
                tensor = input_data.to(self._device)

            with torch.no_grad():
                output = self._model(tensor)

            return {
                "output_shape": list(output.shape),
                "output_sample": output[0, :5].cpu().numpy().tolist(),
            }
        except Exception as e:
            return {"error": str(e)}


# Global accessor
# Singleton instance
_engine_instance = None

def get_model_engine() -> TransformerModelEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TransformerModelEngine()
    return _engine_instance
