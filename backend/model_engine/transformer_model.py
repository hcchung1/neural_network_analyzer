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
                self._model = MahjongTransformer()
                checkpoint = torch.load(path, map_location=device)
                
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["model_state_dict"])
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict):
                    self._model.load_state_dict(checkpoint)
                else:
                    self._model = checkpointisz  # full model object saved directly
                    
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
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.fc_out = nn.Linear(d_model, vocab_size)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                seq_len = x.size(1)
                x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
                x = self.transformer(x)
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
                tensor = torch.tensor(input_data, dtype=torch.long, device=self._device)
            elif isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data).long().to(self._device)
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
def get_model_engine() -> TransformerModelEngine:
    return TransformerModelEngine()
