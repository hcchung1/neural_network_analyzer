"""Model comparison API: weight delta metrics and attention cosine similarity.

Loads two checkpoints (A and B), computes per-layer weight differences and
attention-matrix cosine similarity, and returns JSON-serialisable results
suitable for Plotly visualisation on the frontend.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from model_engine.compatibility import (
    inspect_checkpoint,
    infer_model_name_and_spec,
    load_checkpoint_strictly,
)

router = APIRouter(tags=["compare"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    checkpoint_a: str = Field(..., description="Path to Model A checkpoint")
    checkpoint_b: str = Field(..., description="Path to Model B checkpoint")
    device: str = Field(default="cpu")
    # Optional: limit attention comparison to specific layers
    attention_layers: Optional[List[int]] = Field(
        default=None,
        description="Layer indices to compare attention on (None = all)",
    )


class WeightDeltaLayer(BaseModel):
    """Per-layer weight delta statistics."""
    layer_name: str
    shape: List[int]
    l2_norm_a: float
    l2_norm_b: float
    delta_l2_norm: float
    delta_mean: float
    delta_std: float
    delta_abs_max: float
    cosine_similarity: float
    # histogram data for the weight delta distribution
    hist_counts: List[int]
    hist_bin_edges: List[float]


class AttentionSimilarityLayer(BaseModel):
    """Per-layer attention cosine similarity."""
    layer_index: int
    cosine_similarity: float
    # flattened heatmap of difference (A - B) for one head (averaged across heads)
    diff_matrix: List[List[float]]
    matrix_size: int


class CompareResponse(BaseModel):
    status: str = "ok"
    model_a_name: str
    model_b_name: str
    total_params_a: int
    total_params_b: int
    weight_deltas: List[WeightDeltaLayer]
    attention_similarity: List[AttentionSimilarityLayer]
    # Summary
    mean_weight_cosine: float
    mean_attention_cosine: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state_dict(checkpoint_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a checkpoint and return its state_dict + inferred model name."""
    from pathlib import Path

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    blob = torch.load(path, map_location="cpu")
    state_dict, _ = inspect_checkpoint(blob)
    spec, _ = infer_model_name_and_spec(state_dict)
    return state_dict, spec.name


def _compute_weight_deltas(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    n_bins: int = 50,
) -> List[WeightDeltaLayer]:
    """Compute per-parameter delta statistics between two state dicts."""
    common_keys = sorted(set(sd_a.keys()) & set(sd_b.keys()))
    results: List[WeightDeltaLayer] = []

    for key in common_keys:
        ta = sd_a[key].float()
        tb = sd_b[key].float()
        if ta.shape != tb.shape:
            continue  # skip shape-mismatched params

        delta = ta - tb
        delta_flat = delta.flatten()

        # cosine similarity between two weight tensors
        cos_sim = F.cosine_similarity(
            ta.flatten().unsqueeze(0),
            tb.flatten().unsqueeze(0),
        ).item()

        # histogram of delta values
        counts, bin_edges = np.histogram(
            delta_flat.numpy(), bins=n_bins
        )

        results.append(WeightDeltaLayer(
            layer_name=key,
            shape=list(ta.shape),
            l2_norm_a=ta.norm().item(),
            l2_norm_b=tb.norm().item(),
            delta_l2_norm=delta.norm().item(),
            delta_mean=delta_flat.mean().item(),
            delta_std=delta_flat.std().item(),
            delta_abs_max=delta_flat.abs().max().item(),
            cosine_similarity=cos_sim,
            hist_counts=counts.tolist(),
            hist_bin_edges=bin_edges.tolist(),
        ))

    return results


def _build_model_and_get_attention(
    checkpoint_path: str,
    state_dict: Dict[str, torch.Tensor],
    model_name: str,
    device: str = "cpu",
) -> List[Optional[torch.Tensor]]:
    """Build a model from checkpoint, run a dummy forward pass, and capture
    attention weight matrices.

    Returns a list of tensors, one per attention layer.
    Each tensor has shape [batch, heads, seq_len, seq_len].
    """
    try:
        model, spec, _ = load_checkpoint_strictly(
            checkpoint_path=checkpoint_path,
            device=device,
        )
    except Exception:
        return []

    model.eval()

    # Generate a dummy input based on model schema
    seq_len = getattr(model, "seq_len", 25)
    feature_dim = getattr(model, "_expected_feature_dim", None)
    if feature_dim is None:
        feature_dim = getattr(model, "input_features", 256)
    if not isinstance(seq_len, int) or seq_len <= 0:
        seq_len = 25
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        feature_dim = 256

    dummy_input = torch.randn(1, seq_len, feature_dim, device=device)

    # Capture attention weights via hooks
    captured: List[Optional[torch.Tensor]] = []

    def make_hook(idx: int):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                while len(captured) <= idx:
                    captured.append(None)
                captured[idx] = out[1].detach().cpu()
        return hook

    handles = []
    layer_idx = 0
    for name, module in model.named_modules():
        mtype = type(module).__name__
        if mtype in ("MultiheadAttention", "MultiHeadAttention"):
            handles.append(module.register_forward_hook(make_hook(layer_idx)))
            layer_idx += 1

    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception:
            pass

    # Also try to pull from module attributes
    if not any(c is not None for c in captured):
        idx = 0
        for name, module in model.named_modules():
            mtype = type(module).__name__
            if mtype == "MultiHeadAttention":
                if hasattr(module, "last_attention_weights") and module.last_attention_weights is not None:
                    while len(captured) <= idx:
                        captured.append(None)
                    captured[idx] = module.last_attention_weights.detach().cpu()
                idx += 1

    for h in handles:
        h.remove()

    return captured


def _compute_attention_similarity(
    attn_a: List[Optional[torch.Tensor]],
    attn_b: List[Optional[torch.Tensor]],
    layer_indices: Optional[List[int]] = None,
    max_matrix_size: int = 32,
) -> List[AttentionSimilarityLayer]:
    """Compute per-layer cosine similarity of attention matrices."""
    n_layers = min(len(attn_a), len(attn_b))
    if n_layers == 0:
        return []

    results: List[AttentionSimilarityLayer] = []
    for i in range(n_layers):
        if layer_indices is not None and i not in layer_indices:
            continue
        wa = attn_a[i]
        wb = attn_b[i]
        if wa is None or wb is None:
            continue

        # Average over batch and heads -> [seq, seq]
        while wa.dim() > 2:
            wa = wa.mean(dim=0)
        while wb.dim() > 2:
            wb = wb.mean(dim=0)

        # Ensure same size
        min_size = min(wa.shape[0], wb.shape[0], wa.shape[1], wb.shape[1])
        wa = wa[:min_size, :min_size]
        wb = wb[:min_size, :min_size]

        cos_sim = F.cosine_similarity(
            wa.flatten().unsqueeze(0).float(),
            wb.flatten().unsqueeze(0).float(),
        ).item()

        diff = (wa - wb).float()

        # Downsample if matrix is too large for frontend rendering
        display_size = min(min_size, max_matrix_size)
        if min_size > max_matrix_size:
            diff = F.interpolate(
                diff.unsqueeze(0).unsqueeze(0),
                size=(display_size, display_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        results.append(AttentionSimilarityLayer(
            layer_index=i,
            cosine_similarity=cos_sim,
            diff_matrix=diff.numpy().tolist(),
            matrix_size=display_size,
        ))

    return results


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/compare", response_model=CompareResponse)
async def compare_models(req: CompareRequest):
    """Compare two model checkpoints: weight deltas and attention similarity."""
    try:
        sd_a, name_a = _load_state_dict(req.checkpoint_a, req.device)
        sd_b, name_b = _load_state_dict(req.checkpoint_b, req.device)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"error": {"code": "CHECKPOINT_NOT_FOUND", "message": str(e)}})
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": {"code": "CHECKPOINT_LOAD_FAILED", "message": str(e)}})

    # --- Weight deltas ---
    weight_deltas = _compute_weight_deltas(sd_a, sd_b)

    total_a = sum(t.numel() for t in sd_a.values())
    total_b = sum(t.numel() for t in sd_b.values())

    mean_weight_cos = (
        sum(d.cosine_similarity for d in weight_deltas) / len(weight_deltas)
        if weight_deltas else 0.0
    )

    # --- Attention similarity ---
    attn_a = _build_model_and_get_attention(req.checkpoint_a, sd_a, name_a, req.device)
    attn_b = _build_model_and_get_attention(req.checkpoint_b, sd_b, name_b, req.device)

    attention_similarity = _compute_attention_similarity(
        attn_a, attn_b, req.attention_layers
    )

    mean_attn_cos = (
        sum(a.cosine_similarity for a in attention_similarity) / len(attention_similarity)
        if attention_similarity else None
    )

    return CompareResponse(
        model_a_name=name_a,
        model_b_name=name_b,
        total_params_a=total_a,
        total_params_b=total_b,
        weight_deltas=weight_deltas,
        attention_similarity=attention_similarity,
        mean_weight_cosine=mean_weight_cos,
        mean_attention_cosine=mean_attn_cos,
    )
