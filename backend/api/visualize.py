"""視覺化數據 API."""

from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/layers")
async def list_layers():
    """Return available model layers for visualization."""
    return {
        "layers": [
            {"id": "embedding", "name": "Embedding Projection", "type": "embedding"},
            {"id": "attention_0", "name": "Attention Layer 0", "type": "attention"},
            {"id": "ffn_0", "name": "FFN Layer 0", "type": "ffn"},
            {"id": "attention_1", "name": "Attention Layer 1", "type": "attention"},
            {"id": "ffn_1", "name": "FFN Layer 1", "type": "ffn"},
            {"id": "output", "name": "Output Layer", "type": "output"},
        ]
    }


@router.get("/token-features")
async def get_token_features(layer: str = Query(...), token_idx: int = Query(0)):
    """Get features for a specific token at a given layer."""
    return {
        "layer": layer,
        "token_idx": token_idx,
        "features": [],  # Would be populated from hook registry cache
    }


@router.get("/embedding")
async def get_embedding(token_idx: int = Query(0)):
    """Get embedding vector for a specific token."""
    return {
        "token_idx": token_idx,
        "embedding": [],  # Would be populated from hook registry cache
    }


@router.get("/attention")
async def get_attention(layer: int = Query(0)):
    """Get attention scores for a specific layer."""
    return {
        "layer": layer,
        "attention": [],  # Would be populated from hook registry cache
    }
