"""Single model-construction entry point shared by training and analyzers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch.nn as nn

from models import baselines, sequence


@dataclass(frozen=True)
class ModelBuildSpec:
    name: str
    input_shape: Tuple[int, int]
    num_classes: int = 2
    normalization: Optional[str] = None
    old_single_encoder_ablation: Optional[str] = None
    use_cls_token: Optional[bool] = None
    f_act_add_dim: Optional[int] = None
    simple_linear_head: Optional[str] = None


@contextmanager
def _temporary_architecture_overrides(spec: ModelBuildSpec) -> Iterator[None]:
    overrides = {
        "USE_CLS_TOKEN": spec.use_cls_token,
        "F_ACT_ADD_DIM": spec.f_act_add_dim,
        "USE_SIMPLE_LINEAR_HEAD": spec.simple_linear_head,
    }
    original = {
        name: getattr(sequence, name)
        for name, value in overrides.items()
        if value is not None
    }
    try:
        for name, value in overrides.items():
            if value is not None:
                setattr(sequence, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(sequence, name, value)


def build_model(spec: ModelBuildSpec) -> nn.Module:
    """Build one configured architecture without caller-side module patching."""
    norm_kwargs = (
        {"norm": spec.normalization}
        if spec.normalization is not None
        else {}
    )
    common = {
        "input_shape": spec.input_shape,
        "num_classes": int(spec.num_classes),
        **norm_kwargs,
    }

    with _temporary_architecture_overrides(spec):
        if spec.name == "trans":
            return sequence.MahjongTransformer(**common)
        if spec.name == "transBertBase":
            return sequence.TransformerBERTBASE(**common)
        if spec.name == "transBertLarge":
            return sequence.TransformerBERTLARGE(**common)
        if spec.name == "transOriginal":
            if spec.old_single_encoder_ablation is not None:
                common["old_single_encoder_ablation"] = (
                    spec.old_single_encoder_ablation
                )
            return sequence.TransformerORI(**common)
        if spec.name == "transTest":
            return sequence.TransformerTiny(**common)
        if spec.name == "transVOG":
            return sequence.TransformerVOG(**common)
        if spec.name == "cnn":
            return baselines.CNN(**common)
        if spec.name == "resnet10":
            return baselines.ResNet10(**common)
        if spec.name == "resnet14":
            return baselines.ResNet14(**common)
        if spec.name == "resnet18":
            return baselines.ResNet18(**common)
        if spec.name == "resnet22":
            return baselines.ResNet22(**common)
        if spec.name == "conformerTest":
            return sequence.ConformerTiny(**common)
        if spec.name == "conformerSmall":
            return sequence.ConformerSmall(**common)
        if spec.name == "conformerOriginal":
            return sequence.ConformerORI(**common)
        if spec.name == "conformerLarge":
            return sequence.ConformerLarge(**common)
    raise ValueError(f"Unsupported model name: {spec.name!r}")


__all__ = ["ModelBuildSpec", "build_model"]
