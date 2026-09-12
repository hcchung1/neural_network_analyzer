"""Model build, checkpoint compatibility, and capability registry module for ArchAnalyzer.

Uses Transformer.models.factory.build_model and ModelBuildSpec as the authoritative path.
Strictly validates checkpoints with zero missing or unexpected keys.
Exposes capability descriptors per model family as defined in spec.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import torch
import torch.nn as nn

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if BACKEND_DIR.exists() and str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from models.factory import ModelBuildSpec, build_model
    from models import sequence, baselines
    _FACTORY_AVAILABLE = True
except Exception as err:
    _FACTORY_AVAILABLE = False
    ModelBuildSpec = None
    build_model = None
    print(f"[Compatibility] Failed to import Transformer factory: {err}")


ALL_MODEL_NAMES = [
    "trans", "transBertBase", "transBertLarge", "transOriginal", "transTest",
    "transVOG", "cnn",
    "resnet10", "resnet14", "resnet18", "resnet22",
    "conformerTest", "conformerSmall", "conformerOriginal", "conformerLarge"
]

RESNET_MODELS = {"resnet10", "resnet14", "resnet18", "resnet22"}
CNN_MODELS = {"cnn"}
CONFORMER_MODELS = {"conformerTest", "conformerSmall", "conformerOriginal", "conformerLarge"}
TRANSFORMER_MODELS = {"trans", "transBertBase", "transBertLarge", "transOriginal", "transTest"}
VOG_MODELS = {"transVOG"}


@dataclass
class ModelCapabilities:
    family: str
    has_input_token: bool = False
    has_embedding: bool = False
    has_layer_input: bool = False
    has_attention_heatmap: bool = False
    has_ffn_activation: bool = False
    has_latent_distribution: bool = False
    has_convolution_feature_map: bool = False
    has_channel_activation: bool = False
    has_residual_stage: bool = False
    has_logits: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def describe_capabilities(model_name: str) -> ModelCapabilities:
    """Return model-family capability descriptor."""
    if model_name in VOG_MODELS:
        return ModelCapabilities(
            family="VOG",
            has_input_token=True,
            has_embedding=True,
            has_layer_input=True,
            has_attention_heatmap=True,
            has_ffn_activation=True,
            has_latent_distribution=True,
            has_logits=True,
        )
    if model_name in CONFORMER_MODELS:
        return ModelCapabilities(
            family="Conformer",
            has_embedding=True,
            has_attention_heatmap=True,
            has_convolution_feature_map=True,
            has_ffn_activation=True,
            has_logits=True,
        )
    if model_name in CNN_MODELS:
        return ModelCapabilities(
            family="CNN",
            has_convolution_feature_map=True,
            has_channel_activation=True,
            has_logits=True,
        )
    if model_name in RESNET_MODELS:
        return ModelCapabilities(
            family="ResNet",
            has_convolution_feature_map=True,
            has_residual_stage=True,
            has_logits=True,
        )
    # Default to Transformer family
    return ModelCapabilities(
        family="Transformer",
        has_input_token=True,
        has_embedding=True,
        has_layer_input=True,
        has_attention_heatmap=True,
        has_ffn_activation=True,
        has_logits=True,
    )


def inspect_checkpoint(checkpoint_blob: Any, requested_phase: Optional[str] = None) -> Tuple[Dict[str, torch.Tensor], Optional[str]]:
    """Extract model state_dict and phase name from checkpoint formats:
    1. Plain state_dict
    2. Dict with model_state_dict
    3. Dict with state_dict
    4. Dict with phase_states -> phase_name -> model_state_dict
    """
    if not isinstance(checkpoint_blob, dict):
        raise ValueError("CHECKPOINT_FORMAT_UNSUPPORTED: Checkpoint is not a dict")

    if "phase_states" in checkpoint_blob and isinstance(checkpoint_blob["phase_states"], dict):
        phase_states = checkpoint_blob["phase_states"]
        if not phase_states:
            raise ValueError("CHECKPOINT_FORMAT_UNSUPPORTED: Empty phase_states in checkpoint")

        selected_phase = requested_phase
        if selected_phase is None:
            if "all" in phase_states:
                selected_phase = "all"
            elif len(phase_states) == 1:
                selected_phase = next(iter(phase_states.keys()))
            else:
                phases_str = ", ".join(sorted(phase_states.keys()))
                raise ValueError(
                    f"CHECKPOINT_PHASE_REQUIRED: Multiple phases available [{phases_str}]. Specify requested_phase."
                )

        if selected_phase not in phase_states:
            phases_str = ", ".join(sorted(phase_states.keys()))
            raise ValueError(f"CHECKPOINT_PHASE_REQUIRED: Phase '{selected_phase}' not in [{phases_str}]")

        phase_blob = phase_states[selected_phase]
        if isinstance(phase_blob, dict) and "model_state_dict" in phase_blob:
            return phase_blob["model_state_dict"], selected_phase
        raise ValueError(f"CHECKPOINT_FORMAT_UNSUPPORTED: Missing model_state_dict in phase '{selected_phase}'")

    if "model_state_dict" in checkpoint_blob and isinstance(checkpoint_blob["model_state_dict"], dict):
        return checkpoint_blob["model_state_dict"], None

    if "state_dict" in checkpoint_blob and isinstance(checkpoint_blob["state_dict"], dict):
        return checkpoint_blob["state_dict"], None

    if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint_blob.items()):
        return checkpoint_blob, None

    raise ValueError("CHECKPOINT_FORMAT_UNSUPPORTED: Unrecognized checkpoint structure")


def infer_model_name_and_spec(state_dict: Dict[str, torch.Tensor], manifest_spec: Optional[Dict[str, Any]] = None) -> Tuple[ModelBuildSpec, Optional[nn.Module]]:
    """Infer ModelBuildSpec or construct exact model matching state_dict shapes."""
    if manifest_spec and isinstance(manifest_spec, dict):
        name = manifest_spec.get("name", "transOriginal")
        input_shape = tuple(manifest_spec.get("input_shape", [25, 12308]))
        num_classes = int(manifest_spec.get("num_classes", 2))
        normalization = manifest_spec.get("normalization")
        old_ablation = manifest_spec.get("old_single_encoder_ablation")
        use_cls = manifest_spec.get("use_cls_token")
        f_act_add = manifest_spec.get("f_act_add_dim")
        linear_head = manifest_spec.get("simple_linear_head")
        spec = ModelBuildSpec(
            name=name,
            input_shape=input_shape,
            num_classes=num_classes,
            normalization=normalization,
            old_single_encoder_ablation=old_ablation,
            use_cls_token=use_cls,
            f_act_add_dim=f_act_add,
            simple_linear_head=linear_head,
        )
        return spec, None

    keys = set(state_dict.keys())

    num_classes = 2
    for head_key in ("tenpai_classifier.weight", "classifier.weight", "fc2.weight", "fc.weight"):
        if head_key in state_dict:
            num_classes = int(state_dict[head_key].shape[0])
            break

    # ResNet family
    if "layer1.0.conv1.weight" in keys or "layer5.0.conv1.weight" in keys:
        if "layer5.0.conv1.weight" in keys or "layer4.2.conv1.weight" in keys:
            name = "resnet22"
        elif "layer4.1.conv1.weight" in keys:
            if "layer2.1.conv1.weight" in keys:
                name = "resnet18"
            else:
                name = "resnet14"
        else:
            name = "resnet10"
        return ModelBuildSpec(name=name, input_shape=(362, 34), num_classes=num_classes, normalization="gn"), None

    # CNN family
    if "conv1.weight" in keys and "conv4.weight" in keys:
        return ModelBuildSpec(name="cnn", input_shape=(362, 34), num_classes=num_classes, normalization="bn"), None

    # VOG family
    if "prior_encoder.proj_init.weight" in keys or "prior_z_proj.weight" in keys:
        return ModelBuildSpec(name="transVOG", input_shape=(25, 12308), num_classes=num_classes, normalization="ln"), None

    # Check for direct custom MahjongTransformer instantiation when shapes differ from standard factory presets
    seq_len = 25
    d_model = 512
    d_ff = 2048
    if "pos_encoding.pe" in state_dict:
        seq_len = int(state_dict["pos_encoding.pe"].shape[1])
        d_model = int(state_dict["pos_encoding.pe"].shape[2])

    layer_indices = [int(k.split(".")[1]) for k in keys if k.startswith("transformer_layers.") and k.split(".")[1].isdigit()]
    n_layers = max(layer_indices) + 1 if layer_indices else 6

    # Infer d_ff from feed_forward layer 0
    for ff_key in ("transformer_layers.0.feed_forward.0.weight", "transformer_layers.0.ff1.0.weight"):
        if ff_key in state_dict:
            d_ff = int(state_dict[ff_key].shape[0])
            break

    # Infer n_heads from attention w_q shape
    n_heads = 8
    if "transformer_layers.0.attention.w_q.weight" in state_dict:
        for candidate in (16, 12, 8, 4, 3, 2, 1):
            if d_model % candidate == 0:
                n_heads = candidate
                break

    # Check input projection structure (input_projection vs proj_init/proj_act)
    feature_dim = 12308
    if "input_projection.weight" in state_dict:
        feature_dim = int(state_dict["input_projection.weight"].shape[1])
    elif "proj_init.weight" in state_dict:
        feature_dim = int(state_dict["proj_init.weight"].shape[1])

    # Standard factory specs matched by name
    is_conformer = any(k.startswith("transformer_layers.0.conv_module") or k.startswith("transformer_layers.0.ff2") for k in keys)

    if is_conformer:
        if n_layers <= 2: name = "conformerTest"
        elif n_layers <= 4: name = "conformerSmall"
        elif n_layers <= 6: name = "conformerOriginal"
        else: name = "conformerLarge"
        spec = ModelBuildSpec(name=name, input_shape=(seq_len, feature_dim), num_classes=num_classes, normalization="ln")
        return spec, None

    # Standard Transformer names
    if n_layers <= 2 and d_model <= 128:
        name = "transTest"
    elif n_layers >= 24 or d_model >= 1024:
        name = "transBertLarge"
    elif n_layers >= 12 or d_model >= 768:
        name = "transBertBase"
    else:
        name = "transOriginal"

    spec = ModelBuildSpec(name=name, input_shape=(seq_len, feature_dim), num_classes=num_classes, normalization="ln")

    # If parameters deviate from standard factory preset, build custom MahjongTransformer instance directly
    expected_default_d_ff = 512 if name == "transTest" else (3072 if name == "transBertBase" else (4096 if name == "transBertLarge" else 2048))
    if d_model not in (128, 512, 768, 1024) or seq_len != 25 or feature_dim != 12308 or d_ff != expected_default_d_ff:
        flinear_in_shape = None
        if "tenpai_classifier.weight" in state_dict:
            w_shape = state_dict["tenpai_classifier.weight"].shape
            if len(w_shape) == 2 and w_shape[1] == seq_len * d_model:
                flinear_in_shape = w_shape[1]

        custom_model = sequence.MahjongTransformer(
            input_shape=(seq_len, feature_dim),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            num_classes=num_classes,
            norm="ln"
        )
        if "input_projection.weight" in state_dict:
            custom_model.input_projection = nn.Linear(feature_dim, d_model, bias=False)
            custom_model.use_heter_tokens = False
            if hasattr(custom_model, "proj_init"): delattr(custom_model, "proj_init")
            if hasattr(custom_model, "proj_act"): delattr(custom_model, "proj_act")

        if flinear_in_shape is not None:
            custom_model.tenpai_classifier = nn.Linear(flinear_in_shape, num_classes)
            custom_model.head_mode_str = "flinear"

        return spec, custom_model

    return spec, None


def load_checkpoint_strictly(
    checkpoint_path: str,
    manifest_spec: Optional[Dict[str, Any]] = None,
    requested_phase: Optional[str] = None,
    device: str = "cpu"
) -> Tuple[nn.Module, ModelBuildSpec, Optional[str]]:
    """Load model checkpoint with strict=True.
    Raises structured exceptions if tensors miss or mismatch.
    """
    if not _FACTORY_AVAILABLE:
        raise RuntimeError("MODEL_BUILD_FAILED: Transformer model factory is unavailable")

    if not checkpoint_path or checkpoint_path.strip() in ("", "dummy"):
        name_for_dummy = manifest_spec.get("name", "transTest") if manifest_spec else "transTest"
        spec = ModelBuildSpec(name=name_for_dummy, input_shape=(25, 12308), num_classes=2, normalization="ln")
        model = build_model(spec)
        model.to(device)
        model.eval()
        return model, spec, None

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(f"ARTIFACT_OUTSIDE_ALLOWED_ROOT: Checkpoint path does not exist or is a directory: {path}")

    blob = torch.load(path, map_location="cpu")
    state_dict, loaded_phase = inspect_checkpoint(blob, requested_phase)

    spec, custom_model = infer_model_name_and_spec(state_dict, manifest_spec)
    model = custom_model if custom_model is not None else build_model(spec)

    model_state = model.state_dict()
    missing_keys = set(model_state.keys()) - set(state_dict.keys())
    unexpected_keys = set(state_dict.keys()) - set(model_state.keys())

    if missing_keys:
        missing_sample = sorted(list(missing_keys))[:5]
        raise ValueError(
            f"STATE_DICT_MISSING_KEYS: Model '{spec.name}' has {len(missing_keys)} missing keys in checkpoint. "
            f"Sample missing: {missing_sample}"
        )

    if unexpected_keys:
        unexpected_sample = sorted(list(unexpected_keys))[:5]
        raise ValueError(
            f"STATE_DICT_UNEXPECTED_KEYS: Checkpoint has {len(unexpected_keys)} unexpected keys for '{spec.name}'. "
            f"Sample unexpected: {unexpected_sample}"
        )

    shape_mismatches = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            shape_mismatches.append(f"{k}: expected {tuple(model_state[k].shape)}, got {tuple(v.shape)}")

    if shape_mismatches:
        raise ValueError(
            f"STATE_DICT_SHAPE_MISMATCH: {len(shape_mismatches)} tensor shape mismatch(es). "
            f"First mismatch: {shape_mismatches[0]}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, spec, loaded_phase
