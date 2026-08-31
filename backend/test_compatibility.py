import unittest
import tempfile
import sys
from pathlib import Path
import torch

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model_engine.compatibility import (
    ALL_MODEL_NAMES,
    ModelBuildSpec,
    build_model,
    load_checkpoint_strictly,
    inspect_checkpoint,
    infer_model_name_and_spec,
    describe_capabilities,
)

class ModelCompatibilityTest(unittest.TestCase):
    def test_capabilities_descriptor_per_family(self):
        """Verify model capabilities match expectations per model family."""
        trans_cap = describe_capabilities("transOriginal")
        self.assertEqual(trans_cap.family, "Transformer")
        self.assertTrue(trans_cap.has_attention_heatmap)
        self.assertFalse(trans_cap.has_convolution_feature_map)

        cnn_cap = describe_capabilities("cnn")
        self.assertEqual(cnn_cap.family, "CNN")
        self.assertFalse(cnn_cap.has_attention_heatmap)
        self.assertTrue(cnn_cap.has_convolution_feature_map)

        resnet_cap = describe_capabilities("resnet18")
        self.assertEqual(resnet_cap.family, "ResNet")
        self.assertTrue(resnet_cap.has_residual_stage)
        self.assertFalse(resnet_cap.has_attention_heatmap)

    def test_factory_builds_all_15_models(self):
        """Construct all 15 model architectures defined in ModelBuildSpec."""
        for name in ALL_MODEL_NAMES:
            input_shape = (362, 34) if name.startswith("resnet") or name == "cnn" else (4, 10)
            spec = ModelBuildSpec(name=name, input_shape=input_shape, num_classes=2)
            model = build_model(spec)
            self.assertIsNotNone(model, f"Failed to build {name}")
            self.assertTrue(isinstance(model, torch.nn.Module))

    def test_strict_checkpoint_loading_single_state_dict(self):
        """Verify strict loading for plain state_dict checkpoint."""
        spec = ModelBuildSpec(name="transTest", input_shape=(4, 10), num_classes=2, normalization="ln")
        model = build_model(spec)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_model.pth"
            torch.save(model.state_dict(), path)

            loaded_model, loaded_spec, phase = load_checkpoint_strictly(str(path))
            self.assertEqual(loaded_spec.name, "transTest")
            self.assertIsNone(phase)
            self.assertTrue(isinstance(loaded_model, torch.nn.Module))

    def test_strict_checkpoint_loading_phase_states(self):
        """Verify strict loading for train_state phase_states checkpoint."""
        spec = ModelBuildSpec(name="transTest", input_shape=(4, 10), num_classes=2, normalization="ln")
        model = build_model(spec)
        checkpoint = {
            "checkpoint_type": "train_state",
            "phase_states": {
                "early": {"model_state_dict": model.state_dict()},
                "late": {"model_state_dict": model.state_dict()},
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train_state.pth"
            torch.save(checkpoint, path)

            with self.assertRaises(ValueError) as ctx:
                load_checkpoint_strictly(str(path))
            self.assertIn("CHECKPOINT_PHASE_REQUIRED", str(ctx.exception))

            loaded_model, loaded_spec, phase = load_checkpoint_strictly(str(path), requested_phase="early")
            self.assertEqual(phase, "early")
            self.assertEqual(loaded_spec.name, "transTest")

    def test_strict_loading_rejects_missing_keys(self):
        """Verify strict loading fails when state_dict has missing keys."""
        spec = ModelBuildSpec(name="transTest", input_shape=(4, 10), num_classes=2, normalization="ln")
        model = build_model(spec)
        sd = model.state_dict()
        del sd["tenpai_classifier.weight"]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad_model.pth"
            torch.save(sd, path)

            with self.assertRaises(ValueError) as ctx:
                load_checkpoint_strictly(str(path))
            self.assertIn("STATE_DICT_MISSING_KEYS", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
