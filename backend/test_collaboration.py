"""Regression checks for isolated collaboration engines and attention capture."""
import unittest
from unittest.mock import patch
import torch
from api.collaboration import analyze, AnalysisRequest
from api.sidecar import LoadModelRequest
from model_engine.transformer_model import TransformerModelEngine, get_model_engine


class CollaborationTests(unittest.TestCase):
    def test_isolated_engines_do_not_replace_singleton(self):
        shared = get_model_engine()
        first = TransformerModelEngine(isolated=True)
        second = TransformerModelEngine(isolated=True)
        self.assertIsNot(first, second)
        self.assertIsNot(first, shared)
        self.assertIs(get_model_engine(), shared)
        first._captured_attention.append("first")
        self.assertEqual(second._captured_attention, [])

    def test_native_attention_hooks_survive_forward(self):
        engine = TransformerModelEngine(isolated=True)
        engine._model = engine._create_dummy_model()
        engine._model.eval()
        engine._is_loaded = True
        result = engine.forward(torch.zeros(1, 128, 256))
        self.assertNotIn("error", result)
        self.assertEqual(len(engine._captured_attention), 4)
        self.assertIn("attention", engine.get_attention_weights(0))
        engine.clear_attention_cache()

    def test_failed_job_releases_compute_lock(self):
        request = AnalysisRequest(model=LoadModelRequest(checkpoint_path="missing-test-checkpoint"))
        with patch("api.collaboration.load_checkpoint_strictly", side_effect=ValueError("invalid")):
            from fastapi import HTTPException
            from model_engine.resources import compute_gate
            with self.assertRaises(HTTPException):
                analyze(request)
            with compute_gate.slot():
                pass

    def test_collaboration_and_compare_share_bounded_admission(self):
        from fastapi import HTTPException
        from model_engine.resources import ComputeGate
        from api.compare import compare_models, CompareRequest
        gate = ComputeGate(max_waiting=0)
        with patch('api.collaboration.compute_gate', gate), patch('api.compare.compute_gate', gate), gate.slot():
            for operation in (
                lambda: analyze(AnalysisRequest(model=LoadModelRequest(checkpoint_path='unused'))),
                lambda: compare_models(CompareRequest(checkpoint_a='unused', checkpoint_b='unused')),
            ):
                with self.assertRaises(HTTPException) as caught:
                    operation()
                self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
