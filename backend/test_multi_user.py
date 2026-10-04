"""Ownership and scheduling regressions, exercising HTTP routes with real tiny models."""
import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import torch
from fastapi import FastAPI, HTTPException
from api import sidecar


class TinyModel(torch.nn.Module):
    seq_len = 2
    input_features = 3

    def __init__(self, value):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([value, -value], dtype=torch.float32))

    def forward(self, inputs):
        return inputs.mean(dim=(1, 2)).unsqueeze(1) + self.bias


def checkpoint_loader(**kwargs):
    name = kwargs['checkpoint_path']
    if name == 'broken':
        raise ValueError('Invalid checkpoint')
    return TinyModel(1.0 if name == 'Alice' else 2.0), SimpleNamespace(name=name), None


class PersonalHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = FastAPI()
        self.app.include_router(sidecar.sidecar_router)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')
        self.addAsyncCleanup(self.client.aclose)
        self.enterContext(patch.object(sidecar, 'load_checkpoint_strictly', side_effect=checkpoint_loader))
        if hasattr(sidecar, '_MODEL_STORE'):
            from model_engine.resources import ModelStore
            self.enterContext(patch.object(sidecar, '_MODEL_STORE', ModelStore(max_models=4)))
        self.enterContext(patch.object(sidecar, '_SESSION_STORE', sidecar.SessionStore()))

    async def load(self, name, handle=None):
        return await self.client.post('/internal/v1/models/load', json={'checkpoint_path': name},
                                      headers={'X-Model-Handle': handle} if handle else {})

    async def infer(self, handle=None, inputs=None):
        return await self.client.post('/internal/v1/sessions',
            json={'input_data': inputs if inputs is not None else [[0, 0, 0], [0, 0, 0]]},
            headers={'X-Model-Handle': handle} if handle else {})

    async def test_two_clients_keep_their_model_and_real_output(self):
        alice = (await self.load('Alice')).json()
        bob = (await self.load('Bob')).json()
        self.assertIn('model_handle', alice)
        self.assertNotEqual(alice['model_handle'], bob['model_handle'])
        for selection, expected in [(alice, [1.0, -1.0]), (bob, [2.0, -2.0]), (alice, [1.0, -1.0])]:
            result = await self.infer(selection['model_handle'])
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()['model_name'], selection['model_name'])
            self.assertEqual(result.json()['output_sample'], expected)
        self.assertEqual(alice['input_schema']['shape'], [1, 2, 3])
        self.assertEqual(alice['layer_count'], 0)

    async def test_missing_handle_never_uses_last_loaded_model(self):
        await self.load('Alice')
        response = await self.infer()
        self.assertEqual(response.status_code, 401, response.text)
        unknown = await self.infer('not-a-model')
        self.assertEqual(unknown.status_code, 404, unknown.text)

    async def test_another_model_cannot_read_or_delete_results(self):
        alice = (await self.load('Alice')).json()
        bob = (await self.load('Bob')).json()
        self.assertIn('model_handle', alice)
        session = (await self.infer(alice['model_handle'])).json()['session_id']
        url = f'/internal/v1/sessions/{session}'
        for method, suffix in [('GET', ''), ('GET', '/activations'), ('DELETE', '')]:
            denied = await self.client.request(method, url + suffix, headers={'X-Model-Handle': bob['model_handle']})
            self.assertEqual(denied.status_code, 404)
        own = await self.client.get(url, headers={'X-Model-Handle': alice['model_handle']})
        self.assertEqual(own.status_code, 200)

    async def test_failed_replacement_preserves_old_model_and_success_revokes_handle(self):
        alice = (await self.load('Alice')).json()
        self.assertIn('model_handle', alice)
        handle = alice['model_handle']
        old_session = (await self.infer(handle)).json()['session_id']
        self.assertEqual((await self.load('broken', handle)).status_code, 400)
        self.assertEqual((await self.infer(handle)).json()['output_sample'], [1.0, -1.0])
        replacement = (await self.load('Bob', handle)).json()
        self.assertNotEqual(replacement['model_handle'], handle)
        self.assertEqual((await self.infer(handle)).status_code, 404)
        self.assertEqual((await self.client.get(f'/internal/v1/sessions/{old_session}',
            headers={'X-Model-Handle': replacement['model_handle']})).status_code, 404)

    async def test_slow_load_does_not_block_event_loop(self):
        def slow_loader(**kwargs):
            time.sleep(0.3)
            return checkpoint_loader(**kwargs)
        with patch.object(sidecar, 'load_checkpoint_strictly', side_effect=slow_loader):
            task = asyncio.create_task(self.load('Alice'))
            started = time.monotonic()
            await asyncio.sleep(0.02)
            elapsed = time.monotonic() - started
            health = await self.client.get('/internal/v1/health')
            await task
        self.assertEqual(health.status_code, 200)
        self.assertLess(elapsed, 0.2, 'model loading blocked the ASGI event loop')

    async def test_release_revokes_model_and_cleans_results(self):
        handle = (await self.load('Alice')).json()['model_handle']
        await self.infer(handle)
        deleted = await self.client.delete('/internal/v1/models/current', headers={'X-Model-Handle': handle})
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual((await self.infer(handle)).status_code, 404)
        self.assertEqual(len(sidecar._SESSION_STORE.sessions), 0)

    async def test_overlapping_inference_is_serialized_and_health_remains_responsive(self):
        handle = (await self.load('Alice')).json()['model_handle']
        entered, release = threading.Event(), threading.Event()
        with sidecar._MODEL_STORE.lease(handle) as engine:
            original = engine.model.forward
            def slow_forward(inputs):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test did not release forward')
                return original(inputs)
            engine.model.forward = slow_forward
        first = asyncio.create_task(self.infer(handle))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            second = asyncio.create_task(self.infer(handle, [[1, 1, 1], [1, 1, 1]]))
            health = await asyncio.wait_for(self.client.get('/internal/v1/health'), 1)
            self.assertEqual(health.status_code, 200)
            deletion = await self.client.delete('/internal/v1/models/current', headers={'X-Model-Handle': handle})
            self.assertEqual(deletion.status_code, 409)
        finally:
            release.set()
        results = await asyncio.gather(first, second)
        self.assertEqual(results[0].json()['output_sample'], [1.0, -1.0])
        self.assertEqual(results[1].json()['output_sample'], [2.0, 0.0])


class ResourceTests(unittest.TestCase):
    def engine(self, value=1.0):
        from model_engine.transformer_model import TransformerModelEngine
        engine = TransformerModelEngine(isolated=True)
        engine._model = TinyModel(value)
        engine._is_loaded = True
        return engine

    def test_output_contract_preserves_all_values_and_shape(self):
        engine = self.engine()
        engine.model.bias = torch.nn.Parameter(torch.arange(7, dtype=torch.float32))
        result = engine.forward([[0, 0, 0], [0, 0, 0]])
        self.assertEqual(result['output_shape'], [1, 7])
        self.assertEqual(result['output_sample'], list(range(7)))
        self.assertEqual(result['output_kind'], 'raw')

    def test_output_contract_flattens_tensor_without_inventing_classes(self):
        engine = self.engine()
        engine.model.forward = lambda inputs: torch.arange(12).reshape(1, 2, 6)
        result = engine.forward([[0, 0, 0], [0, 0, 0]])
        self.assertEqual(result['output_shape'], [1, 2, 6])
        self.assertEqual(result['output_sample'], list(range(12)))

    def test_nonfinite_output_returns_error_instead_of_invalid_json(self):
        engine = self.engine()
        engine.model.forward = lambda inputs: torch.tensor([[float('nan')]])
        self.assertIn('error', engine.forward([[0, 0, 0], [0, 0, 0]]))

    def test_analysis_capability_does_not_claim_unimplemented_embedding_probe(self):
        from model_engine.compatibility import describe_analysis_capabilities
        capabilities = describe_analysis_capabilities('transTest')
        self.assertFalse(capabilities['has_embedding'])
        self.assertIn('embedding', capabilities['unavailable_probes'])

    def test_capacity_replacement_and_failure_are_atomic(self):
        from model_engine.resources import ModelStore
        store = ModelStore(max_models=1)
        first = self.engine()
        handle = store.load(lambda: first)
        with self.assertRaises(HTTPException) as full:
            store.load(self.engine)
        self.assertEqual(full.exception.status_code, 503)
        with self.assertRaises(ValueError):
            store.load(lambda: (_ for _ in ()).throw(ValueError('bad model')), handle)
        with store.lease(handle) as engine:
            self.assertIs(engine, first)
        new = store.load(lambda: self.engine(2), handle)
        with self.assertRaises(HTTPException):
            with store.lease(handle):
                pass
        with store.lease(new) as engine:
            self.assertEqual(engine.model.bias.tolist(), [2.0, -2.0])
        self.assertIsNone(first.model)

    def test_idle_expiry_does_not_dispose_active_lease(self):
        from model_engine.resources import ModelStore
        now = [0.0]
        store = ModelStore(ttl_seconds=10, clock=lambda: now[0])
        handle = store.load(self.engine)
        with store.lease(handle) as engine:
            now[0] = 20
            store.cleanup_expired()
            self.assertIsNotNone(engine.model)
            with self.assertRaises(HTTPException) as busy:
                store.delete(handle)
            self.assertEqual(busy.exception.status_code, 409)
        now[0] = 31
        store.cleanup_expired()
        self.assertIsNone(engine.model)
        with self.assertRaises(HTTPException):
            with store.lease(handle):
                pass

    def test_model_memory_limit_rejects_without_losing_previous(self):
        from model_engine.resources import ModelStore
        store = ModelStore(max_bytes=8)
        handle = store.load(self.engine)
        large = self.engine()
        large._model = torch.nn.Linear(10, 10)
        with self.assertRaises(HTTPException):
            store.load(lambda: large, handle)
        with store.lease(handle) as old:
            self.assertEqual(old.model.bias.tolist(), [1.0, -1.0])
        self.assertIsNone(large.model)

    def test_replacement_rejects_new_leases_until_published(self):
        from model_engine.resources import ModelStore
        store = ModelStore()
        handle = store.load(self.engine)
        def replacement():
            with self.assertRaises(HTTPException) as busy:
                with store.lease(handle):
                    pass
            self.assertEqual(busy.exception.status_code, 409)
            return self.engine(2)
        store.load(replacement, handle)

    def test_compute_overload_and_failure_release_slots(self):
        from model_engine.resources import ComputeGate
        gate = ComputeGate(max_waiting=0, wait_seconds=0.01)
        with gate.slot():
            with self.assertRaises(HTTPException) as busy:
                with gate.slot():
                    pass
            self.assertEqual(busy.exception.status_code, 503)
        with self.assertRaises(ValueError):
            with gate.slot():
                raise ValueError('failed job')
        with gate.slot():
            pass

    def test_compute_wait_timeout_does_not_leak_admission(self):
        from model_engine.resources import ComputeGate
        gate = ComputeGate(max_waiting=1, wait_seconds=0.01)
        with gate.slot():
            for _ in range(2):
                with self.assertRaises(HTTPException):
                    with gate.slot():
                        pass
        with gate.slot():
            pass

    def test_result_limits_do_not_evict_other_users(self):
        store = sidecar.SessionStore(max_sessions=1)
        handle = store.create('a', 'Alice', {'output_sample': [1]}, {}, [])
        with self.assertRaises(HTTPException):
            store.create('b', 'Bob', {'output_sample': [2]}, {}, [])
        self.assertIsNotNone(store.get(handle, 'a'))
        self.assertIsNone(store.get(handle, 'b'))
        self.assertFalse(store.delete(handle, 'b'))
        self.assertTrue(store.delete(handle, 'a'))
        small = sidecar.SessionStore(max_bytes=1)
        with self.assertRaises(HTTPException):
            small.create('a', 'Alice', {'output_sample': [1]}, {}, [])

    def test_cleanup_reclaims_results_of_expired_models(self):
        from model_engine.resources import ModelStore
        now = [0.0]
        models = ModelStore(ttl_seconds=10, clock=lambda: now[0])
        sessions = sidecar.SessionStore()
        handle = models.load(self.engine)
        sessions.create(handle, 'Alice', {}, {}, [])
        with patch.object(sidecar, '_MODEL_STORE', models), patch.object(sidecar, '_SESSION_STORE', sessions):
            now[0] = 11
            sidecar.cleanup_resources()
        self.assertEqual(len(sessions.sessions), 0)

    def test_clear_attention_releases_model_resident_tensors(self):
        engine = self.engine()
        engine.model.last_attention_weights = torch.ones(2, 2)
        engine.model._last_attention_weights = [torch.ones(2, 2)]
        engine.clear_attention_cache()
        self.assertIsNone(engine.model.last_attention_weights)
        self.assertIsNone(engine.model._last_attention_weights)


if __name__ == '__main__':
    unittest.main()
