import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import torch
from fastapi import FastAPI
from api import sidecar, collaboration, visualize_chat
from model_engine.resources import ModelStore
from backend.test_multi_user import TinyModel


class AnalysisContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = FastAPI()
        app.include_router(sidecar.sidecar_router)
        app.include_router(collaboration.router)
        app.include_router(visualize_chat.router, prefix='/assistant')
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test')
        self.addAsyncCleanup(self.client.aclose)
        self.enterContext(patch.object(sidecar, '_MODEL_STORE', ModelStore()))
        self.enterContext(patch.object(sidecar, '_SESSION_STORE', sidecar.SessionStore()))
        def loader(**kwargs):
            model = TinyModel(1)
            model.bias = torch.nn.Parameter(torch.arange(7, dtype=torch.float32))
            return model, SimpleNamespace(name='transTest'), None
        self.enterContext(patch.object(sidecar, 'load_checkpoint_strictly', side_effect=loader))
        self.enterContext(patch.object(collaboration, 'load_checkpoint_strictly', side_effect=loader))

    async def test_personal_and_shared_outputs_have_complete_raw_contract(self):
        selection = {'checkpoint_path': 'fixture', 'device': 'cpu'}
        load = await self.client.post('/internal/v1/models/load', json=selection)
        self.assertEqual(load.status_code, 200, load.text)
        metadata = load.json()
        self.assertFalse(metadata['capabilities']['has_embedding'])
        headers = {'X-Model-Handle': metadata['model_handle']}
        personal = await self.client.post('/internal/v1/sessions', headers=headers,
            json={'input_data': [[0, 0, 0], [0, 0, 0]]})
        self.assertEqual(personal.status_code, 200, personal.text)
        result = personal.json()
        self.assertEqual(result['output_sample'], list(range(7)))
        self.assertEqual(result['output_shape'], [1, 7])
        self.assertEqual(result['output_kind'], 'raw')
        read = await self.client.get('/internal/v1/sessions/' + result['session_id'], headers=headers)
        self.assertEqual(read.json(), result)
        shared = await self.client.post('/internal/v1/collaboration/analyze', json={
            'model': selection, 'input_data': [[0, 0, 0], [0, 0, 0]]})
        self.assertEqual(shared.status_code, 200, shared.text)
        self.assertEqual(shared.json()['output'], list(range(7)))
        self.assertEqual(shared.json()['output_shape'], [1, 7])
        self.assertEqual(shared.json()['output_kind'], 'raw')
        self.assertFalse(shared.json()['capabilities']['has_embedding'])

    async def test_assistant_accepts_explore_context_and_returns_reply(self):
        body = {'messages': [{'role': 'user', 'content': 'Explain this model'}], 'page_context': {
            'model_path': 'transTest', 'selected_token': 0, 'current_layer': 1,
            'model_input_schema': {'dtype': 'float32', 'rank': 3, 'batch_size': 1,
                'seq_len': 4, 'feature_dim': 100, 'shape': [1, 4, 100]}}}
        # Only the external language model provider is replaced.
        with patch.object(visualize_chat, '_run_free_chat', return_value='Model explanation') as provider:
            response = await self.client.post('/assistant/chat', json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {'success': True, 'kind': 'chat', 'reply': 'Model explanation'})
        self.assertIn('transTest', provider.call_args.args[1])
        self.assertIn('100', provider.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
