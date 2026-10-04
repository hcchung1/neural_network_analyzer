"""Performance contracts exercised with real small CPU models and on-disk fixtures."""
import asyncio
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import torch
from fastapi import FastAPI, HTTPException
from model_engine import resources
from model_engine.transformer_model import TransformerModelEngine
from api import collaboration, training, binary_samples
from api.sidecar import LoadModelRequest


def engine(value=1):
    result = TransformerModelEngine(isolated=True)
    result._model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        result._model.weight.fill_(value)
    result._is_loaded = True
    return result


class CacheTests(unittest.TestCase):
    def cache(self, **kwargs):
        cls = getattr(resources, 'SharedModelCache', None)
        self.assertIsNotNone(cls, 'Shared model reuse is missing')
        cache = cls(**kwargs)
        self.addCleanup(cache.close)
        return cache

    def test_identity_metadata_reuse_and_checkpoint_mutation(self):
        cache = self.cache()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pt'
            path.write_bytes(b'a')
            a = LoadModelRequest(checkpoint_path=str(path), manifest_spec={'a': 1, 'b': 2})
            built = engine()
            with cache.lease(a, lambda: (built, {'schema': {'shape': [1, 2]}})) as first:
                self.assertIs(first.engine, built)
                with self.assertRaises(TypeError):
                    first.metadata['schema']['shape'] = []
            b = LoadModelRequest(checkpoint_path=str(path.parent / '.' / path.name), manifest_spec={'b': 2, 'a': 1})
            with cache.lease(b, lambda: self.fail('warm checkpoint rebuilt')) as second:
                self.assertIs(second.engine, built)
            path.write_bytes(b'changed')
            fresh = engine(2)
            with cache.lease(a, lambda: (fresh, {})) as third:
                self.assertIs(third.engine, fresh)
            with cache.lease(LoadModelRequest(checkpoint_path=str(path), requested_phase='late'), lambda: (engine(3), {})) as fourth:
                self.assertIsNot(fourth.engine, fresh)

    def test_leases_failed_build_and_capacity_preserve_valid_selection(self):
        cache = self.cache(max_models=1, max_bytes=16)
        selection = LoadModelRequest(checkpoint_path='a')
        old = engine()
        with cache.lease(selection, lambda: (old, {})):
            with self.assertRaises(HTTPException) as caught:
                with cache.lease(LoadModelRequest(checkpoint_path='b'), lambda: (engine(), {})):
                    pass
            self.assertEqual(caught.exception.status_code, 503)
            self.assertIsNotNone(old.model)
        with self.assertRaises(ValueError):
            with cache.lease(LoadModelRequest(checkpoint_path='broken'), lambda: (_ for _ in ()).throw(ValueError('bad'))):
                pass
        with cache.lease(selection, lambda: self.fail('failure removed valid entry')):
            pass
        oversized = TransformerModelEngine(isolated=True)
        oversized._model = torch.nn.Linear(3, 3)
        with self.assertRaises(HTTPException):
            with cache.lease(LoadModelRequest(checkpoint_path='big'), lambda: (oversized, {})):
                pass
        self.assertIsNone(oversized.model)
        with cache.lease(selection, lambda: self.fail('capacity rejection removed valid entry')):
            pass

    def test_lru_expiry_and_shutdown_release_attention(self):
        now = [0.0]
        cache = self.cache(max_models=2, ttl_seconds=5, clock=lambda: now[0])
        a, b, c = engine(), engine(), engine()
        for name, model in [('a', a), ('b', b)]:
            with cache.lease(LoadModelRequest(checkpoint_path=name), lambda model=model: (model, {})):
                pass
            now[0] += 1
        with cache.lease(LoadModelRequest(checkpoint_path='a'), lambda: self.fail('rebuild')):
            pass
        now[0] += 1
        with cache.lease(LoadModelRequest(checkpoint_path='c'), lambda: (c, {})):
            pass
        self.assertIsNone(b.model)
        with cache.lease(LoadModelRequest(checkpoint_path='a'), lambda: self.fail('rebuild')):
            now[0] = 20
            cache.cleanup_expired()
            self.assertIsNotNone(a.model)
        cache.cleanup_expired()
        self.assertIsNone(c.model)
        cache.close()
        self.assertIsNone(a.model)

    def test_shared_load_and_inference_reuse_engine_and_clear_attention(self):
        cache = self.cache()
        models = []
        class MultiHeadAttention(torch.nn.Module):
            def forward(self, x):
                return x, torch.tensor([[[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]]])
        class Tiny(torch.nn.Module):
            seq_len = 2
            input_features = 2
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2))
                self.attention = MultiHeadAttention()
            def forward(self, x):
                x, weights = self.attention(x)
                self.last_attention_weights = [weights]
                return x.sum((1, 2)).unsqueeze(1) * self.weight
        def loader(**kwargs):
            model = Tiny()
            models.append(model)
            return model, SimpleNamespace(name='test'), None
        with patch.object(collaboration, '_SHARED_MODEL_CACHE', cache, create=True), patch.object(collaboration, 'load_checkpoint_strictly', side_effect=loader):
            model = LoadModelRequest(checkpoint_path='test')
            collaboration.analyze(collaboration.AnalysisRequest(model=model))
            for _ in range(2):
                result = collaboration.analyze(collaboration.AnalysisRequest(model=model, input_data=[[1, 2], [3, 4]]))
                self.assertEqual(result['output'], [10.0, 10.0])
                self.assertEqual(result['attention'], {'0': [[2., 3.], [4., 5.]]})
            with self.assertRaises(HTTPException):
                collaboration.analyze(collaboration.AnalysisRequest(model=model, input_data=[[float('nan'), 2], [3, 4]]))
            self.assertIsNone(models[0].last_attention_weights)
            with cache.lease(model, lambda: self.fail('failed inference removed warm selection')) as entry:
                self.assertEqual(entry.engine._captured_attention, [])
                self.assertEqual(entry.engine._attention_hooks, [])
            self.assertEqual(len(models), 1)
            self.assertIsNone(models[0].last_attention_weights)


    def test_failed_selection_preserves_every_valid_cached_engine(self):
        cache = self.cache(max_models=2, max_bytes=32)
        first, second = engine(1), engine(2)
        for name, item in [('a', first), ('b', second)]:
            with cache.lease(LoadModelRequest(checkpoint_path=name), lambda item=item: (item, {})):
                pass
        with self.assertRaises(ValueError):
            with cache.lease(LoadModelRequest(checkpoint_path='broken'), lambda: (_ for _ in ()).throw(ValueError('invalid'))):
                pass
        oversized = TransformerModelEngine(isolated=True)
        oversized._model = torch.nn.Linear(4, 4)
        with self.assertRaises(HTTPException):
            with cache.lease(LoadModelRequest(checkpoint_path='big'), lambda: (oversized, {})):
                pass
        for name, expected in [('a', first), ('b', second)]:
            with cache.lease(LoadModelRequest(checkpoint_path=name), lambda: self.fail('valid selection was removed')) as entry:
                self.assertIs(entry.engine, expected)
                self.assertIsNotNone(entry.engine.model)


class GateAndAttentionTests(unittest.TestCase):
    def test_fifo_waiter_order_and_exception_release(self):
        gate = resources.ComputeGate(max_waiting=3, wait_seconds=1)
        order, threads = [], []
        def work(index):
            with gate.slot():
                order.append(index)
        with gate.slot():
            for i in range(3):
                thread = threading.Thread(target=work, args=(i,))
                threads.append(thread)
                thread.start()
                deadline = time.monotonic() + 1
                while len(getattr(gate, '_waiters', [])) < i + 1 and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.assertEqual(len(getattr(gate, '_waiters', [])), i + 1, 'FIFO admission queue is missing')
            with self.assertRaises(HTTPException):
                with gate.slot():
                    pass
        for thread in threads:
            thread.join(2)
        self.assertEqual(order, [0, 1, 2])
        with self.assertRaises(ValueError):
            with gate.slot():
                raise ValueError('job failed')
        with gate.slot():
            pass

    def test_timeout_removes_waiter_and_next_job_can_run(self):
        gate = resources.ComputeGate(max_waiting=1, wait_seconds=0.03)
        errors = []
        def wait():
            try:
                with gate.slot():
                    pass
            except HTTPException as error:
                errors.append(error)
        with gate.slot():
            worker = threading.Thread(target=wait)
            worker.start()
            worker.join(1)
            self.assertEqual(errors[0].detail['error']['code'], 'COMPUTE_BUSY')
        with gate.slot():
            pass

    def test_attention_tensor_is_detached_and_compatibility_values_match(self):
        item = engine()
        item._captured_attention = [torch.arange(8., requires_grad=True).reshape(1, 2, 2, 2)]
        self.assertTrue(hasattr(item, 'get_attention_tensor'), 'Tensor attention accessor is missing')
        weights = item.get_attention_tensor(0)
        self.assertEqual(weights.device.type, 'cpu')
        self.assertFalse(weights.requires_grad)
        self.assertEqual(weights.mean((0, 1)).tolist(), [[2., 3.], [4., 5.]])
        self.assertEqual(item.get_attention_weights(0)['attention'], weights.tolist())


class BinaryIndexTests(unittest.TestCase):
    def test_direct_read_is_bounded_and_line_index_persists_invalidates(self):
        from importlib.util import find_spec
        self.assertIsNotNone(Path('backend/data/binary_index.py').is_file() or None, 'Persistent binary index is missing')
        from backend.data.binary_index import BinaryRecordIndex
        with tempfile.TemporaryDirectory() as folder:
            path, db = Path(folder) / 'records.bin', Path(folder) / 'index.sqlite3'
            records = [n.to_bytes(4, 'little') + bytes([n]) * 12 for n in [9, 7, 3]]
            path.write_bytes(b''.join(records))
            index = BinaryRecordIndex(db)
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('unbounded whole file read')):
                read_sizes = []
                original_open = Path.open
                class TrackedStream:
                    def __init__(self, stream):
                        self.stream = stream
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        self.stream.close()
                    def seek(self, offset):
                        return self.stream.seek(offset)
                    def read(self, size=-1):
                        read_sizes.append(size)
                        return self.stream.read(size)
                with patch.object(Path, 'open', lambda path, *args, **kwargs: TrackedStream(original_open(path, *args, **kwargs))):
                    self.assertEqual(index.read_record(path, 1, 16), records[1])
                self.assertEqual(read_sizes, [16])
                self.assertEqual(index.find_line(path, 7, 16), (1, records[1]))
                with patch('backend.data.binary_index.mmap.mmap', side_effect=AssertionError('warm lookup rescanned file')):
                    self.assertEqual(BinaryRecordIndex(db).find_line(path, 3, 16), (2, records[2]))
                records[1] = (5).to_bytes(4, 'little') + b'x' * 12
                path.write_bytes(b''.join(records))
                self.assertIsNone(index.find_line(path, 7, 16))
                self.assertEqual(index.find_line(path, 5, 16), (1, records[1]))
                path.write_bytes(b'bad')
                with self.assertRaises(ValueError):
                    index.find_line(path, 5, 16)

    def test_corrupt_database_is_rebuilt_without_changing_sample(self):
        from importlib.util import find_spec
        self.assertIsNotNone(Path('backend/data/binary_index.py').is_file() or None, 'Persistent binary index is missing')
        from backend.data.binary_index import BinaryRecordIndex
        with tempfile.TemporaryDirectory() as folder:
            path, db = Path(folder) / 'records.bin', Path(folder) / 'index.sqlite3'
            path.write_bytes((6).to_bytes(4, 'little') + b'x' * 12)
            db.write_bytes(b'broken sqlite')
            self.assertEqual(BinaryRecordIndex(db).find_line(path, 6, 16), (0, (6).to_bytes(4, 'little') + b'x' * 12))


    def test_truncation_after_mapping_never_crashes_index_process(self):
        import subprocess
        import sys
        script = r"""
import mmap
import resource
import tempfile
from pathlib import Path
from unittest.mock import patch
from backend.data.binary_index import BinaryRecordIndex
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
with tempfile.TemporaryDirectory() as folder:
    path, db = Path(folder) / 'records.bin', Path(folder) / 'index.sqlite3'
    path.write_bytes((6).to_bytes(4, 'little') + b'x' * 12)
    original_mmap = mmap.mmap
    class TruncatingMapping:
        def __init__(self, *args, **kwargs):
            self.mapping = original_mmap(*args, **kwargs)
        def __enter__(self):
            path.write_bytes(b'')
            return self.mapping
        def __exit__(self, *args):
            self.mapping.close()
    with patch('backend.data.binary_index.mmap.mmap', TruncatingMapping):
        try:
            BinaryRecordIndex(db).find_line(path, 6, 16)
        except ValueError:
            pass
        else:
            raise SystemExit('changed binary file was accepted')
"""
        child = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, f'Index process crashed or accepted mutation: {child.returncode} {child.stderr}')


    def test_mutation_after_fallback_rebuild_never_returns_wrong_binary_line(self):
        import sqlite3
        from backend.data.binary_index import BinaryRecordIndex
        with tempfile.TemporaryDirectory() as folder:
            path, db = Path(folder) / 'records.bin', Path(folder) / 'index.sqlite3'
            first = (9).to_bytes(4, 'little') + b'a' * 12
            second = (7).to_bytes(4, 'little') + b'b' * 12
            path.write_bytes(first + second)
            index = BinaryRecordIndex(db)
            self.assertEqual(index.find_line(path, 7, 16), (1, second))
            with sqlite3.connect(db) as connection:
                connection.execute('UPDATE records SET line_number=7 WHERE record_index=0')
            original_build = index._build
            def rebuild_then_mutate(*args):
                original_build(*args)
                path.write_bytes(first + (9).to_bytes(4, 'little') + b'c' * 12)
            with patch.object(index, '_build', side_effect=rebuild_then_mutate):
                with self.assertRaises(ValueError):
                    index.find_line(path, 7, 16)


class InfoIndexTests(unittest.TestCase):
    def test_high_info_line_reuses_persistent_offsets_and_bounded_read(self):
        from backend.data.binary_index import BinaryRecordIndex
        with tempfile.TemporaryDirectory() as folder:
            path, db = Path(folder) / 'info.txt', Path(folder) / 'index.sqlite3'
            raw = '  {"name":"東"}  \r\n'.encode('utf-8')
            path.write_bytes(b'{}\n' * 2000 + raw + b'last without newline')
            with patch.object(binary_samples, '_BINARY_INDEX', BinaryRecordIndex(db)):
                self.assertEqual(binary_samples._read_info_line(path, 2000), '{"name":"東"}')
            read_sizes, seek_offsets = [], []
            original_open = Path.open
            class BoundedStream:
                def __init__(self, stream):
                    self.stream = stream
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    self.stream.close()
                def __iter__(self):
                    raise AssertionError('warm info lookup rescanned preceding lines')
                def readline(self):
                    raise AssertionError('warm info lookup rescanned preceding lines')
                def seek(self, offset):
                    seek_offsets.append(offset)
                    return self.stream.seek(offset)
                def read(self, size=-1):
                    if size < 0:
                        raise AssertionError('unbounded info file read')
                    read_sizes.append(size)
                    return self.stream.read(size)
            with patch.object(binary_samples, '_BINARY_INDEX', BinaryRecordIndex(db)), patch.object(Path, 'open', lambda path, *args, **kwargs: BoundedStream(original_open(path, *args, **kwargs))):
                content = binary_samples._read_info_line(path, 2000)
                self.assertEqual(content, '{"name":"東"}')
                self.assertEqual(binary_samples._tenhou_url(content), 'https://tenhou.net/6/#json=%7B%22name%22%3A%22%E6%9D%B1%22%7D&ts=0')
            self.assertEqual(seek_offsets, [6000])
            self.assertEqual(read_sizes, [len(raw)])

    def test_info_offset_invalidation_missing_and_unterminated_lines(self):
        from backend.data.binary_index import BinaryRecordIndex
        self.assertTrue(hasattr(BinaryRecordIndex, 'read_line'), 'Text offset index is missing')
        with tempfile.TemporaryDirectory() as folder:
            path, db = Path(folder) / 'info.txt', Path(folder) / 'index.sqlite3'
            index = BinaryRecordIndex(db)
            self.assertEqual(index.read_line(path, 0), b'')
            path.write_bytes(b'first\nnext\n')
            self.assertEqual(index.read_line(path, 1), b'next\n')
            previous = path.stat()
            path.write_bytes(b'prior\nlast\n')
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1000))
            self.assertEqual(index.read_line(path, 1), b'last\n')
            path.write_bytes(b'a\nb\nc\nfinal')
            self.assertEqual(index.read_line(path, 3), b'final')
            self.assertEqual(index.read_line(path, 4), b'')
            self.assertEqual(index.read_line(path, -1), b'')
            path.write_bytes(b'')
            self.assertEqual(index.read_line(path, 0), b'')
            path.unlink()
            self.assertEqual(index.read_line(path, 0), b'')

    def test_info_scan_streams_and_mutation_never_publishes_partial_index(self):
        import sqlite3
        from backend.data.binary_index import BinaryRecordIndex
        self.assertTrue(hasattr(BinaryRecordIndex, 'read_line'), 'Text offset index is missing')
        with tempfile.TemporaryDirectory() as folder:
            path, db = Path(folder) / 'info.txt', Path(folder) / 'index.sqlite3'
            path.write_bytes(b'one\ntwo\n')
            index = BinaryRecordIndex(db)
            original_open = Path.open
            class ChangingStream:
                def __init__(self, stream):
                    self.stream = stream
                    self.changed = False
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    self.stream.close()
                def __getattr__(self, name):
                    return getattr(self.stream, name)
                def read(self, size=-1):
                    raise AssertionError('initial info index used a whole-file read')
                def readline(self):
                    result = self.stream.readline()
                    if not self.changed:
                        self.changed = True
                        with original_open(path, 'wb') as changed:
                            changed.write(b'new\nreplacement\n')
                    return result
            with patch.object(Path, 'open', lambda path, *args, **kwargs: ChangingStream(original_open(path, *args, **kwargs))):
                with self.assertRaises(ValueError):
                    index.read_line(path, 1)
            with sqlite3.connect(db) as connection:
                self.assertEqual(connection.execute('SELECT count(*) FROM text_lines').fetchone()[0], 0)
            self.assertEqual(index.read_line(path, 1), b'replacement\n')


    def test_corrupt_text_index_recovers_and_preserves_replacement_decoding(self):
        from backend.data.binary_index import BinaryRecordIndex
        with tempfile.TemporaryDirectory() as folder:
            path, db = Path(folder) / 'info.txt', Path(folder) / 'index.sqlite3'
            path.write_bytes(b'first\n  bad \xff value  \n')
            db.write_bytes(b'corrupt database')
            with patch.object(binary_samples, '_BINARY_INDEX', BinaryRecordIndex(db)):
                self.assertEqual(binary_samples._read_info_line(path, 1), 'bad \ufffd value')


    def test_logically_corrupt_text_offset_rebuilds_instead_of_returning_wrong_line(self):
        import sqlite3
        from backend.data.binary_index import BinaryRecordIndex
        with tempfile.TemporaryDirectory() as folder:
            path, db = Path(folder) / 'info.txt', Path(folder) / 'index.sqlite3'
            path.write_bytes(b'first\nother\n')
            index = BinaryRecordIndex(db)
            self.assertEqual(index.read_line(path, 1), b'other\n')
            with sqlite3.connect(db) as connection:
                connection.execute('UPDATE text_lines SET byte_offset=0 WHERE line_number=1')
            self.assertEqual(index.read_line(path, 1), b'other\n')


class FilesystemHeartbeatTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_training_and_binary_files_do_not_block_health(self):
        app = FastAPI()
        app.include_router(training.router, prefix='/training')
        app.include_router(binary_samples.router, prefix='/binary')
        @app.get('/health')
        async def health():
            return {'ok': True}
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test')
        self.addAsyncCleanup(client.aclose)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.csv'
            path.write_text('value\n1\n')
            for endpoint, body, context_factory in [
                ('/training/read_file', {'file_path': str(path)}, lambda block: patch.object(training, 'open', side_effect=block, create=True)),
                ('/binary/files?directory=blocked', None, lambda block: patch.object(binary_samples, '_safe_directory', side_effect=block))]:
                entered, release = threading.Event(), threading.Event()
                def block(*args, **kwargs):
                    entered.set()
                    release.wait(0.3)
                    raise OSError('controlled blocked filesystem')
                with context_factory(block):
                    job = asyncio.create_task(client.post(endpoint, json=body) if body else client.get(endpoint))
                    deadline = time.monotonic() + 2
                    while not entered.is_set() and not job.done() and time.monotonic() < deadline:
                        await asyncio.sleep(0.005)
                    try:
                        self.assertTrue(entered.is_set())
                        self.assertFalse(job.done(), 'filesystem request ran on the event loop')
                        result = await asyncio.wait_for(client.get('/health'), timeout=0.1)
                        self.assertEqual(result.json(), {'ok': True})
                    finally:
                        release.set()
                        deadline = time.monotonic() + 2
                        while not job.done() and time.monotonic() < deadline:
                            await asyncio.sleep(0.005)
                        await asyncio.wait_for(job, timeout=0.1)



if __name__ == '__main__':
    unittest.main()
