import asyncio
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from backend.api import csv_reader as reader


class CsvLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.enterContext(patch.object(reader, '_SESSIONS', {}))
        self.enterContext(patch.object(reader, '_SESSION_TEMP_ROOT', self.root / 'extracted'))
        self.enterContext(patch.object(reader, '_INDEX_PATH', self.root / 'index.sqlite3'))
        self.path = self.root / 'sample.zip'
        with zipfile.ZipFile(self.path, 'w') as z:
            z.writestr('data.csv', 'line_number,ids,tenhou_link\n1,x,url1\n2,x,url2\n')
            z.writestr('plot.png', b'fake-png-for-stream-lifetime')

    def open(self):
        result = reader.open_csv(reader.OpenCsvRequest(path=str(self.path)))
        self.assertTrue(result.success, result.error)
        return result.session_id

    def test_idle_expiry_reclaims_zip_and_session(self):
        sid = self.open()
        root = Path(reader._SESSIONS[sid].extracted_root)
        with patch.object(reader.time, 'monotonic', return_value=10**12):
            reader.cleanup_csv_sessions()
        self.assertNotIn(sid, reader._SESSIONS)
        self.assertFalse(root.exists())
        self.assertFalse(reader.get_page(reader.CsvPageRequest(session_id=sid)).success)

    def test_capacity_does_not_evict_valid_session_and_releases_on_close(self):
        with patch.object(reader, '_MAX_SESSIONS', 1):
            sid = self.open()
            second = reader.open_csv(reader.OpenCsvRequest(path=str(self.path)))
            self.assertFalse(second.success)
            self.assertEqual(second.error_code, 'CSV_CAPACITY')
            self.assertTrue(reader.get_page(reader.CsvPageRequest(session_id=sid)).success)
            reader.close_session(reader.CloseSessionRequest(session_id=sid))
            self.open()

    def test_close_and_expiry_do_not_delete_zip_during_page_read(self):
        sid = self.open()
        root = Path(reader._SESSIONS[sid].extracted_root)
        entered, release = threading.Event(), threading.Event()
        original = reader._parse_filter
        def blocked(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('reader not released')
            return original(*args)
        results = []
        with patch.object(reader, '_parse_filter', side_effect=blocked):
            worker = threading.Thread(target=lambda: results.append(reader.get_page(reader.CsvPageRequest(session_id=sid))))
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                with patch.object(reader.time, 'monotonic', return_value=10**12):
                    reader.cleanup_csv_sessions()
                self.assertTrue(root.exists())
                reader.close_session(reader.CloseSessionRequest(session_id=sid))
                self.assertTrue(root.exists())
                self.assertFalse(reader.get_page(reader.CsvPageRequest(session_id=sid)).success)
            finally:
                release.set()
                worker.join(4)
        self.assertTrue(results[0].success, results[0].error)
        self.assertFalse(root.exists())
        self.assertNotIn(sid, reader._SESSIONS)

    def test_failed_zip_open_leaves_no_temp_directory_or_reservation(self):
        for members in ({'picture.png': b'bad'}, {'data.csv': b''}, {'data.csv': b'a\n1', '../bad.png': b'bad'}):
            with zipfile.ZipFile(self.path, 'w') as z:
                for name, value in members.items():
                    z.writestr(name, value)
            self.assertFalse(reader.open_csv(reader.OpenCsvRequest(path=str(self.path))).success)
            self.assertEqual(list(reader._SESSION_TEMP_ROOT.iterdir()), [])
            self.assertEqual(reader._OPENING, 0)

    def test_open_reservation_counts_toward_capacity(self):
        entered, release = threading.Event(), threading.Event()
        original = reader._read_headers
        def blocked(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('open not released')
            return original(*args)
        results = []
        with patch.object(reader, '_MAX_SESSIONS', 1), patch.object(reader, '_read_headers', side_effect=blocked):
            worker = threading.Thread(target=lambda: results.append(reader.open_csv(reader.OpenCsvRequest(path=str(self.path)))))
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                response = reader.open_csv(reader.OpenCsvRequest(path=str(self.path)))
                self.assertFalse(response.success)
                self.assertEqual(response.error_code, 'CSV_CAPACITY')
            finally:
                release.set()
                worker.join(4)
        self.assertTrue(results[0].success)

    def test_image_lease_lasts_until_response_finishes_even_when_closed(self):
        sid = self.open()
        root = Path(reader._SESSIONS[sid].extracted_root)
        async def run():
            response = await reader.get_image(sid, 0)
            reader.close_session(reader.CloseSessionRequest(session_id=sid))
            self.assertTrue(root.exists())
            chunks = []
            async def send(message):
                if message['type'] == 'http.response.body':
                    chunks.append(message.get('body', b''))
            await response({'type': 'http', 'method': 'GET', 'headers': []}, None, send)
            self.assertEqual(b''.join(chunks), b'fake-png-for-stream-lifetime')
        asyncio.run(run())
        self.assertFalse(root.exists())

    def test_image_send_failure_releases_lease(self):
        sid = self.open()
        root = Path(reader._SESSIONS[sid].extracted_root)
        async def run():
            response = await reader.get_image(sid, 0)
            reader.close_session(reader.CloseSessionRequest(session_id=sid))
            async def send(message):
                raise RuntimeError('disconnected')
            with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                await response({'type': 'http', 'method': 'GET', 'headers': []}, None, send)
        asyncio.run(run())
        self.assertFalse(root.exists())

    def test_image_cleanup_does_not_block_event_loop(self):
        sid = self.open()
        original_dispose = reader._dispose_csv
        async def run():
            loop = asyncio.get_running_loop()
            started = asyncio.Event()
            timestamp = []
            def slow_dispose(session):
                timestamp.append(time.monotonic())
                loop.call_soon_threadsafe(started.set)
                time.sleep(0.3)
                original_dispose(session)
            response = await reader.get_image(sid, 0)
            reader.close_session(reader.CloseSessionRequest(session_id=sid))
            async def send(message):
                pass
            with patch.object(reader, '_dispose_csv', side_effect=slow_dispose):
                task = asyncio.create_task(response({'type': 'http', 'method': 'GET', 'headers': []}, None, send))
                try:
                    await asyncio.wait_for(started.wait(), 2)
                    self.assertLess(time.monotonic() - timestamp[0], 0.2)
                finally:
                    await task
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
