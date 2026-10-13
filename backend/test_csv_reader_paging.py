import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from backend.api import csv_reader


class CsvReaderPagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.csv_path = self.root / "paging.csv"
        with self.csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["number", "note"])
            for number in range(2505):
                writer.writerow([number, "line one\nline two" if number == 999 else "中文" if number == 2000 else "ordinary"])
        self.index_patch = patch.object(csv_reader, "_INDEX_PATH", self.root / "index.sqlite3")
        self.index_patch.start()
        csv_reader._SESSIONS["paging-test"] = csv_reader.CsvSession(
            source_path=str(self.csv_path), csv_path=str(self.csv_path),
            display_path=str(self.csv_path), display_name="paging.csv",
            headers=["number", "note"],
        )

    def tearDown(self) -> None:
        csv_reader._SESSIONS.pop("paging-test", None)
        self.index_patch.stop()
        self.temp_dir.cleanup()

    def test_repeated_deep_page_reuses_saved_position_without_losing_csv_records(self) -> None:
        request = csv_reader.CsvPageRequest(session_id="paging-test", page=2, page_size=1000)
        first = csv_reader.get_page(request)
        csv_reader._SESSIONS["another-session"] = csv_reader._SESSIONS["paging-test"]
        try:
            second = csv_reader.get_page(csv_reader.CsvPageRequest(
                session_id="another-session", page=2, page_size=1000,
            ))
        finally:
            csv_reader._SESSIONS.pop("another-session", None)

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(first.rows[0], ["2000", "中文"])
        self.assertEqual(second.rows, first.rows)
        self.assertLess(second.scanned_rows, first.scanned_rows)
        self.assertLessEqual(second.scanned_rows, 1001)

    def test_changed_file_does_not_reuse_stale_position(self) -> None:
        request = csv_reader.CsvPageRequest(session_id="paging-test", page=2, page_size=1000)
        csv_reader.get_page(request)
        with self.csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["number", "note"])
            for number in range(2505):
                writer.writerow([number, "new content"])

        result = csv_reader.get_page(request)
        self.assertTrue(result.success)
        self.assertEqual(result.rows[0], ["2000", "new content"])
        self.assertGreater(result.scanned_rows, 2000)

    def test_csv_remains_readable_when_optional_index_is_unavailable(self) -> None:
        with patch.object(csv_reader, "_connect_index", side_effect=OSError("cache unavailable")):
            result = csv_reader.get_page(csv_reader.CsvPageRequest(
                session_id="paging-test", page=1, page_size=1000,
            ))
        self.assertTrue(result.success)
        self.assertEqual(result.rows[0], ["1000", "ordinary"])

    def test_zip_reopen_reuses_positions_from_the_same_archive(self) -> None:
        zip_path = self.root / "predictions.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(self.csv_path, "predictions.csv")
        first_open = csv_reader.open_csv(csv_reader.OpenCsvRequest(path=str(zip_path)))
        self.assertTrue(first_open.success)
        first = csv_reader.get_page(csv_reader.CsvPageRequest(
            session_id=first_open.session_id, page=2, page_size=1000,
        ))
        csv_reader.close_session(csv_reader.CloseSessionRequest(session_id=first_open.session_id))

        second_open = csv_reader.open_csv(csv_reader.OpenCsvRequest(path=str(zip_path)))
        self.assertTrue(second_open.success)
        try:
            second = csv_reader.get_page(csv_reader.CsvPageRequest(
                session_id=second_open.session_id, page=2, page_size=1000,
            ))
            self.assertEqual(second.rows, first.rows)
            self.assertLess(second.scanned_rows, first.scanned_rows)
            self.assertLessEqual(second.scanned_rows, 1001)
        finally:
            csv_reader.close_session(csv_reader.CloseSessionRequest(session_id=second_open.session_id))


if __name__ == "__main__":
    unittest.main()
