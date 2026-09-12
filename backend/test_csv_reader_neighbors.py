import csv
import tempfile
import unittest
from pathlib import Path

from backend.api.csv_reader import (
    CsvNeighborsRequest,
    CsvSession,
    _SESSIONS,
    get_neighbors,
)


class CsvReaderNeighborTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "predictions.csv"
        with self.csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["probability", "tenhou_link", "line_number", "ids"])
            writer.writerows(
                [
                    ["0.1", "https://example.test/first", "101", "3_2_2"],
                    ["0.2", "https://example.test/other-a", "102", "0_0_1"],
                    ["0.3", "https://example.test/current", "103", "3_2_2"],
                    ["0.4", "https://example.test/other-b", "104", "1_0_2"],
                    ["0.5", "https://example.test/next", "105", "3_2_2"],
                    ["0.6", "   ", "106", "3_2_2"],
                ]
            )

        self.session_id = "neighbor-test"
        _SESSIONS[self.session_id] = CsvSession(
            source_path=str(self.csv_path),
            csv_path=str(self.csv_path),
            display_path=str(self.csv_path),
            display_name=self.csv_path.name,
            headers=["probability", "tenhou_link", "line_number", "ids"],
        )

    def tearDown(self) -> None:
        _SESSIONS.pop(self.session_id, None)
        self.temp_dir.cleanup()

    def test_finds_same_prediction_pair_before_and_after_current_row(self) -> None:
        result = get_neighbors(
            CsvNeighborsRequest(
                session_id=self.session_id,
                line_number="103",
                ids="3_2_2",
                page_size=2,
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.previous.source_row_number, 1)
        self.assertEqual(result.previous.page, 0)
        self.assertEqual(result.previous.row_index, 0)
        self.assertEqual(result.previous.tenhou_link, "https://example.test/first")
        self.assertEqual(result.next.source_row_number, 5)
        self.assertEqual(result.next.page, 2)
        self.assertEqual(result.next.row_index, 0)
        self.assertEqual(result.next.tenhou_link, "https://example.test/next")

    def test_returns_empty_neighbor_at_file_boundary(self) -> None:
        result = get_neighbors(
            CsvNeighborsRequest(
                session_id=self.session_id,
                line_number="101",
                ids="3_2_2",
                page_size=2,
            )
        )

        self.assertTrue(result.success)
        self.assertIsNone(result.previous)
        self.assertEqual(result.next.source_row_number, 3)

    def test_normalizes_whitespace_only_neighbor_link_as_missing(self) -> None:
        result = get_neighbors(
            CsvNeighborsRequest(
                session_id=self.session_id,
                line_number="105",
                ids="3_2_2",
                page_size=2,
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.next.source_row_number, 6)
        self.assertEqual(result.next.tenhou_link, "")


if __name__ == "__main__":
    unittest.main()
