"""CSV reader endpoints ported from TenhouCsvReader concepts.

Read-only, paged CSV/ZIP reader with filtering and PNG preview support.
"""

from __future__ import annotations

import csv
import os
import sqlite3
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import anyio
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

router = APIRouter()


Operator = Literal["=", "==", "!=", ">", ">=", "<", "<="]


@dataclass
class CsvSession:
    source_path: str
    csv_path: str
    display_path: str
    display_name: str
    headers: list[str]
    extracted_root: Optional[str] = None
    image_paths: list[str] = field(default_factory=list)
    model_paths: list[str] = field(default_factory=list)
    training_folder: Optional[str] = None
    cache_key: Optional[tuple[str, int, int]] = None
    touched: float = field(default_factory=time.monotonic)
    leases: int = 0
    closing: bool = False


_SESSIONS: dict[str, CsvSession] = {}
_SESSION_LOCK = threading.RLock()
_OPENING = 0
_SESSION_TTL = float(os.getenv("ANALYZER_CSV_TTL_SECONDS", "1800"))
_MAX_SESSIONS = int(os.getenv("ANALYZER_MAX_CSV_SESSIONS", "32"))
if _SESSION_TTL <= 0 or _MAX_SESSIONS <= 0:
    raise ValueError("CSV session limits must be positive")


def _dispose_csv(session: CsvSession):
    if session.extracted_root:
        shutil.rmtree(session.extracted_root, ignore_errors=True)


def cleanup_csv_sessions():
    expired = []
    with _SESSION_LOCK:
        now = time.monotonic()
        for sid, session in list(_SESSIONS.items()):
            if not session.leases and (session.closing or now - session.touched >= _SESSION_TTL):
                expired.append(_SESSIONS.pop(sid))
    for session in expired:
        _dispose_csv(session)


def _acquire_session(session_id: str) -> Optional[CsvSession]:
    cleanup_csv_sessions()
    with _SESSION_LOCK:
        session = _SESSIONS.get(session_id)
        if session is None or session.closing:
            return None
        session.leases += 1
        return session


def _release_session(session_id: str, session: CsvSession):
    dispose = False
    with _SESSION_LOCK:
        session.leases -= 1
        session.touched = time.monotonic()
        if session.closing and not session.leases and _SESSIONS.get(session_id) is session:
            del _SESSIONS[session_id]
            dispose = True
    if dispose:
        _dispose_csv(session)


@contextmanager
def _session_lease(session_id: str):
    session = _acquire_session(session_id)
    try:
        yield session
    finally:
        if session is not None:
            _release_session(session_id, session)


class CsvImageResponse(FileResponse):
    """Keep extracted files alive until streaming completes or fails."""
    def __init__(self, path: str, session_id: str, session: CsvSession):
        super().__init__(path, media_type="image/png")
        self.session_id = session_id
        self.session = session

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Finish the lease even if the client disconnects; filesystem cleanup
            # must not run on the ASGI event loop.
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(_release_session, self.session_id, self.session)

_SESSION_TEMP_ROOT = Path(tempfile.gettempdir()) / "archanalyzer_csv_reader"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_ROOT = (_REPO_ROOT / "Mahjong-Hidden-Information-Forecast" / "Transformer" / "output").resolve()
_INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "csv_reader_index.sqlite3"
_INDEX_SCAN_LOCK = threading.Lock()
_CSV_CHECKPOINT_INTERVAL = 1000


class OpenCsvRequest(BaseModel):
    path: str


class OpenCsvResponse(BaseModel):
    success: bool
    error_code: Optional[str] = None
    session_id: str = ""
    display_name: str = ""
    display_path: str = ""
    headers: list[str] = []
    images: list[dict[str, str]] = []
    model_paths: list[str] = []
    training_folder: Optional[str] = None
    error: Optional[str] = None


class CsvPageRequest(BaseModel):
    session_id: str
    page: int = 0
    page_size: int = 1000
    filter_text: str = ""
    include_total: bool = False


class CsvPageResponse(BaseModel):
    success: bool
    headers: list[str] = []
    rows: list[list[str]] = []
    page: int = 0
    page_size: int = 1000
    start_row: int = 0
    end_row: int = 0
    has_previous: bool = False
    has_next: bool = False
    total_rows: Optional[int] = None
    scanned_rows: int = 0
    matched_rows: Optional[int] = None
    error: Optional[str] = None


class CsvNeighborsRequest(BaseModel):
    session_id: str
    line_number: str
    ids: str
    page_size: int = 1000


class CsvNeighbor(BaseModel):
    source_row_number: int
    page: int
    row_index: int
    row: list[str]
    tenhou_link: str


class CsvNeighborsResponse(BaseModel):
    success: bool
    previous: Optional[CsvNeighbor] = None
    next: Optional[CsvNeighbor] = None
    error: Optional[str] = None


class CloseSessionRequest(BaseModel):
    session_id: str


class OutputSearchResponse(BaseModel):
    success: bool
    root: str
    paths: list[str] = []
    indexed_count: int = 0
    last_scanned_at: Optional[float] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class FilterCondition:
    column_index: int
    column_name: str
    operator: str
    raw_value: str
    numeric_value: Optional[float]


def _safe_abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _read_headers(csv_path: str) -> list[str]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError("CSV header cannot be empty")
    if not headers:
        raise ValueError("CSV header cannot be empty")
    return _build_unique_column_names(headers)


def _build_unique_column_names(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for i, header in enumerate(headers):
        name = header.strip() or f"Column{i + 1}"
        key = name.lower()
        if key not in seen:
            seen[key] = 1
            result.append(name)
        else:
            seen[key] += 1
            result.append(f"{name}_{seen[key]}")
    return result


def _extract_zip_session(zip_path: str) -> tuple[str, list[str], list[str], str, str, str]:
    _SESSION_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    session_root = _SESSION_TEMP_ROOT / f"{uuid.uuid4().hex}"
    session_root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            entries = [e for e in zf.infolist() if not e.is_dir()]
            csv_entries = [e for e in entries if Path(e.filename).suffix.lower() == ".csv"]
            if not csv_entries:
                raise ValueError("No CSV file found in zip")
            csv_entries.sort(key=lambda e: (("pred" not in Path(e.filename).name.lower()), -e.file_size, e.filename.lower()))
            csv_entry = csv_entries[0]
            csv_path = _safe_extract_member(zf, csv_entry, session_root / "csv")

            image_paths: list[str] = []
            for entry in sorted([e for e in entries if Path(e.filename).suffix.lower() == ".png"], key=lambda e: e.filename.lower()):
                image_paths.append(_safe_extract_member(zf, entry, session_root / "images"))
            model_paths = [
                _safe_extract_member(zf, entry, session_root / "models")
                for entry in sorted(entries, key=lambda e: e.filename.lower())
                if Path(entry.filename).suffix.lower() in {".pth", ".pt"}
            ]

        display_path = f"{zip_path} | CSV: {csv_entry.filename}"
        member_identity = f"{csv_entry.filename}:{csv_entry.CRC}:{csv_entry.file_size}"
        return csv_path, image_paths, model_paths, str(session_root), display_path, member_identity
    except Exception:
        shutil.rmtree(session_root, ignore_errors=True)
        raise


def _has_training_results(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    suffixes = ("_history.csv", "_phase_summary.csv", "_turn_metrics.csv")
    try:
        return any(p.name == "comparison_table.csv" or p.name.endswith(suffixes) for p in folder.rglob("*.csv"))
    except OSError:
        return False


def _companion_models(source_path: Path) -> list[str]:
    stem = source_path.stem.lower().removesuffix("_pred")
    candidates = list(source_path.parent.glob("*.pth")) + list(source_path.parent.glob("*.pt"))
    matching = [p for p in candidates if stem in p.stem.lower() or p.stem.lower() in stem]
    return [str(p.resolve()) for p in sorted(matching)]


def _connect_index() -> sqlite3.Connection:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(_INDEX_PATH)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, relative_path TEXT NOT NULL, name TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS csv_page_offsets ("
        "path TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, "
        "row_number INTEGER NOT NULL, offset INTEGER NOT NULL, "
        "PRIMARY KEY (path, size, mtime_ns, row_number))"
    )
    return connection


def _read_index_metadata(connection: sqlite3.Connection) -> tuple[int, Optional[float]]:
    count = int(connection.execute("SELECT COUNT(*) FROM files").fetchone()[0])
    row = connection.execute("SELECT value FROM metadata WHERE key = 'last_scanned_at'").fetchone()
    return count, float(row[0]) if row else None


@router.post("/output_index/scan", response_model=OutputSearchResponse)
def scan_output_index() -> OutputSearchResponse:
    if not _INDEX_SCAN_LOCK.acquire(blocking=False):
        return OutputSearchResponse(
            success=False,
            root=str(_OUTPUT_ROOT),
            error="An output index scan is already in progress",
        )
    try:
        if not _OUTPUT_ROOT.is_dir():
            raise ValueError(f"Transformer output folder not found: {_OUTPUT_ROOT}")
        records: list[tuple[str, str, str]] = []
        for root, _, filenames in os.walk(_OUTPUT_ROOT):
            for filename in filenames:
                if Path(filename).suffix.lower() not in {".csv", ".zip"}:
                    continue
                path = (Path(root) / filename).resolve()
                records.append((str(path), str(path.relative_to(_OUTPUT_ROOT)), filename))
        scanned_at = time.time()
        with _connect_index() as connection:
            connection.execute("DELETE FROM files")
            connection.executemany("INSERT INTO files(path, relative_path, name) VALUES (?, ?, ?)", records)
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_scanned_at', ?)",
                (str(scanned_at),),
            )
        return OutputSearchResponse(success=True, root=str(_OUTPUT_ROOT), indexed_count=len(records), last_scanned_at=scanned_at)
    except Exception as exc:
        return OutputSearchResponse(success=False, root=str(_OUTPUT_ROOT), error=str(exc))
    finally:
        _INDEX_SCAN_LOCK.release()


@router.get("/output_index/status", response_model=OutputSearchResponse)
def get_output_index_status() -> OutputSearchResponse:
    """Return persisted index metadata without opening a search result list."""
    try:
        with _connect_index() as connection:
            count, scanned_at = _read_index_metadata(connection)
        return OutputSearchResponse(
            success=True,
            root=str(_OUTPUT_ROOT),
            indexed_count=count,
            last_scanned_at=scanned_at,
        )
    except Exception as exc:
        return OutputSearchResponse(success=False, root=str(_OUTPUT_ROOT), error=str(exc))


@router.get("/output_index/search", response_model=OutputSearchResponse)
def search_output_index(query: str = "", limit: int = 40) -> OutputSearchResponse:
    try:
        with _connect_index() as connection:
            count, scanned_at = _read_index_metadata(connection)
            pattern = f"%{query.strip()}%"
            rows = connection.execute(
                "SELECT path FROM files WHERE relative_path LIKE ? OR name LIKE ? ORDER BY relative_path LIMIT ?",
                (pattern, pattern, max(1, min(limit, 100))),
            ).fetchall()
        return OutputSearchResponse(
            success=True, root=str(_OUTPUT_ROOT), paths=[row[0] for row in rows],
            indexed_count=count, last_scanned_at=scanned_at,
        )
    except Exception as exc:
        return OutputSearchResponse(success=False, root=str(_OUTPUT_ROOT), error=str(exc))


def _safe_extract_member(zf: zipfile.ZipFile, entry: zipfile.ZipInfo, root: Path) -> str:
    root = root.resolve()
    dest = (root / entry.filename).resolve()
    if not str(dest).startswith(str(root) + os.sep):
        raise ValueError(f"Archive contains unsafe path: {entry.filename}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(entry) as src, open(dest, "wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return str(dest)


def _parse_filter(filter_text: str, headers: list[str]) -> list[FilterCondition]:
    filter_text = filter_text.strip()
    if not filter_text:
        return []

    operators = [">=", "<=", "!=", "==", "=", ">", "<"]
    header_lookup = {h.lower(): i for i, h in enumerate(headers)}
    conditions: list[FilterCondition] = []

    for clause in [c.strip() for c in filter_text.split(",") if c.strip()]:
        found: Optional[tuple[str, int]] = None
        for op in operators:
            idx = clause.find(op)
            if idx > 0:
                found = (op, idx)
                break
        if found is None:
            raise ValueError(f"Invalid filter clause: '{clause}'. Use formats like prediction = 1 or probability > 0.7")
        op, idx = found
        col_name = clause[:idx].strip()
        raw_value = clause[idx + len(op):].strip().strip("'\"")
        if not col_name or raw_value == "":
            raise ValueError(f"Invalid filter clause: '{clause}'")

        col_idx = header_lookup.get(col_name.lower())
        if col_idx is None:
            matches = [h for h in headers if h.lower().startswith(col_name.lower())]
            if len(matches) == 1:
                col_idx = header_lookup[matches[0].lower()]
            elif len(matches) > 1:
                raise ValueError(f"Column '{col_name}' is ambiguous. Matches: {', '.join(matches)}")
            else:
                raise ValueError(f"Unknown column '{col_name}'")

        numeric_value: Optional[float]
        try:
            numeric_value = float(raw_value.replace(",", ""))
        except ValueError:
            numeric_value = None
        if op in {">", ">=", "<", "<="} and numeric_value is None:
            raise ValueError(f"Filter clause '{clause}' expects a numeric value")
        conditions.append(FilterCondition(col_idx, col_name, op, raw_value, numeric_value))
    return conditions


def _matches(row: list[str], conditions: list[FilterCondition]) -> bool:
    for cond in conditions:
        cell = row[cond.column_index] if cond.column_index < len(row) else ""
        if cond.numeric_value is not None:
            try:
                val = float(cell.replace(",", ""))
            except ValueError:
                val = None
            if val is not None:
                if cond.operator in {"=", "=="} and not (val == cond.numeric_value):
                    return False
                if cond.operator == "!=" and not (val != cond.numeric_value):
                    return False
                if cond.operator == ">" and not (val > cond.numeric_value):
                    return False
                if cond.operator == ">=" and not (val >= cond.numeric_value):
                    return False
                if cond.operator == "<" and not (val < cond.numeric_value):
                    return False
                if cond.operator == "<=" and not (val <= cond.numeric_value):
                    return False
                continue
        if cond.operator in {">", ">=", "<", "<="}:
            return False
        same = cell.lower() == cond.raw_value.lower()
        if cond.operator in {"=", "=="} and not same:
            return False
        if cond.operator == "!=" and same:
            return False
    return True


def _image_payload(session_id: str, session: CsvSession) -> list[dict[str, str]]:
    return [
        {"id": str(i), "name": os.path.basename(path), "url": f"/api/csv_reader/image/{session_id}/{i}"}
        for i, path in enumerate(session.image_paths)
    ]


@router.post("/open", response_model=OpenCsvResponse)
def open_csv(request: OpenCsvRequest) -> OpenCsvResponse:
    global _OPENING
    cleanup_csv_sessions()
    with _SESSION_LOCK:
        if len(_SESSIONS) + _OPENING >= _MAX_SESSIONS:
            return OpenCsvResponse(success=False, error_code="CSV_CAPACITY",
                                   error="CSV session 數量已達上限，請關閉未使用的檔案後重試。")
        _OPENING += 1
    extracted_root = None
    published = False
    try:
        path = _safe_abs_path(request.path)
        if not os.path.isfile(path):
            return OpenCsvResponse(success=False, error=f"File not found: {path}")
        ext = Path(path).suffix.lower()
        extracted_root = None
        image_paths: list[str] = []
        model_paths: list[str] = []
        cache_key = None
        display_path = path
        csv_path = path
        if ext == ".zip":
            csv_path, image_paths, model_paths, extracted_root, display_path, member_identity = _extract_zip_session(path)
            source_stat = os.stat(path)
            cache_key = (f"zip:{path}:{member_identity}", source_stat.st_size, source_stat.st_mtime_ns)
        elif ext != ".csv":
            return OpenCsvResponse(success=False, error="Only .csv and .zip are supported")

        headers = _read_headers(csv_path)
        source = Path(path)
        if not model_paths:
            model_paths = _companion_models(source)
        training_folder = str(source.parent.resolve()) if _has_training_results(source.parent) else None
        session_id = uuid.uuid4().hex
        session = CsvSession(
            source_path=path,
            csv_path=csv_path,
            display_path=display_path,
            display_name=(f"ZIP: {os.path.basename(path)}" if ext == ".zip" else f"CSV: {os.path.basename(path)}"),
            headers=headers,
            extracted_root=extracted_root,
            image_paths=image_paths,
            model_paths=model_paths,
            training_folder=training_folder,
            cache_key=cache_key,
        )
        with _SESSION_LOCK:
            _SESSIONS[session_id] = session
            published = True
        return OpenCsvResponse(
            success=True,
            session_id=session_id,
            display_name=session.display_name,
            display_path=session.display_path,
            headers=headers,
            images=_image_payload(session_id, session),
            model_paths=session.model_paths,
            training_folder=session.training_folder,
        )
    except Exception as exc:
        return OpenCsvResponse(success=False, error=str(exc))
    finally:
        if extracted_root and not published:
            shutil.rmtree(extracted_root, ignore_errors=True)
        with _SESSION_LOCK:
            _OPENING -= 1


@router.post("/page", response_model=CsvPageResponse)
def get_page(request: CsvPageRequest) -> CsvPageResponse:
    with _session_lease(request.session_id) as session:
        if session is None:
            return CsvPageResponse(success=False, error="Unknown CSV session")
        return _get_page(request, session)


def _get_page(request: CsvPageRequest, session: CsvSession) -> CsvPageResponse:

    page_size = max(1, min(int(request.page_size), 5000))
    page = max(0, int(request.page))
    start_match = page * page_size
    end_match_exclusive = start_match + page_size
    rows: list[list[str]] = []
    scanned = 0
    matched = 0
    has_next = False
    total_rows: Optional[int] = None

    try:
        conditions = _parse_filter(request.filter_text, session.headers)
        if session.cache_key is None:
            file_stat = os.stat(session.csv_path)
            file_key = (session.csv_path, file_stat.st_size, file_stat.st_mtime_ns)
        else:
            file_key = session.cache_key
        checkpoint_row = 0
        checkpoint_offset: Optional[int] = None
        if not conditions and start_match:
            try:
                with closing(_connect_index()) as connection:
                    checkpoint = connection.execute(
                        "SELECT row_number, offset FROM csv_page_offsets "
                        "WHERE path = ? AND size = ? AND mtime_ns = ? AND row_number <= ? "
                        "ORDER BY row_number DESC LIMIT 1",
                        (*file_key, start_match),
                    ).fetchone()
                    if checkpoint:
                        checkpoint_row, checkpoint_offset = checkpoint
            except (OSError, sqlite3.Error):
                pass

        new_checkpoints: list[tuple[str, int, int, int, int]] = []
        physical_row = checkpoint_row
        matched = checkpoint_row
        with open(session.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            if checkpoint_offset is None:
                reader = csv.reader(iter(f.readline, ""))
                next(reader, None)  # header
            else:
                f.seek(checkpoint_offset)
                reader = csv.reader(iter(f.readline, ""))
            while True:
                checkpoint_due = physical_row % _CSV_CHECKPOINT_INTERVAL == 0
                row_offset = f.tell() if checkpoint_due else None
                try:
                    row = next(reader)
                except StopIteration:
                    total_rows = matched
                    break
                if checkpoint_due and row_offset is not None:
                    new_checkpoints.append((*file_key, physical_row, row_offset))
                physical_row += 1
                scanned += 1
                if conditions and not _matches(row, conditions):
                    continue
                if matched >= end_match_exclusive:
                    has_next = True
                    if not request.include_total:
                        break
                if start_match <= matched < end_match_exclusive:
                    rows.append(row)
                matched += 1

        if new_checkpoints:
            try:
                with closing(_connect_index()) as connection:
                    with connection:
                        connection.executemany(
                            "INSERT OR IGNORE INTO csv_page_offsets "
                            "(path, size, mtime_ns, row_number, offset) VALUES (?, ?, ?, ?, ?)",
                            new_checkpoints,
                        )
            except (OSError, sqlite3.Error):
                pass

        if request.include_total:
            total_rows = matched
            has_next = end_match_exclusive < matched

        start_row = start_match + 1 if rows else 0
        end_row = start_match + len(rows) if rows else 0
        return CsvPageResponse(
            success=True,
            headers=session.headers,
            rows=rows,
            page=page,
            page_size=page_size,
            start_row=start_row,
            end_row=end_row,
            has_previous=page > 0,
            has_next=has_next,
            total_rows=total_rows,
            scanned_rows=scanned,
            matched_rows=matched if conditions or total_rows is not None else None,
        )
    except Exception as exc:
        return CsvPageResponse(success=False, error=str(exc), headers=session.headers)


@router.post("/neighbors", response_model=CsvNeighborsResponse)
def get_neighbors(request: CsvNeighborsRequest) -> CsvNeighborsResponse:
    with _session_lease(request.session_id) as session:
        if session is None:
            return CsvNeighborsResponse(success=False, error="Unknown CSV session")
        return _get_neighbors(request, session)


def _get_neighbors(request: CsvNeighborsRequest, session: CsvSession) -> CsvNeighborsResponse:

    header_lookup = {
        header.strip().lower(): index for index, header in enumerate(session.headers)
    }
    required_columns = ("line_number", "ids", "tenhou_link")
    missing_columns = [name for name in required_columns if name not in header_lookup]
    if missing_columns:
        return CsvNeighborsResponse(
            success=False,
            error=f"CSV is missing required columns: {', '.join(missing_columns)}",
        )

    line_index = header_lookup["line_number"]
    ids_index = header_lookup["ids"]
    link_index = header_lookup["tenhou_link"]
    page_size = min(2000, max(1, int(request.page_size)))
    requested_line = request.line_number.strip()
    requested_ids = request.ids.strip()
    previous_row: Optional[tuple[int, list[str]]] = None
    next_row: Optional[tuple[int, list[str]]] = None
    found_current = False

    try:
        with open(session.csv_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)
            for source_row_number, row in enumerate(reader, start=1):
                row_ids = row[ids_index].strip() if ids_index < len(row) else ""
                row_line = row[line_index].strip() if line_index < len(row) else ""
                if not found_current:
                    if row_ids == requested_ids and row_line == requested_line:
                        found_current = True
                    elif row_ids == requested_ids:
                        previous_row = (source_row_number, row)
                elif row_ids == requested_ids:
                    next_row = (source_row_number, row)
                    break

        if not found_current:
            return CsvNeighborsResponse(
                success=False,
                error="Current prediction row was not found in the CSV",
            )

        def to_neighbor(candidate: Optional[tuple[int, list[str]]]) -> Optional[CsvNeighbor]:
            if candidate is None:
                return None
            source_row_number, row = candidate
            return CsvNeighbor(
                source_row_number=source_row_number,
                page=(source_row_number - 1) // page_size,
                row_index=(source_row_number - 1) % page_size,
                row=row,
                tenhou_link=row[link_index].strip() if link_index < len(row) else "",
            )

        return CsvNeighborsResponse(
            success=True,
            previous=to_neighbor(previous_row),
            next=to_neighbor(next_row),
        )
    except Exception as exc:
        return CsvNeighborsResponse(success=False, error=str(exc))


@router.get("/image/{session_id}/{image_index}")
async def get_image(session_id: str, image_index: int) -> FileResponse:
    # TTL cleanup can remove large ZIP directories; run it off the event loop.
    session = await run_in_threadpool(_acquire_session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        if image_index < 0 or image_index >= len(session.image_paths):
            raise HTTPException(status_code=404, detail="Image not found")
        return CsvImageResponse(session.image_paths[image_index], session_id, session)
    except Exception:
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(_release_session, session_id, session)
        raise


@router.post("/close")
def close_session(request: CloseSessionRequest) -> dict[str, Any]:
    dispose = None
    with _SESSION_LOCK:
        session = _SESSIONS.get(request.session_id)
        if session is not None:
            session.closing = True
            if not session.leases:
                dispose = _SESSIONS.pop(request.session_id)
    if dispose is not None:
        _dispose_csv(dispose)
    return {"success": True}
