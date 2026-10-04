"""Persistent binary and text locations; payload reads stay bounded to one record/line."""
from __future__ import annotations

from contextlib import closing
import hashlib
import mmap
import os
from pathlib import Path
import sqlite3
import tempfile
import threading

_LOCKS = {}
_LOCKS_LOCK = threading.Lock()


class BinaryRecordIndex:
    def __init__(self, database_path=None):
        self.database_path = Path(database_path or Path(__file__).with_name('binary_sample_index.sqlite3')).resolve()
        with _LOCKS_LOCK:
            self._lock = _LOCKS.setdefault(str(self.database_path), threading.RLock())

    @staticmethod
    def _identity(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    @staticmethod
    def _validate(size, record_size):
        if record_size < 4 or size == 0 or size % record_size:
            raise ValueError('Invalid or truncated binary records')

    def read_record(self, path, record_index, record_size):
        path = Path(path).resolve()
        size, mtime = self._identity(path)
        self._validate(size, record_size)
        if record_index < 0 or record_index >= size // record_size:
            raise ValueError('Record index out of range')
        with path.open('rb') as stream:
            stream.seek(record_index * record_size)
            record = stream.read(record_size)
        if len(record) != record_size or self._identity(path) != (size, mtime):
            raise ValueError('Binary file changed or was truncated during read')
        return record

    @staticmethod
    def _schema(connection):
        connection.execute('CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, record_size INTEGER NOT NULL)')
        connection.execute('CREATE TABLE IF NOT EXISTS records (path TEXT NOT NULL, line_number INTEGER NOT NULL, record_index INTEGER NOT NULL, PRIMARY KEY(path, record_index))')
        connection.execute('CREATE INDEX IF NOT EXISTS record_lines ON records(path, line_number, record_index)')
        connection.execute('CREATE TABLE IF NOT EXISTS text_files (path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL)')
        connection.execute('CREATE TABLE IF NOT EXISTS text_lines (path TEXT NOT NULL, line_number INTEGER NOT NULL, byte_offset INTEGER NOT NULL, byte_length INTEGER NOT NULL, digest BLOB NOT NULL, PRIMARY KEY(path, line_number))')
        if 'digest' not in {row[1] for row in connection.execute('PRAGMA table_info(text_lines)')}:
            connection.execute('ALTER TABLE text_lines ADD COLUMN digest BLOB')
            connection.execute('DELETE FROM text_files')
        connection.commit()

    def _reset_database(self):
        """Publish a complete empty replacement atomically after SQLite corruption."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=self.database_path.name + '.', dir=self.database_path.parent)
        os.close(fd)
        try:
            with closing(sqlite3.connect(name)) as connection:
                self._schema(connection)
            os.replace(name, self.database_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _build(self, connection, path, identity, record_size):
        size, mtime = identity
        self._validate(size, record_size)
        with connection:
            connection.execute('DELETE FROM records WHERE path=?', (str(path),))
            connection.execute('DELETE FROM files WHERE path=?', (str(path),))
            with path.open('rb') as stream:
                before = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns) != identity:
                    raise ValueError('Binary file changed during index construction')
                # A writer can truncate a mapped source and SIGBUS the process.
                # Snapshot in bounded chunks, then mmap only the private temporary file.
                with tempfile.TemporaryFile() as snapshot:
                    remaining = size
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError('Binary file was truncated during snapshot')
                        snapshot.write(chunk)
                        remaining -= len(chunk)
                    copied = os.fstat(stream.fileno())
                    if (copied.st_size, copied.st_mtime_ns) != identity or self._identity(path) != identity:
                        raise ValueError('Binary file changed during snapshot')
                    snapshot.flush()
                    with mmap.mmap(snapshot.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                        connection.executemany('INSERT INTO records VALUES (?, ?, ?)',
                            ((str(path), int.from_bytes(mapped[offset:offset + 4], 'little'), offset // record_size)
                             for offset in range(0, size, record_size)))
                after = os.fstat(stream.fileno())
                if (after.st_size, after.st_mtime_ns) != identity or self._identity(path) != identity:
                    raise ValueError('Binary file changed during index construction')
            connection.execute('INSERT INTO files VALUES (?, ?, ?, ?)', (str(path), size, mtime, record_size))

    def _lookup(self, path, line_number, record_size):
        identity = self._identity(path)
        self._validate(identity[0], record_size)
        with closing(sqlite3.connect(self.database_path)) as connection:
            self._schema(connection)
            known = connection.execute('SELECT size, mtime_ns, record_size FROM files WHERE path=?', (str(path),)).fetchone()
            if known != (*identity, record_size):
                self._build(connection, path, identity, record_size)
            row = connection.execute('SELECT record_index FROM records WHERE path=? AND line_number=? ORDER BY record_index LIMIT 1', (str(path), line_number)).fetchone()
            if row is None:
                return None
            try:
                record = self.read_record(path, row[0], record_size)
                if int.from_bytes(record[:4], 'little') != line_number:
                    raise ValueError('Stale binary index row')
            except ValueError:
                self._build(connection, path, self._identity(path), record_size)
                row = connection.execute('SELECT record_index FROM records WHERE path=? AND line_number=? ORDER BY record_index LIMIT 1', (str(path), line_number)).fetchone()
                if row is None:
                    return None
                record = self.read_record(path, row[0], record_size)
                if int.from_bytes(record[:4], "little") != line_number:
                    raise ValueError("Binary file changed after index rebuilding")
            return row[0], record

    def find_line(self, path, line_number, record_size):
        path = Path(path).resolve()
        with self._lock:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                return self._lookup(path, line_number, record_size)
            except sqlite3.DatabaseError:
                self._reset_database()
                return self._lookup(path, line_number, record_size)


    def _build_text(self, connection, path, identity):
        """Record offsets in one atomic transaction using a streaming binary scan."""
        with connection:
            connection.execute('DELETE FROM text_lines WHERE path=?', (str(path),))
            connection.execute('DELETE FROM text_files WHERE path=?', (str(path),))
            with path.open('rb') as stream:
                before = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns) != identity:
                    raise ValueError('Info file changed during index construction')

                def offsets():
                    offset = 0
                    line_number = 0
                    while True:
                        raw = stream.readline()
                        if not raw:
                            break
                        yield str(path), line_number, offset, len(raw), hashlib.sha256(raw).digest()
                        offset += len(raw)
                        line_number += 1

                connection.executemany('INSERT INTO text_lines VALUES (?, ?, ?, ?, ?)', offsets())
                after = os.fstat(stream.fileno())
                if (after.st_size, after.st_mtime_ns) != identity or self._identity(path) != identity:
                    raise ValueError('Info file changed during index construction')
            connection.execute('INSERT INTO text_files VALUES (?, ?, ?)', (str(path), *identity))

    def _lookup_text(self, path, line_number):
        identity = self._identity(path)
        with closing(sqlite3.connect(self.database_path)) as connection:
            self._schema(connection)
            known = connection.execute('SELECT size, mtime_ns FROM text_files WHERE path=?', (str(path),)).fetchone()
            if known != identity:
                self._build_text(connection, path, identity)
            for attempt in range(2):
                row = connection.execute('SELECT byte_offset, byte_length, digest FROM text_lines WHERE path=? AND line_number=?', (str(path), line_number)).fetchone()
                if row is None:
                    return b''
                if row[0] >= 0 and row[1] > 0 and row[0] + row[1] <= identity[0]:
                    with path.open('rb') as stream:
                        stream.seek(row[0])
                        raw = stream.read(row[1])
                    if len(raw) != row[1] or self._identity(path) != identity:
                        raise ValueError('Info file changed or was truncated during read')
                    if hashlib.sha256(raw).digest() == row[2]:
                        return raw
                if attempt:
                    raise ValueError('Invalid info offset index after rebuilding')
                self._build_text(connection, path, identity)

    def read_line(self, path, line_number):
        """Read one zero-based info line; missing files and lines return empty bytes."""
        path = Path(path).resolve()
        if line_number < 0 or not path.is_file():
            return b''
        with self._lock:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                return self._lookup_text(path, line_number)
            except sqlite3.DatabaseError:
                self._reset_database()
                return self._lookup_text(path, line_number)
