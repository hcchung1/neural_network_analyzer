"""Process-local model ownership and bounded admission for expensive computation."""
from contextlib import contextmanager
from collections import deque
import json
from pathlib import Path
from types import MappingProxyType
from dataclasses import dataclass
import os
import secrets
import threading
import time
from typing import Callable

from fastapi import HTTPException
from model_engine.transformer_model import TransformerModelEngine


def resource_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"error": {"code": code, "message": message}},
                         headers={"Retry-After": "2"} if status == 503 else None)


class ComputeGate:
    """One running job; a bounded number of worker threads may wait for it."""

    def __init__(self, max_waiting: int = 4, wait_seconds: float = 120):
        if max_waiting < 0 or wait_seconds <= 0:
            raise ValueError("Compute limits must be nonnegative with a positive timeout")
        self._condition = threading.Condition()
        self._waiters = deque()
        self._running = False
        self.max_waiting = max_waiting
        self.wait_seconds = wait_seconds

    @contextmanager
    def slot(self):
        token = object()
        acquired = False
        deadline = time.monotonic() + self.wait_seconds
        with self._condition:
            if self._running or self._waiters:
                if len(self._waiters) >= self.max_waiting:
                    raise resource_error(503, "COMPUTE_BUSY", "分析服務已滿載，請稍後重試。")
                self._waiters.append(token)
                try:
                    while self._running or self._waiters[0] is not token:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise resource_error(503, "COMPUTE_BUSY", "等待分析逾時，請稍後重試。")
                        self._condition.wait(remaining)
                    self._waiters.popleft()
                except BaseException:
                    if token in self._waiters:
                        self._waiters.remove(token)
                    self._condition.notify_all()
                    raise
            self._running = True
            acquired = True
        try:
            yield
        finally:
            if acquired:
                with self._condition:
                    self._running = False
                    self._condition.notify_all()


def dispose_engine(engine: TransformerModelEngine):
    engine.clear_attention_cache()
    engine._model = None
    engine._is_loaded = False


@dataclass
class _Model:
    engine: TransformerModelEngine
    touched: float
    size: int
    leases: int = 0
    replacing: bool = False


class ModelStore:
    def __init__(self, ttl_seconds: float = 1800, max_models: int = 4,
                 max_bytes: int = 2048 * 1024 * 1024, clock: Callable = time.monotonic):
        if min(ttl_seconds, max_models, max_bytes) <= 0:
            raise ValueError("Model resource limits must be positive")
        self.ttl = ttl_seconds
        self.max_models = max_models
        self.max_bytes = max_bytes
        self.clock = clock
        self._models: dict[str, _Model] = {}
        self._lock = threading.RLock()
        self._loading = 0

    def cleanup_expired(self):
        with self._lock:
            for handle, entry in list(self._models.items()):
                if not entry.leases and self.clock() - entry.touched >= self.ttl:
                    dispose_engine(entry.engine)
                    del self._models[handle]

    def _find(self, handle: str | None) -> _Model:
        if not handle:
            raise resource_error(401, "MODEL_HANDLE_REQUIRED", "請先在此分頁載入模型。")
        entry = self._models.get(handle)
        if entry is None:
            raise resource_error(404, "MODEL_EXPIRED", "模型已到期或已釋放，請重新載入。")
        return entry

    @contextmanager
    def lease(self, handle: str | None):
        with self._lock:
            self.cleanup_expired()
            entry = self._find(handle)
            if entry.replacing:
                raise resource_error(409, "MODEL_BUSY", "模型正在更換中，請稍後重試。")
            entry.leases += 1
        try:
            yield entry.engine
        finally:
            with self._lock:
                entry.leases -= 1
                entry.touched = self.clock()

    def load(self, builder: Callable[[], TransformerModelEngine], previous_handle: str | None = None) -> str:
        """Build before publishing. Failed replacement leaves the previous model intact."""
        with self._lock:
            self.cleanup_expired()
            previous = self._find(previous_handle) if previous_handle else None
            if previous and previous.leases:
                raise resource_error(409, "MODEL_BUSY", "模型仍在使用中，請稍後重試。")
            if not previous and len(self._models) + self._loading >= self.max_models:
                raise resource_error(503, "MODEL_CAPACITY", "模型數量已達上限，請釋放未使用的模型後重試。")
            if previous:
                previous.leases += 1
                previous.replacing = True
            self._loading += 1
        engine = None
        published = False
        try:
            engine = builder()
            size = sum(t.numel() * t.element_size() for t in
                       list(engine.model.parameters()) + list(engine.model.buffers()))
            with self._lock:
                retained = sum(e.size for e in self._models.values()) - (previous.size if previous else 0)
                if retained + size > self.max_bytes:
                    raise resource_error(503, "MODEL_CAPACITY", "模型記憶體額度不足，請釋放模型或使用較小模型。")
                handle = secrets.token_urlsafe(32)
                self._models[handle] = _Model(engine, self.clock(), size)
                if previous:
                    del self._models[previous_handle]
                    dispose_engine(previous.engine)
                published = True
                return handle
        finally:
            if engine is not None and not published:
                dispose_engine(engine)
            with self._lock:
                self._loading -= 1
                if previous:
                    previous.leases -= 1
                    previous.replacing = False
                    previous.touched = self.clock()

    def delete(self, handle: str | None):
        with self._lock:
            self.cleanup_expired()
            entry = self._find(handle)
            if entry.leases:
                raise resource_error(409, "MODEL_BUSY", "模型仍在使用中，請稍後重試。")
            del self._models[handle]
            dispose_engine(entry.engine)

    def handles(self) -> set[str]:
        with self._lock:
            return set(self._models)


compute_gate = ComputeGate(
    max_waiting=int(os.getenv("ANALYZER_MAX_WAITING_JOBS", "4")),
    wait_seconds=float(os.getenv("ANALYZER_COMPUTE_WAIT_SECONDS", "120")),
)


def _freeze_metadata(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    return value


@dataclass
class SharedModelEntry:
    engine: TransformerModelEngine
    metadata: object
    touched: float
    size: int
    leases: int = 0


class SharedModelCache:
    """Recreatable resident engines keyed by checkpoint identity, with protected leases."""

    def __init__(self, ttl_seconds=300, max_models=2, max_bytes=2048 * 1024 * 1024,
                 clock=time.monotonic):
        if min(ttl_seconds, max_models, max_bytes) <= 0:
            raise ValueError("Shared model resource limits must be positive")
        self.ttl, self.max_models, self.max_bytes = ttl_seconds, max_models, max_bytes
        self.clock = clock
        self._entries = {}
        self._lock = threading.RLock()
        self._closed = False

    @staticmethod
    def key(selection):
        path = Path(selection.checkpoint_path).expanduser().resolve()
        identity = None
        if path.is_file():
            stat = path.stat()
            identity = (stat.st_size, stat.st_mtime_ns)
        return (str(path), identity,
                json.dumps(selection.manifest_spec, sort_keys=True, separators=(",", ":")),
                selection.requested_phase, selection.device)

    def cleanup_expired(self):
        with self._lock:
            for key, entry in list(self._entries.items()):
                if not entry.leases and self.clock() - entry.touched >= self.ttl:
                    del self._entries[key]
                    dispose_engine(entry.engine)

    @contextmanager
    def lease(self, selection, builder):
        with self._lock:
            if self._closed:
                raise resource_error(503, "MODEL_CAPACITY", "模型快取已關閉。")
            self.cleanup_expired()
            key = self.key(selection)
            entry = self._entries.get(key)
            if entry is None:
                # Construct before eviction: a failed build never destroys a valid selection.
                engine = None
                try:
                    engine, metadata = builder()
                    size = sum(t.numel() * t.element_size() for t in
                               list(engine.model.parameters()) + list(engine.model.buffers()))
                    if size > self.max_bytes:
                        raise resource_error(503, "MODEL_CAPACITY", "模型記憶體額度不足。")
                    victims = []
                    count, retained = len(self._entries), sum(e.size for e in self._entries.values())
                    for candidate_key, candidate in sorted(self._entries.items(), key=lambda pair: pair[1].touched):
                        if count + 1 <= self.max_models and retained + size <= self.max_bytes:
                            break
                        if not candidate.leases:
                            victims.append(candidate_key)
                            count -= 1
                            retained -= candidate.size
                    if count + 1 > self.max_models or retained + size > self.max_bytes:
                        raise resource_error(503, "MODEL_CAPACITY", "模型快取仍在使用中，請稍後重試。")
                    if self.key(selection) != key:
                        raise resource_error(409, "CHECKPOINT_CHANGED", "模型檔案載入時已變更，請重試。")
                    entry = SharedModelEntry(engine, _freeze_metadata(metadata), self.clock(), size)
                    for victim_key in victims:
                        dispose_engine(self._entries.pop(victim_key).engine)
                    self._entries[key] = entry
                except BaseException:
                    if engine is not None:
                        dispose_engine(engine)
                    raise
            entry.leases += 1
            entry.touched = self.clock()
        try:
            yield entry
        finally:
            with self._lock:
                entry.leases -= 1
                entry.touched = self.clock()
                if self._closed and not entry.leases:
                    self._entries.pop(key, None)
                    dispose_engine(entry.engine)

    def close(self):
        with self._lock:
            self._closed = True
            for key, entry in list(self._entries.items()):
                if not entry.leases:
                    dispose_engine(entry.engine)
                    del self._entries[key]


shared_model_cache = SharedModelCache(
    ttl_seconds=float(os.getenv("ANALYZER_SHARED_MODEL_TTL_SECONDS", "300")),
    max_models=int(os.getenv("ANALYZER_MAX_SHARED_MODELS", "2")),
    max_bytes=int(os.getenv("ANALYZER_SHARED_MODEL_MEMORY_MB", "2048")) * 1024 * 1024,
)
