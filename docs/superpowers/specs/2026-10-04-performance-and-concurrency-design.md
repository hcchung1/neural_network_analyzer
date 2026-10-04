# ArchAnalyzer performance and concurrency design

Date: 2026-10-04

Status: approved in-chat design, awaiting written-spec review

## Purpose

Improve ArchAnalyzer's perceived speed and throughput for both individual and shared use without weakening per-tab model isolation or changing the meaning of existing analysis results.

The work is intentionally ordered by measured architectural impact:

1. remove avoidable backend model-loading and file-I/O costs;
2. reduce collaboration polling and contention;
3. bound browser rendering work for large tables, matrices, and outputs.

## Current constraints

- Preserve the current Next.js App Router plus single-process FastAPI deployment.
- Preserve personal `model_handle` ownership and result ownership checks.
- Preserve the global one-at-a-time admission rule for heavyweight PyTorch work until GPU load testing proves a higher concurrency safe.
- Preserve existing successful API response fields. Internal endpoints may gain additive fields or optional query parameters.
- Preserve all existing uncommitted work in the checkout. Changes for this effort must be narrowly staged and reviewed.
- Add no infrastructure service or runtime dependency in this iteration.
- Use Python 3.11 as the supported backend and test runtime. Next.js continues to require a supported Node 18+ runtime.

## Non-goals

- Multi-host deployment, Redis, PostgreSQL, or distributed inference.
- More than one simultaneous heavyweight GPU computation.
- WebSocket or SSE collaboration transport in this iteration.
- Changing raw model-output semantics or inventing class labels.
- Implementing the currently unavailable embedding probe.
- Restarting the user's existing long-running service as part of implementation.

## Design overview

### Stage 1: backend performance foundation

#### Shared model cache

Collaboration currently constructs and disposes an isolated model for each load and infer request. Replace this with a bounded process-local `SharedModelCache` in `backend/model_engine/resources.py`.

Each cache key contains:

- normalized checkpoint path;
- checkpoint size and nanosecond modification time when it is a real file;
- canonical JSON for `manifest_spec`;
- requested phase;
- device.

Including file identity prevents a checkpoint overwritten at the same path from silently reusing stale weights. Dummy selections use the canonical request fields because no file identity exists.

Each entry owns:

- one isolated `TransformerModelEngine`;
- immutable model metadata used by load responses;
- estimated parameter and buffer bytes;
- last-access time and a lease count.

The cache has independent configurable limits:

- `ANALYZER_SHARED_MODEL_TTL_SECONDS`, default 300;
- `ANALYZER_MAX_SHARED_MODELS`, default 2;
- `ANALYZER_SHARED_MODEL_MEMORY_MB`, default 2048.

Shared models are recreatable resources, so the cache may evict the least-recently-used unleased entry when a new selection needs capacity. It must never evict or dispose a leased entry. A model larger than the configured byte budget fails with the existing structured capacity error style.

`/collaboration/analyze` continues to run under the global compute gate. It leases the cached engine, executes or reports metadata, clears request-specific attention state in `finally`, and returns the current response contract. A load action warms the cache; the following infer action reuses the same engine when the checkpoint identity is unchanged.

The application cleanup loop expires idle shared models. Shutdown disposes remaining engines.

#### Attention reduction

Attention stays as a tensor until the requested batch/head reduction is complete. Add `TransformerModelEngine.get_attention_tensor(layer_idx)`, returning the captured attention as a detached CPU tensor without converting it to Python lists. Personal and collaboration handlers reduce dimensions against that tensor and convert to nested lists exactly once at the response boundary. Keep `get_attention_weights()` as a compatibility wrapper for any existing callers.

This removes the current tensor-to-list-to-tensor round trip while preserving the existing two-dimensional attention response.

#### Event-loop-safe file APIs

Filesystem-heavy handlers must be synchronous FastAPI handlers (`def`) so Starlette runs them in its worker thread pool, or explicitly call `run_in_threadpool` when an async response lifecycle is required.

Apply this to:

- binary file listing and sample search;
- training-result directory scanning and file reading.

The CSV image response remains async because its stream lifetime is async, but cleanup remains off the event loop as it is today.

#### Binary record access and index

Move binary record location into a focused `BinaryRecordIndex` abstraction.

- A request specifying `record_index` and one file reads only that record with `seek`/bounded `read`; it never calls `Path.read_bytes()`.
- A line-number lookup memory-maps the file for its initial scan and stores `(path, size, mtime_ns, line_number, record_index, record_size)` in a persistent SQLite index.
- Repeated line-number lookups use the index and bounded record reads.
- File size or modification-time changes invalidate stale rows for that path.
- Index construction is serialized per database with a lock; unrelated direct record reads remain concurrent.
- Invalid/truncated records return the existing failure envelope and never publish a partial index transaction.

The database is separate from CSV paging state (`backend/data/binary_sample_index.sqlite3`) so either index can be rebuilt independently.

#### Runtime preflight

Document Python 3.11 as required and make `start.sh` fail early with an actionable version message when the interpreter is older. Add the corresponding package engine metadata for Node. This converts the current Python 3.8 import-time failures into a clear startup error.

### Stage 2: collaboration efficiency and fairness

#### Delta polling

The client sends both its current room `revision` and `analysisRevision`.

- When the room revision changed, the server returns the full mutable room metadata and includes analysis only when `analysisRevision` changed.
- When the room revision did not change, the server returns a small heartbeat containing `unchanged: true`, connection-critical identifiers, online membership, host identity, and busy state. It omits entries, view, and analysis.
- The client merges heartbeat fields without replacing stable room data and does not call `setRoom` when nothing observable changed.

Presence writes are throttled to at most once per member every three seconds. The response still treats the requesting member as online immediately. Other members may observe presence up to three seconds late, safely below the existing 15-second offline threshold.

#### Client polling policy

- visible, healthy page: retain approximately one-second polling so collaboration remains responsive;
- hidden page: poll every five seconds;
- failure: exponential backoff with jitter, starting near two seconds and capped at 15 seconds;
- successful response: reset the failure delay;
- identity change or unmount: abort the active request and timer.

No overlapping polls are allowed.

#### Fair compute admission

Replace lock-acquisition scheduling in `ComputeGate` with a bounded FIFO waiter queue while preserving:

- one running heavyweight job;
- the configured maximum number of waiters;
- timeout and `COMPUTE_BUSY` behavior;
- release after exceptions.

No queue-position API is added yet. The existing `Retry-After` response remains the overload signal presented to clients.

### Stage 3: browser and interface performance

#### Truthful health status

`TopContextBar` obtains real health state through a reusable hook. It displays checking, healthy with measured round-trip latency, or offline. Checks run on mount, after a page becomes visible, and at a modest interval while visible. Requests are abortable and do not overlap.

#### CSV row virtualization

Keep backend paging, but render only the visible row window plus overscan. Preserve:

- sticky headers;
- row and cell selection;
- copy actions;
- navigation to a selected prediction row;
- horizontal scrolling and side-panel resizing.

The DOM row count remains bounded even when the selected page size is 2,000.

#### Attention visualization

Render the heatmap in a canvas instead of one DOM element per matrix cell. Pointer coordinates map back to query/key indices for the existing value tooltip. The selected token row and column remain visually highlighted. Empty and all-zero states retain their current explanations.

#### Raw-output preview

Show shape, value count, and a bounded preview by default. Users can expand the complete output and copy or download it as JSON. Probability rendering remains unchanged when an explicit compatible schema exists.

#### Remaining request reductions

- Fetch run artifacts with one catalog query instead of one request per run.
- Read training-result files with bounded concurrency rather than serial requests.
- Debounce durable training-result persistence and move large parsed results from synchronous `localStorage` to IndexedDB when they exceed a small metadata-only threshold.

## Error handling

- Cache load failures do not publish entries and do not remove another valid cached selection.
- Shared cache capacity errors use structured 503 responses and `Retry-After`.
- Stale binary-index rows are treated as cache misses and rebuilt; corrupt index data never changes sample semantics.
- Poll heartbeat parsing is backward compatible: a client receiving a full legacy snapshot still accepts it.
- Abort and timeout errors caused by navigation or hidden-page policy do not surface as user-facing failures.
- Real service errors continue to appear in the existing status and alert regions.

## Testing strategy

All implementation follows red-green-refactor.

### Backend tests

- A shared load followed by multiple inferences invokes the checkpoint loader once.
- A changed checkpoint identity or different model selection loads a distinct engine.
- Leases prevent eviction; expiry and LRU eviction dispose engines and clear attention state.
- Failed model construction leaves the cache consistent and releases capacity.
- A deliberately blocked binary or training filesystem operation does not delay an async heartbeat.
- Direct binary record lookup does not call `Path.read_bytes()` and reads a bounded byte range.
- First line lookup builds an index; repeated lookup avoids rescanning; file mutation invalidates the entry.
- FIFO compute tests prove arrival order, overload, timeout, and exception release.

### Node and browser tests

- Unchanged collaboration polls omit analysis and entries while keeping presence accurate.
- Hidden-page and failure timing use controlled fake timers or request timestamps.
- Health status covers checking, healthy latency, offline, abort, and recovery.
- A 2,000-row CSV page renders a bounded number of rows while selection/navigation still work.
- Canvas heatmap maps hover coordinates and highlights the selected token.
- Raw output initially renders a bounded preview and exposes the full-data actions.

### Full verification

Run with Python 3.11 and a supported Node runtime:

```bash
PYTHONPATH=backend:. python -m unittest discover -s backend -p 'test_*.py'
npx tsc --noEmit
node --test tests/sidecar-client.test.cjs tests/collaboration.spec.cjs
bash tests/test_startup.sh
npm run build
npx playwright test
```

Use isolated ports and a separate Next build directory. Do not restart or reuse the user's existing service for automated verification.

## Acceptance criteria

- A warmed shared workspace performs repeated inference without rebuilding the model.
- Heavy file requests no longer block unrelated FastAPI event-loop work.
- Direct binary record access is bounded and repeated line lookups use a valid persistent index.
- Collaboration preserves existing behavior while reducing unchanged response payloads and background traffic.
- Heavy frontend views have bounded DOM work relative to viewport size rather than dataset size.
- Personal model/result isolation and existing API semantics remain intact.
- All available backend, type, Node, startup, build, and browser suites pass in supported runtimes.
- Any verification unavailable in the current environment is reported explicitly and never represented as passing.

## Delivery sequence

Implementation is split into three reviewable milestones matching the stages above. Each milestone must leave the application runnable and independently testable. Stage 1 is implemented and verified before collaboration transport changes; Stage 2 before browser virtualization and visualization changes.
