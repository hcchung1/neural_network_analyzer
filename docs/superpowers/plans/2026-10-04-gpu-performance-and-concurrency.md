# GPU performance and concurrency implementation plan

> **For agentic workers:** Use the existing performance design and execute the three independently testable milestones below with test-first changes and a final integration review.

**Goal:** Reduce single-user UI latency and multi-user contention while retaining personal model ownership and safe, serialized GPU inference.

**Architecture:** Keep GPU models resident in a bounded shared cache; perform blocking file work in worker threads and use persistent binary indexes. Reduce collaboration traffic with heartbeat polling and FIFO compute admission, and bound browser work with row virtualization, canvas attention, and output previews.

**Tech Stack:** Python 3.11, PyTorch, FastAPI, SQLite, Next.js, React, TypeScript, Playwright. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-10-04-performance-and-concurrency-design.md`

## Global constraints

- Preserve the current Next.js App Router plus single-process FastAPI deployment.
- Preserve personal `model_handle` ownership and result ownership checks.
- Preserve one running heavyweight PyTorch job; no mixed precision or batching changes to numerical semantics.
- Preserve successful API fields and existing uncommitted checkout work.
- Add no infrastructure service or runtime dependency.
- Python 3.11 is required; Node 18+ is required.
- Do not restart the user's service; verification uses isolated ports and build output.
- Deliver one new commit including this plan, implementation, tests and necessary ownership prerequisites. Unrelated pre-existing changes stay uncommitted.

## Review focus

1. Failed cache construction or capacity rejection preserves valid selections; leases prevent disposal.
2. Checkpoint mutation invalidates cache and binary file indexes; corrupt or truncated files never publish partial indexes.
3. Timeout and exceptions remove queued jobs and permit the next FIFO waiter to proceed.
4. Hidden pages, aborts and unchanged heartbeats do not repeatedly render or surface navigation errors.
5. Virtualized rows still support selection, copying and navigating to offscreen predictions; full raw output remains available on demand.

### Task 1: Backend and GPU resource foundation

**Files:** `backend/model_engine/resources.py`, `backend/model_engine/transformer_model.py`, `backend/api/collaboration.py`, `backend/api/sidecar.py`, `backend/api/binary_samples.py`, new `backend/data/binary_index.py`, `backend/api/training.py`, `backend/main.py`, `backend/test_performance.py`.

**Interfaces:** `SharedModelCache.lease(selection, builder)` yields an entry with engine and immutable metadata; `cleanup_expired()` and `close()` own its lifecycle. `get_attention_tensor(layer_idx)` exposes captured tensors; compatibility `get_attention_weights()` remains. `ComputeGate.slot()` grants FIFO admission. `BinaryRecordIndex` returns bounded bytes for record or line lookups.

- [x] Write failing tests for repeated collaboration reuse, distinct checkpoint identity, leased eviction, expiry, failed builds, FIFO/timeout/overload, event-loop responsiveness and bounded binary reads/index invalidation.
- [x] Run `PYTHONPATH=backend:. python3 -m unittest backend.test_performance`; observe the missing behavior.
- [x] Implement canonical cache keys, TTL 300s, count 2, byte budget 2048 MiB; keep model weights on the requested device and clear request attention in finally.
- [x] Use tensor reduction without list round trips; preserve response values.
- [x] Use sync file handlers and atomic SQLite/mmap binary index, including info.txt offsets when applicable.
- [x] Verify new tests and full backend suite; keep GPU access limitations explicit.

### Task 2: Collaboration polling

**Files:** `lib/collaboration.ts`, `lib/collaboration-types.ts`, `features/collaboration/useWorkspace.ts`, `app/api/v1/workspaces/[roomId]/route.ts`, new polling helpers and Node/browser tests.

**Interfaces:** GET accepts room `revision` plus `analysisRevision`; unchanged replies omit stable fields. Heartbeat merges retain stable references. Visible polling is about 1s, hidden 5s, errors back off from about 2s to 15s with jitter.

- [x] Write failing behavior tests for small heartbeat replies, throttled presence, unchanged merge and polling delays.
- [x] Run Node tests and observe failures.
- [x] Implement delta responses and presence throttling at 3s, immediate online status for caller, no overlapping polls, abort cleanup and controlled backoff.
- [x] Verify Node tests and collaboration browser flow.

### Task 3: Bounded browser work and request reduction

**Files:** `components/Navigation.tsx`, new health hook, `features/sources/CsvReaderComponent.tsx`, `components/visualizations/AttentionHeatmap.tsx`, `components/visualizations/OutputView.tsx`, `features/inference/TrainingResultsComponent.tsx`, persistence/concurrency helpers, `features/sources/DataSourceLibrary.tsx`, `lib/catalog.ts`, `app/api/v1/runs/route.ts`, new performance browser tests.

**Interfaces:** Row viewport renders visible rows plus overscan with selection indexes in the original page. Canvas maps pointer position to matrix cells. Raw output defaults to a bounded preview, with full expansion/copy/download. Health reports checking/healthy measured RTT/offline without overlapping requests. Catalog returns artifacts in one additive request. Training files use bounded concurrency and debounced IndexedDB persistence for large parsed results.

- [x] Write failing browser/helper tests for a 2000-row page, offscreen selection navigation, canvas hover, bounded output and health recovery.
- [x] Implement viewport rows, canvas rendering and bounded output with existing data semantics.
- [x] Implement health lifecycle, one-query artifact catalog, bounded file fetching and debounced persistent storage with legacy restore.
- [x] Verify browser tests and `npx tsc --noEmit`.

### Integration, runtime and delivery

**Files:** `start.sh`, `package.json`, `tests/test_startup.sh`, README performance notes and verification record.

- [x] Replace the existing placeholder startup test with real controlled runtime-preflight tests; reject Python older than 3.11 and document Node floor.
- [x] Verify `PYTHONPATH=backend:. python3 -m unittest discover -s backend -p 'test_*.py'`, `npx tsc --noEmit`, Node unit tests, startup tests, production build and Playwright on separate ports.
- [x] Obtain independent review and resolve material findings.
- [x] Assemble a self-contained commit containing feature edits plus necessary model ownership prerequisites, excluding unrelated pre-existing UI/data changes.
- [x] Prepare verified feature edits for integration into the original checkout, preserving remaining uncommitted files; deliver the authorized commit and report verification limitations.

## Execution rulings

- The user explicitly requested planning, modification and a new commit after reviewing the existing design; proceed without repeated approval handoffs.
- Isolation uses a local `/tmp` clone with the current checkout overlaid and a separate baseline snapshot. The main checkout is unchanged until integration.
- Backend, polling and UI tasks have separate file ownership and can be implemented concurrently under the dispatching-parallel-agents workflow. Milestone verification and final delivery remain ordered.
- GPU kernels cannot be benchmarked in the current execution environment; CPU models test lifecycle, isolation and numerical contracts. No GPU performance claim will be made without measurements.

## Verification record (2026-10-05 UTC)

The self-contained delivery tree was verified against its own production build using Next.js port 33108 and FastAPI port 18108. No user service was restarted.

- Backend discovery: **52 tests passed**, including 18 new performance tests. The binary truncation reproduction now exits safely; corrupted in-range text offsets and binary fallback mutation are covered.
- Node test runner: all **5 test files passed** (sidecar ownership, polling, sizing/concurrency, UI helpers and real SQLite artifact query).
- Startup preflight: **3 tests passed**; Python version and missing-interpreter errors occur before service startup.
- `npx tsc --noEmit`: passed; `npm run build` with isolated `NEXT_DIST_DIR`: passed.
- Exact production-tree Playwright: **14 tests passed**. Real multi-browser collaboration uses actual CPU PyTorch inference; browser fixtures cover 2000-row virtualization, offscreen navigation/copy, canvas hover and matrix shrink, output preview/full copy/download, health recovery, IndexedDB migration and restoration races.
- Independent review findings repaired: mmap source truncation, semantically corrupt text offsets, fallback record mutation, successful/failed restoration racing URL-triggered imports, and missing collaboration overload Retry-After.

All production numerical execution remains in the original precision. This environment has no usable CUDA device; GPU latency, peak VRAM and real-user p95 load targets remain deployment measurements. Cache byte limits account for retained parameters/buffers, not transient construction or inference allocations.

Necessary pre-existing personal model/result ownership code is included to make the global compute gate safe and the commit independently runnable. Unrelated CSV lifecycle, model compatibility inference, Assistant redesign and other UI changes are kept outside the commit. Original files were checked byte-for-byte against the initial snapshot before integration.

Additional integration check: the checkout overlay, including the user’s unrelated uncommitted functionality, passed all **37 Playwright tests** after using reachable isolated test services. The initial sandbox-network run was rejected as invalid setup evidence and was rerun.
