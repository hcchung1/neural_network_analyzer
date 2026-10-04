const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const Database = require('better-sqlite3');

function loadModule(file, globals = {}) {
  const source = fs.readFileSync(path.join(__dirname, '..', file), 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  });
  const context = { exports: {}, require, process, AbortController, setTimeout, clearTimeout, ...globals };
  vm.runInNewContext(outputText, context, { filename: file });
  return context.exports;
}
function storage() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'collaboration-polling-'));
  let now = 100000;
  class Clock extends Date { static now() { return now; } }
  const api = loadModule('lib/collaboration.ts', { Date: Clock, process: { ...process, cwd: () => directory } });
  const joined = api.joinRoom('Host');
  const database = new Database(path.join(directory, 'data/collaboration.sqlite3'));
  database.exec('CREATE TABLE presence_writes (seen INTEGER); CREATE TRIGGER record_presence AFTER INSERT ON collaboration_presence BEGIN INSERT INTO presence_writes VALUES (NEW.seen); END;');
  return { api, joined, database, at: value => { now = value; } };
}

test('unchanged polls are small heartbeats and independent analysis revisions still deliver updates', () => {
  const { api, joined, database } = storage();
  const { roomId, token } = joined.identity;
  const reply = api.getRoom(roomId, token, { revision: joined.room.revision, analysisRevision: 0 });
  assert.equal(reply.unchanged, true);
  assert.equal(reply.id, roomId);
  assert.equal(reply.members[0].online, true);
  for (const field of ['entries', 'view', 'analysis']) assert.equal(Object.hasOwn(reply, field), false, field);
  const changed = api.updateRoom(roomId, token, { type: 'chat', text: 'Hello' });
  const delta = api.getRoom(roomId, token, { revision: joined.room.revision, analysisRevision: 0 });
  assert.equal(delta.entries[0].text, 'Hello');
  assert.equal(delta.analysis, undefined);
  const job = api.beginAnalysis(roomId, token, 0);
  const analysis = { tokenCount: 2, maxLayer: 0, attention: {}, output: [7] };
  const analyzed = api.finishAnalysis(roomId, job.jobId, analysis);
  const analysisOnly = api.getRoom(roomId, token, { revision: analyzed.revision, analysisRevision: 0 });
  assert.equal(analysisOnly.unchanged, true);
  assert.deepEqual(Array.from(analysisOnly.analysis.output), [7]);
  assert.equal(analysisOnly.entries, undefined);
  assert.equal(api.getRoom(roomId, token).analysis.output[0], 7, 'legacy callers get a complete snapshot');
  assert.ok(changed.revision > joined.room.revision);
  database.close();
});

test('frequent presence polls write at most every three seconds and immediately mark returning callers online', () => {
  const { api, joined, database, at } = storage();
  const poll = () => api.getRoom(joined.identity.roomId, joined.identity.token);
  for (const time of [100000, 100500, 101000, 102999]) { at(time); poll(); }
  assert.ok(database.prepare('SELECT count(*) AS count FROM presence_writes').get().count <= 1);
  at(103000); poll();
  at(103999); poll();
  const writes = database.prepare('SELECT seen FROM presence_writes ORDER BY seen').all();
  assert.ok(writes.length >= 1, 'presence is persisted after the throttle interval');
  for (let index = 1; index < writes.length; index++) assert.ok(writes[index].seen - writes[index - 1].seen >= 3000);
  at(120000);
  assert.equal(poll().members[0].online, true);
  assert.throws(() => api.getRoom(joined.identity.roomId, 'invalid'), error => error.status === 401);
  database.close();
});

function snapshot() {
  return { id: 'room', revision: 4, analysisRevision: 2, hostId: 'alice', view: { token: 1, layer: 0 },
    analysis: { tokenCount: 3, maxLayer: 0, attention: {}, output: [1, 2] }, entries: [], busy: false,
    members: [{ id: 'alice', name: 'Alice', online: true }] };
}
function helpers() {
  return loadModule('lib/workspace-polling.ts');
}
test('identical heartbeats preserve the snapshot and stable references, while observable membership changes render', () => {
  const { mergeRoomSnapshot } = helpers();
  const previous = snapshot();
  const heartbeat = { id: 'room', revision: 4, analysisRevision: 2, hostId: 'alice', busy: false,
    unchanged: true, members: [{ id: 'alice', name: 'Alice', online: true }] };
  assert.equal(mergeRoomSnapshot(previous, heartbeat), previous);
  const changed = mergeRoomSnapshot(previous, { ...heartbeat, members: [{ id: 'alice', name: 'Alice', online: false }] });
  assert.equal(changed.members[0].online, false);
  assert.equal(changed.entries, previous.entries);
  assert.equal(changed.view, previous.view);
  assert.equal(changed.analysis, previous.analysis);
  const busy = mergeRoomSnapshot(previous, { ...heartbeat, busy: true });
  assert.equal(busy.busy, true);
  assert.equal(busy.analysis, previous.analysis);
  assert.equal(mergeRoomSnapshot(previous, { ...snapshot(), revision: 3 }), previous);
  assert.equal(mergeRoomSnapshot(previous, JSON.parse(JSON.stringify(previous))), previous, 'legacy snapshots remain accepted');
  const updated = mergeRoomSnapshot(previous, { ...snapshot(), revision: 5, entries: [{ id: 'note' }], analysis: undefined });
  assert.equal(updated.analysis, previous.analysis);
  assert.equal(updated.entries[0].id, 'note');
});

const flush = () => new Promise(resolve => setImmediate(resolve));
function timers() {
  const tasks = new Map(); let sequence = 0;
  return {
    schedule: (callback, delay) => { const id = ++sequence; tasks.set(id, { callback, delay }); return id; },
    cancel: id => tasks.delete(id),
    next: () => { const entry = tasks.entries().next().value; assert.ok(entry, 'a next poll is scheduled'); tasks.delete(entry[0]); entry[1].callback(); return entry[1].delay; },
    delay: () => tasks.values().next().value?.delay,
    count: () => tasks.size,
  };
}
test('polling follows page visibility, backs off errors with a cap, and resets after success', async () => {
  const { startWorkspacePolling } = helpers();
  assert.equal(typeof startWorkspacePolling, 'function', 'a controlled polling lifecycle is available');
  const clock = timers(); let hidden = false; let failure = false; let attempts = 0; let errors = 0;
  const stop = startWorkspacePolling({
    poll: async () => { attempts++; if (failure) throw new Error('Unavailable'); },
    hidden: () => hidden, random: () => 0.5, onError: () => { errors++; },
    schedule: clock.schedule, cancel: clock.cancel,
  });
  await flush(); assert.equal(attempts, 1); assert.equal(clock.delay(), 1000);
  hidden = true; clock.next(); await flush(); assert.equal(clock.delay(), 5000);
  failure = true;
  for (const want of [2000, 4000, 8000, 15000, 15000]) {
    clock.next(); await flush(); assert.equal(clock.delay(), want);
  }
  assert.equal(errors, 5);
  failure = false; hidden = false; clock.next(); await flush(); assert.equal(clock.delay(), 1000);
  stop(); assert.equal(clock.count(), 0);
});

test('a pending poll never overlaps and stop aborts in-flight work without reporting navigation errors', async () => {
  const { startWorkspacePolling } = helpers();
  assert.equal(typeof startWorkspacePolling, 'function', 'a controlled polling lifecycle is available');
  const clock = timers(); let reject; let signal; let attempts = 0; let errors = 0;
  const stop = startWorkspacePolling({
    poll: requestSignal => { attempts++; signal = requestSignal; return new Promise((resolve, fail) => { reject = fail; }); },
    hidden: () => false, onError: () => { errors++; }, schedule: clock.schedule, cancel: clock.cancel,
  });
  await flush(); assert.equal(attempts, 1); assert.equal(clock.count(), 0);
  stop(); assert.equal(signal.aborted, true);
  reject(new Error('Abort')); await flush();
  assert.equal(errors, 0); assert.equal(clock.count(), 0); assert.equal(attempts, 1);
});

test('failure jitter spreads retry times and terminal authentication outcomes stop polling', async () => {
  const { startWorkspacePolling } = helpers();
  assert.equal(typeof startWorkspacePolling, 'function', 'a controlled polling lifecycle is available');
  const low = timers(); const high = timers();
  const stops = [0, 1].map((random, index) => startWorkspacePolling({
    poll: async () => { throw new Error('Unavailable'); }, random: () => random,
    schedule: [low, high][index].schedule, cancel: [low, high][index].cancel,
  }));
  await flush(); assert.ok(low.delay() < 2000); assert.ok(high.delay() > 2000);
  stops.forEach(stop => stop());
  const terminal = timers();
  startWorkspacePolling({ poll: async () => false, schedule: terminal.schedule, cancel: terminal.cancel });
  await flush(); assert.equal(terminal.count(), 0);
});

test('GET route reads both revisions and safely returns metadata for invalid revision parameters', async () => {
  const { api, joined, database } = storage();
  const { NextRequest } = require('next/server');
  const route = loadModule('app/api/v1/workspaces/[roomId]/route.ts', {
    require: name => name === '@/lib/collaboration' ? api
      : name === '@/lib/contracts' ? loadModule('lib/contracts.ts') : require(name),
  });
  const get = async query => {
    const request = new NextRequest(`http://localhost/api/v1/workspaces/${joined.identity.roomId}${query}`, {
      headers: { Authorization: `Bearer ${joined.identity.token}` },
    });
    return route.GET(request, { params: { roomId: joined.identity.roomId } });
  };
  const heartbeat = await get(`?revision=${joined.room.revision}&analysisRevision=0`);
  assert.equal(heartbeat.status, 200);
  assert.equal(heartbeat.headers.get('Cache-Control'), 'no-store');
  const data = await heartbeat.json();
  assert.equal(data.unchanged, true); assert.equal(data.entries, undefined); assert.equal(data.analysis, undefined);
  const malformed = await (await get('?revision=oops&analysisRevision=-1')).json();
  assert.deepEqual(malformed.entries, []); assert.ok(malformed.analysis);
  const unauthorized = await route.GET(new NextRequest('http://localhost/api/v1/workspaces/room'), { params: { roomId: joined.identity.roomId } });
  assert.equal(unauthorized.status, 401);
  assert.equal(unauthorized.headers.get("Retry-After"), null);
  database.close();
});

test('overloaded analysis POST retains retry advice and releases room busy state without replacing analysis', async () => {
  const { api, joined, database } = storage();
  const { NextRequest } = require('next/server');
  const route = loadModule('app/api/v1/workspaces/[roomId]/route.ts', {
    AbortSignal,
    fetch: async (url, options) => {
      assert.match(url, /\/collaboration\/analyze$/);
      assert.equal(options.method, 'POST');
      assert.equal(api.getRoom(joined.identity.roomId, joined.identity.token).busy, true);
      return Response.json({ detail: { error: { code: 'COMPUTE_BUSY', message: 'Compute queue is full' } } }, {
        status: 503, headers: { 'Retry-After': '2' },
      });
    },
    require: name => name === '@/lib/collaboration' ? api
      : name === '@/lib/contracts' ? loadModule('lib/contracts.ts') : require(name),
  });
  const request = new NextRequest(`http://localhost/api/v1/workspaces/${joined.identity.roomId}`, {
    method: 'POST', headers: { Authorization: `Bearer ${joined.identity.token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ type: 'load', model: { checkpoint_path: 'dummy', device: 'cpu' }, analysisRevision: 0 }),
  });
  const response = await route.POST(request, { params: { roomId: joined.identity.roomId } });
  assert.equal(response.status, 503);
  assert.equal((await response.json()).error, 'Compute queue is full');
  const after = api.getRoom(joined.identity.roomId, joined.identity.token);
  assert.equal(after.busy, false);
  assert.equal(after.analysisRevision, 0);
  assert.deepEqual(JSON.parse(JSON.stringify(after.analysis)), JSON.parse(JSON.stringify(joined.room.analysis)));
  assert.equal(response.headers.get('Retry-After'), '2');
  database.close();
});
