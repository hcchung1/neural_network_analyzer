const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('typescript');
function load(name, extra = {}) {
  const path = `${__dirname}/../lib/${name}.ts`;
  const context = { exports: {}, AbortController, performance, setTimeout, clearTimeout, ...extra };
  if (fs.existsSync(path)) vm.runInNewContext(ts.transpileModule(fs.readFileSync(path, 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, context);
  return context.exports;
}
test('row window keeps original indexes and bounded overscan at the end of 2000 rows', () => {
  const { rowWindow } = load('performance-ui');
  assert.equal(typeof rowWindow, 'function', 'row viewport behavior is available');
  const result = rowWindow(2000, 59000, 600, 30, 8);
  assert.ok(result.end - result.start <= 38);
  assert.ok(result.start > 1900);
  assert.equal(result.before, result.start * 30);
  assert.equal(result.after, (2000 - result.end) * 30);
  assert.equal(rowWindow(0, 0, 600, 30, 8).end, 0);
});
test('attention aggregation preserves rectangular and ragged multi-head dimensions', () => {
  const { attentionMatrix } = load('performance-ui');
  assert.equal(typeof attentionMatrix, 'function');
  const matrix = attentionMatrix([[[1, 2, 3], [4]], [[3, 4], [6, 8, 10]]]);
  assert.equal(JSON.stringify(matrix.values), '[[2,3,1.5],[5,4,5]]');
  assert.equal(matrix.columns, 3);
  assert.equal(matrix.min, 1.5);
  assert.equal(matrix.max, 5);
  assert.equal(attentionMatrix([]).allZero, true);
  assert.equal(attentionMatrix([[0], []]).allZero, true);
  assert.equal(attentionMatrix(Array.from({length: 500}, () => Array(500).fill(0.1))).max, 0.1);
});
test('pointer mapping uses column count independently from row count', () => {
  const { attentionCell } = load('performance-ui');
  assert.equal(typeof attentionCell, 'function');
  assert.equal(JSON.stringify(attentionCell(150, 25, 300, 100, 2, 3)), '{"row":0,"column":1}');
  assert.equal(attentionCell(300, 25, 300, 100, 2, 3), null);
});
test('raw output preview bounds serialization without losing shape or full data', () => {
  const { outputPreview } = load('performance-ui');
  assert.equal(typeof outputPreview, 'function');
  const values = Array.from({ length: 5000 }, (_, i) => i);
  const result = outputPreview(values);
  assert.ok(result.length <= 100);
  assert.equal(result[0], 0);
  assert.equal(values.length, 5000);
});
test('health checks do not overlap, measure RTT, report failure, and abort on dispose', async () => {
  const { createHealthMonitor } = load('health-monitor');
  assert.equal(typeof createHealthMonitor, 'function');
  const reports = [];
  let resolveFetch;
  let calls = 0;
  let clock = 10;
  let signal;
  const monitor = createHealthMonitor({
    fetch: (_url, options) => { calls++; signal = options.signal; return new Promise(resolve => { resolveFetch = resolve; }); },
    now: () => clock, onChange: state => reports.push(state), isVisible: () => true,
    intervalMs: 100000,
  });
  const pending = monitor.check();
  await monitor.check();
  assert.equal(calls, 1);
  clock = 42;
  resolveFetch({ ok: true });
  await pending;
  assert.equal(reports.at(-1).status, 'healthy');
  assert.equal(reports.at(-1).rttMs, 32);
  const failed = monitor.check();
  resolveFetch({ ok: false });
  await failed;
  assert.equal(reports.at(-1).status, 'offline');
  monitor.check();
  const count = reports.length;
  monitor.dispose();
  assert.equal(signal.aborted, true);
  resolveFetch({ ok: true });
  await Promise.resolve(); await Promise.resolve();
  assert.equal(reports.length, count);
});
test('hidden health checks pause and resume without surfacing aborts', async () => {
  const { createHealthMonitor } = load('health-monitor');
  assert.equal(typeof createHealthMonitor, 'function');
  let visible = false;
  const reports = [];
  let calls = 0;
  const monitor = createHealthMonitor({ fetch: async () => { calls++; return {ok: true}; }, now: () => 1,
    onChange: value => reports.push(value), isVisible: () => visible, intervalMs: 100000 });
  await monitor.check();
  assert.equal(calls, 0);
  visible = true;
  await monitor.visibilityChanged();
  assert.equal(calls, 1);
  assert.equal(reports.at(-1).status, 'healthy');
  monitor.dispose();
});
test('health polls only at its visible interval and cancels timers on hide and dispose', async () => {
  let nextId = 0;
  const timers = new Map();
  const { createHealthMonitor } = load('health-monitor', {
    setTimeout: (callback, delay) => { const id = ++nextId; timers.set(id, {callback, delay}); return id; },
    clearTimeout: id => timers.delete(id),
  });
  let visible = true;
  let calls = 0;
  const monitor = createHealthMonitor({fetch: async () => {calls++; return {ok: true};}, now: () => 1,
    onChange: () => {}, isVisible: () => visible});
  await monitor.check();
  assert.equal(calls, 1);
  assert.equal(timers.size, 1);
  assert.equal([...timers.values()][0].delay, 30000);
  const [id, scheduled] = [...timers][0];
  timers.delete(id); scheduled.callback();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(calls, 2);
  visible = false; monitor.visibilityChanged();
  assert.equal(timers.size, 0);
  visible = true; await monitor.visibilityChanged();
  assert.equal(calls, 3);
  monitor.dispose();
  assert.equal(timers.size, 0);
});
test('a health deadline aborts a stalled request and reports offline', async () => {
  const timers = new Map();
  let nextId = 0;
  const { createHealthMonitor } = load('health-monitor', {
    setTimeout: (callback, delay) => {const id = ++nextId; timers.set(id, {callback, delay}); return id;},
    clearTimeout: id => timers.delete(id),
  });
  const reports = [];
  let signal;
  const monitor = createHealthMonitor({fetch: (_url, options) => new Promise((_resolve, reject) => {
    signal = options.signal; signal.addEventListener('abort', () => reject(new Error('AbortError')));
  }), now: () => 0, onChange: state => reports.push(state), isVisible: () => true});
  const pending = monitor.check();
  const deadline = [...timers.values()].find(timer => timer.delay === 10000);
  deadline.callback();
  await pending;
  assert.equal(signal.aborted, true);
  assert.equal(reports.at(-1).status, 'offline');
  monitor.dispose();
});
