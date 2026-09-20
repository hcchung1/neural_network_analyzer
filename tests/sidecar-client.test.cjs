const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('typescript');

function loadClient(fetch) {
  const source = fs.readFileSync(require.resolve('../lib/sidecar-client.ts'), 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  });
  const context = { exports: {}, require, process, fetch };
  vm.runInNewContext(outputText, context);
  return context.exports.forwardToSidecar;
}

test('unreachable CSV backend returns a readable legacy error', async () => {
  const forward = loadClient(async () => { throw new Error('fetch failed'); });
  const response = await forward('/legacy/csv_reader/output_index/scan', 'POST');
  const data = await response.json();
  assert.equal(response.status, 503);
  assert.equal(typeof data.error, 'string');
  assert.equal(data.success, false);
  assert.match(data.error, /fetch failed/);
  assert.match(data.error, /Python/);
});

test('modern API keeps its structured unreachable error', async () => {
  const forward = loadClient(async () => { throw new Error('fetch failed'); });
  const response = await forward('/models/load', 'POST', {});
  const data = await response.json();
  assert.equal(response.status, 503);
  assert.equal(data.error.code, 'SIDECAR_UNREACHABLE');
});

test('successful scans preserve the backend payload', async () => {
  const payload = { success: true, indexed_count: 12, last_scanned_at: 1234 };
  const forward = loadClient(async () => Response.json(payload));
  const response = await forward('/legacy/csv_reader/output_index/scan', 'POST');
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), payload);
});
