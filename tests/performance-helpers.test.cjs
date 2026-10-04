const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('typescript');

function loadHelpers() {
  const filename = require('node:path').join(__dirname, '../lib/performance.ts');
  const source = fs.readFileSync(filename, 'utf8');
  const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } });
  const context = { exports: {}, require, setTimeout, clearTimeout };
  vm.runInNewContext(outputText, context);
  return context.exports;
}

test('bounded mapper preserves input order and caps simultaneous work', async () => {
  const { mapWithConcurrency } = loadHelpers();
  let active = 0, peak = 0;
  const result = await mapWithConcurrency([1, 2, 3, 4, 5], 2, async value => {
    peak = Math.max(peak, ++active);
    await new Promise(resolve => setTimeout(resolve, 6 - value));
    active--;
    return value * 10;
  });
  assert.deepEqual(Array.from(result), [10, 20, 30, 40, 50]);
  assert.equal(peak, 2);
});

test('bounded mapper rejects invalid concurrency and propagates failures', async () => {
  const { mapWithConcurrency } = loadHelpers();
  await assert.rejects(mapWithConcurrency([1], 0, async x => x), /positive/);
  await assert.rejects(mapWithConcurrency([1, 2], 2, async x => { throw new Error('read failed'); }), /read failed/);
});

test('storage sizing distinguishes metadata from large parsed data without serializing it', () => {
  const { exceedsStorageBudget } = loadHelpers();
  assert.equal(exceedsStorageBudget([{ name: 'tiny', data: [1, 2, 3] }]), false);
  const large = { values: Array.from({ length: 10000 }, () => 1) };
  assert.equal(exceedsStorageBudget(large), true);
  assert.equal(exceedsStorageBudget({ text: 'x'.repeat(40000) }), true);
});
