const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

test('catalog lists artifacts across runs in one ordered query', () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'arch-catalog-test-'));
  const source = fs.readFileSync(path.join(__dirname, '../lib/catalog.ts'), 'utf8');
  const output = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true } }).outputText;
  const context = { exports: {}, require, process: { cwd: () => temporary, env: process.env } };
  vm.runInNewContext(output, context);
  const catalog = context.exports;
  const db = catalog.getDb();
  try {
    db.prepare('INSERT INTO runs(run_id, created_at) VALUES (?, ?)').run('older', 1);
    db.prepare('INSERT INTO runs(run_id, created_at) VALUES (?, ?)').run('newer', 2);
    const insert = db.prepare('INSERT INTO artifacts(artifact_id,run_id,kind,rel_path,abs_path,mtime,size,fingerprint) VALUES(?,?,?,?,?,?,?,?)');
    insert.run('a1','older','prediction','old.csv','/old.csv',1,1,'old');
    insert.run('a2','newer','training_history','new.csv','/new.csv',2,1,'new');
    const rows = catalog.listRunArtifacts();
    assert.deepEqual(Array.from(rows, r => r.artifact_id), ['a2', 'a1']);
    assert.equal(rows[0].run_id, 'newer');
  } finally { db.close(); fs.rmSync(temporary, { recursive: true, force: true }); }
});
