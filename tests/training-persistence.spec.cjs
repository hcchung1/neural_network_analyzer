const { test, expect } = require('@playwright/test');
const ts = require('typescript');
const fs = require('node:fs');
const path = require('node:path');

test('large training results use IndexedDB and legacy small results still restore', async ({ page }) => {
  await page.goto(process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:33107');
  const source = fs.readFileSync(path.join(__dirname, '../lib/training-results-storage.ts'), 'utf8');
  const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
  const sizing = ts.transpileModule(fs.readFileSync(path.join(__dirname, '../lib/performance.ts'), 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
  await page.evaluate(({ compiled, sizing }) => {
    const performanceExports = {};
    new Function('exports', sizing)(performanceExports);
    window.trainingStorage = {};
    new Function('exports', 'require', compiled)(window.trainingStorage, () => performanceExports);
  }, { compiled, sizing });
  const result = await page.evaluate(async () => {
    const key = 'training-performance-test';
    const small = [{ id: 'legacy', values: [1, 2, 3] }];
    localStorage.setItem(key, JSON.stringify(small));
    const restoredLegacy = await window.trainingStorage.readTrainingResults(key);
    const large = [{ id: 'large', values: Array.from({ length: 20000 }, (_, i) => i) }];
    await window.trainingStorage.writeTrainingResults(key, large);
    const marker = localStorage.getItem(key);
    const restoredLarge = await window.trainingStorage.readTrainingResults(key);
    await window.trainingStorage.writeTrainingResults(key, small);
    return { restoredLegacy, markerSize: marker.length, largeCount: restoredLarge[0].values.length,
      largeLast: restoredLarge[0].values.at(-1), restoredSmall: await window.trainingStorage.readTrainingResults(key) };
  });
  expect(result.restoredLegacy[0].id).toBe('legacy');
  expect(result.markerSize).toBeLessThan(200);
  expect(result.largeCount).toBe(20000);
  expect(result.largeLast).toBe(19999);
  expect(result.restoredSmall[0].values).toEqual([1, 2, 3]);
});

test('failed training restore keeps the saved pointer instead of overwriting it with empty results', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('training_results', JSON.stringify({ version: 2, storage: 'indexeddb' })));
  await page.goto(`${process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:33107'}/runs`);
  await expect(page.getByText(/Unable to restore training results/)).toBeVisible();
  // Wait past the persistence debounce to catch an accidental empty save.
  await page.waitForTimeout(750);
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('training_results'))))
    .toEqual({ version: 2, storage: 'indexeddb' });
});

test('folder import waits for delayed IndexedDB restoration and preserves both sets of results', async ({ page }) => {
  const base = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:33107';
  await page.goto(base);
  await page.evaluate(async () => {
    const request = indexedDB.open('archanalyzer-training-results', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('results');
    const db = await new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = reject; });
    const transaction = db.transaction('results', 'readwrite');
    transaction.objectStore('results').put([{ id: 'prior', name: 'Prior', color: '#ccc', copyColor: '#ccc',
      folderPath: '/prior', visible: false, data: null, phaseData: null, randomPhaseData: null,
      turnMetrics: null, randomTurnMetrics: null }], 'training_results');
    await new Promise((resolve, reject) => { transaction.oncomplete = resolve; transaction.onerror = reject; });
    db.close();
    localStorage.setItem('training_results', JSON.stringify({ version: 2, storage: 'indexeddb' }));
  });
  await page.addInitScript(() => {
    const original = IDBDatabase.prototype.transaction;
    IDBDatabase.prototype.transaction = function(...args) {
      const transaction = original.apply(this, args);
      if (args[1] !== 'readonly') return transaction;
      return new Proxy(transaction, {
        get(target, key) { const value = target[key]; return typeof value === 'function' ? value.bind(target) : value; },
        set(target, key, value) {
          target[key] = key === 'oncomplete' ? event => setTimeout(() => value(event), 600) : value;
          return true;
        },
      });
    };
  });
  await page.route('**/api/legacy/training/scan', route => route.fulfill({ json: { success: true,
    folder_path: '/new', files: [{ name: 'new_history.csv', path: '/new/new_history.csv', size: 45 }] } }));
  await page.route('**/api/legacy/training/read_file', route => route.fulfill({ json: { success: true,
    content: '[TRAINING_HISTORY]\nepoch,train_loss,val_loss\n1,0.5,0.4\n' } }));
  await page.goto(`${base}/runs?folder=%2Fnew`);
  await expect.poll(() => page.evaluate(() => {
    const saved = JSON.parse(localStorage.getItem('training_results') || '[]');
    return Array.isArray(saved) ? saved.map(result => result.folderPath).sort() : [];
  })).toEqual(['/new', '/prior']);
});


test('failed restore blocks URL-triggered import and retains the saved pointer', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('training_results', JSON.stringify({ version: 2, storage: 'indexeddb' })));
  let scans = 0;
  await page.route('**/api/legacy/training/scan', route => {
    scans++;
    return route.fulfill({ json: { success: true, folder_path: '/new',
      files: [{ name: 'new_history.csv', path: '/new/new_history.csv', size: 45 }] } });
  });
  await page.route('**/api/legacy/training/read_file', route => route.fulfill({ json: { success: true,
    content: '[TRAINING_HISTORY]\nepoch,train_loss,val_loss\n1,0.5,0.4\n' } }));
  await page.goto(`${process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:33107'}/runs?folder=%2Fnew`);
  await expect(page.getByText(/Unable to restore training results/)).toBeVisible();
  await page.waitForTimeout(750);
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('training_results'))))
    .toEqual({ version: 2, storage: 'indexeddb' });
  expect(scans).toBe(0);
});
