const { test, expect } = require('@playwright/test');
const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:3001';

async function inferenceFixture(page, attention = [[0.1, 0.2, 0.3], [0.4, 0.5]], output = Array.from({length: 5000}, (_, i) => i)) {
  await page.route('**/api/v1/health', r => r.fulfill({json: {status: 'ok'}}));
  await page.route('**/api/v1/models/load', r => r.fulfill({json: {
    status: 'ok', model_handle: 'performance-fixture', model_name: 'Fixture', device: 'cpu',
    capabilities: {}, input_schema: {seq_len: 2, feature_dim: 3}, layer_count: 1,
  }}));
  await page.route('**/api/legacy/binary_samples/files?*', r => r.fulfill({json: {success: true, files: [{name: 'fixture.bin', size: 100}]}}));
  await page.route('**/api/legacy/binary_samples/search', r => r.fulfill({json: {
    success: true, compatible: true, feature: [[0,0,0],[0,0,0]], shape: [2,3], valid_len: 2,
    file_name: 'fixture.bin', record_index: 0,
  }}));
  await page.route('**/api/v1/inference-sessions', r => r.fulfill({json: {session_id: 'result', output_sample: output, output_shape: [1, output.length]}}));
  await page.route('**/api/v1/inference-sessions/result/activations', r => r.fulfill({json: {activations: {attention_layer_0: attention}}}));
  await page.goto(`${baseURL}/explore`);
  await page.getByRole('button', {name: 'Load Model', exact: true}).click();
  await page.getByRole('button', {name: 'Load Sample', exact: true}).click();
  await expect(page.getByText('Raw Model Output', {exact: true})).toBeVisible();
}

test('health starts checking, reports offline and recovers with measured RTT on visibility', async ({page}) => {
  let release;
  const initialGate = new Promise(resolve => { release = resolve; });
  let phase = 'checking';
  await page.route('**/api/v1/health', async r => {
    if (phase === 'checking') await initialGate;
    await r.fulfill({status: phase === 'offline' ? 503 : 200, json: {status: phase === 'offline' ? 'offline' : 'ok'}});
  });
  await page.goto(`${baseURL}/explore`);
  const status = page.getByRole('status', {name: 'Sidecar health'});
  await expect(status).toHaveText('Checking');
  // React development remounts can abort a first request; block every initial
  // request so status assertions do not depend on mount count.
  phase = 'offline';
  release();
  await expect(status).toHaveText('Offline');
  phase = 'healthy';
  await page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')));
  await expect(status).toHaveText(/Healthy · \d+ ms/);
});

test('2000 CSV rows stay bounded and navigate to an offscreen prediction with exact copy', async ({page}) => {
  const rows = Array.from({length: 2000}, (_, i) => ['same-id', String(i+1), `https://example.test/${i+1}`, `value-${i+1}-${'x'.repeat(180)}`]);
  const headers = ['ids', 'line_number', 'tenhou_link', 'value'];
  await page.addInitScript(() => {window.copied = ''; Object.defineProperty(navigator, 'clipboard', {value: {writeText: async text => {window.copied = text;}}, configurable: true});});
  await page.route('https://example.test/**', r => r.fulfill({body: '<p>Prediction</p>', contentType: 'text/html'}));
  await page.route('**/api/legacy/csv_reader/output_index/status', r => r.fulfill({json: {success: true, paths: [], indexed_count: 0, last_scanned_at: null}}));
  await page.route('**/api/legacy/csv_reader/open', r => r.fulfill({json: {success: true, session_id: 'rows', display_name: 'rows.csv', display_path: '/test/rows.csv', headers, images: [], model_paths: [], training_folder: null}}));
  await page.route('**/api/legacy/csv_reader/page', r => r.fulfill({json: {success: true, headers, rows, page: 0, page_size: 2000, start_row: 1, end_row: 2000, has_previous: false, has_next: false, total_rows: 2000, scanned_rows: 2000, matched_rows: 2000}}));
  await page.route('**/api/legacy/csv_reader/neighbors', r => r.fulfill({json: {success: true, previous: null, next: {source_row_number: 1901, page: 0, row_index: 1900, row: rows[1900], tenhou_link: rows[1900][2]}}}));
  await page.goto(`${baseURL}/data`);
  await page.locator('select').first().selectOption('2000');
  await page.getByPlaceholder('輸入 server 上的 .csv 或 .zip 路徑').fill('/test/rows.csv');
  await page.getByRole('button', {name: 'Open CSV/ZIP'}).click();
  await expect(page.locator('[data-csv-row="0"][data-csv-column="2"]')).toBeVisible();
  expect(await page.locator('tbody tr').count()).toBeLessThan(100);
  await page.locator('[data-csv-row="0"][data-csv-column="2"]').dblclick();
  await page.getByRole('button', {name: '下一次預測 →'}).click();
  const target = page.locator('[data-csv-row="1900"][data-csv-column="2"]');
  await expect(target).toBeVisible();
  expect(await page.locator('tbody tr').count()).toBeLessThan(100);
  await page.getByRole('button', {name: 'Copy Cell', exact: true}).click();
  await expect.poll(() => page.evaluate(() => window.copied)).toBe(rows[1900][2]);
  await page.getByRole('button', {name: 'Copy Row As CSV / Ctrl+C'}).click();
  await expect.poll(() => page.evaluate(() => window.copied)).toBe(`${headers.join(',')}\n${rows[1900].join(',')}`);
  const separator = page.getByRole('separator', {name: 'Resize side panel'});
  await expect(separator).toBeVisible();
  const before = Number(await separator.getAttribute('aria-valuenow'));
  const bounds = await separator.boundingBox();
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.mouse.down();
  await page.mouse.move(bounds.x - 60, bounds.y + bounds.height / 2);
  await page.mouse.up();
  await expect.poll(async () => Number(await separator.getAttribute('aria-valuenow'))).toBeGreaterThan(before);
  const viewport = page.getByTestId('csv-viewport');
  const horizontal = await viewport.evaluate(container => {
    container.scrollLeft = container.scrollWidth;
    return {left: container.scrollLeft, maximum: container.scrollWidth - container.clientWidth};
  });
  expect(horizontal.maximum).toBeGreaterThan(0);
  expect(horizontal.left).toBeGreaterThan(0);
  await page.getByRole('button', {name: 'Copy Cell', exact: true}).click();
  await expect.poll(() => page.evaluate(() => window.copied)).toBe(rows[1900][2]);
  expect(await page.locator('tbody tr').count()).toBeLessThan(100);
});

test('canvas hover preserves non-square values and selected token highlights', async ({page}) => {
  await inferenceFixture(page);
  const canvas = page.getByRole('img', {name: 'Attention heatmap'});
  await expect(canvas).toBeVisible();
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 4);
  await expect(page.getByRole('tooltip')).toHaveText('Token 0 → 1: 0.2000');
  // Selected column has a blue overlay, while an unselected cell has the orange heatmap.
  const pixels = await canvas.evaluate(c => {
    const context = c.getContext('2d');
    return [Array.from(context.getImageData(0, Math.floor(c.height*0.75), 1, 1).data),
      Array.from(context.getImageData(Math.floor(c.width*0.5), Math.floor(c.height*0.75), 1, 1).data)];
  });
  expect(pixels[0][0]).toBeLessThan(255);
  expect(pixels[1][0]).toBe(255);
  await page.getByLabel('Token Index', {exact: true}).selectOption('1');
  await expect(canvas).toHaveAttribute('data-selected-token', '1');
  const updated = await canvas.evaluate(c => {
    const context = c.getContext('2d');
    return [context.getImageData(0, 0, 1, 1).data[0], context.getImageData(1, 0, 1, 1).data[0], context.getImageData(2, 1, 1, 1).data[0]];
  });
  expect(updated[0]).toBe(255);
  expect(updated[1]).toBeLessThan(255);
  expect(updated[2]).toBeLessThan(255);
});

test('raw output preview is bounded and full JSON remains expandable, copyable and downloadable', async ({page}) => {
  const output = Array.from({length: 5000}, (_, i) => i);
  await page.addInitScript(() => Object.defineProperty(navigator, 'clipboard', {value: {writeText: async text => {window.copied = text;}}, configurable: true}));
  await inferenceFixture(page, [[1,0],[0,1]], output);
  await expect(page.getByText('Shape: [1, 5000] · 5000 values', {exact: true})).toBeVisible();
  const raw = page.getByTestId('raw-output');
  expect((await raw.textContent()).length).toBeLessThan(1200);
  await page.getByRole('button', {name: 'Copy full JSON'}).click();
  await expect.poll(() => page.evaluate(() => window.copied)).toBe(JSON.stringify(output));
  const download = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Download JSON'}).click();
  expect((await download).suggestedFilename()).toBe('model-output.json');
  await page.getByRole('button', {name: 'Show full output'}).click();
  await expect(raw).toHaveText(JSON.stringify(output));
  await page.getByRole('button', {name: 'Show preview'}).click();
  expect((await raw.textContent()).length).toBeLessThan(1200);
});

test('hover safely clears when the next inference has a smaller attention matrix', async ({page}) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await inferenceFixture(page);
  const canvas = page.getByRole('img', {name: 'Attention heatmap'});
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width * 0.8, box.y + box.height * 0.75);
  await expect(page.getByRole('tooltip')).toHaveText('Token 1 → 2: 0.0000');
  await page.route('**/api/v1/inference-sessions/result/activations', r => r.fulfill({json: {activations: {attention_layer_0: [[1]]}}}));
  // Trigger inference without moving the pointer off the old hovered matrix.
  await page.getByRole('button', {name: 'Load Sample', exact: true}).evaluate(button => button.click());
  await expect(canvas).toHaveAttribute('width', '1');
  await expect(canvas).toHaveAttribute('height', '1');
  await expect(page.getByRole('tooltip')).toHaveCount(0);
  expect(errors).toEqual([]);
});
