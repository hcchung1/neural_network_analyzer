const { test, expect } = require('@playwright/test');
const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:3001';

async function fixture(page) {
  const requests = [];
  await page.route('**/api/legacy/binary_samples/files?*', r => r.fulfill({ json: {
    success: true, files: [{ name: 'fixture.bin', size: 100 }],
  } }));
  await page.route('**/api/legacy/binary_samples/search', r => {
    requests.push(r.request().postDataJSON());
    return r.fulfill({ json: { success: true, compatible: true, feature: [[0, 0, 0], [0, 0, 0]],
      shape: [2, 3], valid_len: 2, file_name: 'fixture.bin', record_index: 0 } });
  });
  return requests;
}

test('personal schema and capability travel through load, inference, read and release', async ({ page }) => {
  const samples = await fixture(page);
  const headers = [];
  await page.route('**/api/v1/models/load', r => r.fulfill({ json: {
    status: 'ok', model_handle: 'alice-secret', model_name: 'Alice', device: 'cpu', capabilities: {},
    input_schema: { seq_len: 2, feature_dim: 3 }, layer_count: 1,
  } }));
  await page.route('**/api/v1/inference-sessions', r => {
    headers.push(r.request().headers()['x-model-handle']);
    return r.fulfill({ json: { session_id: 'alice-result', output_sample: [1, -1], output_shape: [1, 2] } });
  });
  await page.route('**/api/v1/inference-sessions/alice-result/activations', r => {
    headers.push(r.request().headers()['x-model-handle']);
    return r.fulfill({ json: { activations: { attention_layer_0: [[1, 0], [0, 1]] } } });
  });
  let released;
  await page.route('**/api/v1/models/current', r => {
    released = r.request().headers()['x-model-handle'];
    return r.fulfill({ json: { status: 'deleted' } });
  });
  await page.goto(`${baseURL}/explore`);
  await page.getByRole('button', { name: 'Load Model', exact: true }).click();
  await expect(page.getByLabel('Token Index', { exact: true }).locator('option')).toHaveCount(2);
  await page.getByRole('button', { name: 'Load Sample', exact: true }).click();
  await expect.poll(() => headers.length).toBe(2);
  expect(samples[0]).toMatchObject({ seq_len: 2, feature_dim: 3 });
  expect(headers).toEqual(['alice-secret', 'alice-secret']);
  await page.getByRole('button', { name: 'Release Model', exact: true }).click();
  await expect.poll(() => released).toBe('alice-secret');
  await expect(page.getByRole('button', { name: 'Load Sample', exact: true })).toHaveCount(0);
});

test('expired personal model clears the handle and allows a fresh load', async ({ page }) => {
  await fixture(page);
  const loadHeaders = [];
  await page.route('**/api/v1/models/load', r => {
    loadHeaders.push(r.request().headers()['x-model-handle']);
    return r.fulfill({ json: { status: 'ok', model_handle: 'expired-model', model_name: 'Alice',
      device: 'cpu', capabilities: {}, input_schema: { seq_len: 2, feature_dim: 3 }, layer_count: 1 } });
  });
  await page.route('**/api/v1/inference-sessions', r => r.fulfill({ status: 404,
    json: { error: { code: 'MODEL_EXPIRED', message: '模型已到期，請重新載入。' } } }));
  await page.goto(`${baseURL}/explore`);
  await page.getByRole('button', { name: 'Load Model', exact: true }).click();
  await page.getByRole('button', { name: 'Load Sample', exact: true }).click();
  await expect(page.getByText('模型已到期，請重新載入。', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Load Sample', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Load Model', exact: true }).click();
  await expect.poll(() => loadHeaders.length).toBe(2);
  expect(loadHeaders).toEqual([undefined, undefined]);
});

test('late personal load cannot replace a newly joined workspace view', async ({ page }) => {
  let finishLoad;
  let released;
  await page.route('**/api/v1/models/load', async route => {
    await new Promise(resolve => { finishLoad = resolve; });
    await route.fulfill({ json: { status: 'ok', model_handle: 'late-personal', model_name: 'Late',
      device: 'cpu', capabilities: {}, input_schema: { seq_len: 2, feature_dim: 3 }, layer_count: 1 } });
  });
  await page.route('**/api/v1/models/current', route => {
    released = route.request().headers()['x-model-handle'];
    return route.fulfill({ json: { status: 'deleted' } });
  });
  await page.goto(`${baseURL}/explore`);
  await page.getByRole('button', { name: 'Load Model', exact: true }).click();
  await expect.poll(() => !!finishLoad).toBe(true);
  await page.getByLabel('協作暱稱').fill('Joining during load');
  await page.getByRole('button', { name: '建立工作區', exact: true }).click();
  await expect(page.getByText('你是主持人', { exact: true })).toBeVisible();
  finishLoad();
  await expect.poll(() => released).toBe('late-personal');
  await expect(page.getByLabel('Token Index', { exact: true }).locator('option')).toHaveCount(16);
  await expect(page.getByRole('button', { name: 'Load Sample', exact: true })).toHaveCount(0);
});

for (const transition of ['replace model', 'join workspace']) {
  test(`late sample search is discarded after ${transition}`, async ({ page }) => {
    await fixture(page);
    let finishSearch;
    let loads = 0;
    const unexpectedInferences = [];
    await page.route('**/api/v1/models/load', r => r.fulfill({ json: {
      status: 'ok', model_handle: `model-${++loads}`, model_name: `Model ${loads}`, device: 'cpu',
      capabilities: {}, input_schema: { seq_len: 2, feature_dim: 3 }, layer_count: 1,
    } }));
    await page.route('**/api/legacy/binary_samples/search', async route => {
      await new Promise(resolve => { finishSearch = resolve; });
      await route.fulfill({ json: { success: true, compatible: true, feature: [[0, 0, 0], [0, 0, 0]],
        shape: [2, 3], valid_len: 2, file_name: 'old.bin', record_index: 0 } });
    });
    page.on('request', request => {
      if (request.method() === 'POST' && (request.url().endsWith('/inference-sessions') ||
          /\/workspaces\//.test(request.url()) && request.postDataJSON()?.type === 'infer')) {
        unexpectedInferences.push(request.url());
      }
    });
    await page.route('**/api/v1/inference-sessions', r => r.fulfill({ status: 400, json: { error: { message: 'Stale sample submitted' } } }));
    await page.route('**/api/v1/models/current', r => r.fulfill({ json: { status: 'deleted' } }));
    await page.goto(`${baseURL}/explore`);
    await page.getByRole('button', { name: 'Load Model', exact: true }).click();
    await page.getByRole('button', { name: 'Load Sample', exact: true }).click();
    await expect.poll(() => !!finishSearch).toBe(true);
    if (transition === 'replace model') {
      await page.getByRole('button', { name: 'Load Model', exact: true }).click();
      await expect.poll(() => loads).toBe(2);
      await expect(page.getByRole('button', { name: 'Load Sample', exact: true })).toBeVisible();
    } else {
      await page.getByLabel('協作暱稱').fill('Moving workspaces');
      await page.getByRole('button', { name: '建立工作區', exact: true }).click();
      await expect(page.getByText('你是主持人', { exact: true })).toBeVisible();
    }
    const completed = page.waitForResponse(r => r.url().endsWith('/binary_samples/search'));
    finishSearch();
    await completed;
    // Allow the response body/callback and any unwanted infer fetch to finish.
    await page.waitForTimeout(300);
    expect(unexpectedInferences).toEqual([]);
  });
}

test('two independent browsers keep real models while a third uses a shared workspace', async ({ browser }) => {
  test.setTimeout(180000);
  const contexts = await Promise.all([browser.newContext(), browser.newContext(), browser.newContext()]);
  const pages = await Promise.all(contexts.map(context => context.newPage()));
  const handles = [];
  const results = [[], []];
  const errors = [];
  try {
    for (const [index, page] of pages.entries()) {
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/api/legacy/binary_samples/files?*', r => r.fulfill({ json: {
        success: true, files: [{ name: 'fixture.bin', size: 100 }],
      } }));
      await page.route('**/api/legacy/binary_samples/search', route => {
        const { seq_len, feature_dim } = route.request().postDataJSON();
        return route.fulfill({ json: { success: true, compatible: true,
          feature: Array.from({ length: seq_len }, () => Array(feature_dim).fill(0.01)),
          shape: [seq_len, feature_dim], valid_len: seq_len, file_name: 'fixture.bin', record_index: index,
        } });
      });
      await page.goto(`${baseURL}/explore`);
    }
    const architectures = ['transTest', 'conformerTest'];
    for (let i = 0; i < 2; i++) {
      const page = pages[i];
      await page.getByTitle('Target compute device', { exact: true }).selectOption('cpu');
      await page.getByTitle('Model architecture', { exact: true }).selectOption(architectures[i]);
      const loaded = page.waitForResponse(r => r.url().endsWith('/api/v1/models/load'));
      await page.getByRole('button', { name: 'Load Model', exact: true }).click();
      const response = await loaded;
      expect(response.status()).toBe(200);
      const data = await response.json();
      handles.push(data.model_handle);
      expect(data.model_name).toBe(architectures[i]);
      await expect(page.getByLabel('Token Index', { exact: true }).locator('option')).toHaveCount(data.input_schema.seq_len);
    }
    expect(handles[0]).not.toBe(handles[1]);
    const infer = async i => {
      const page = pages[i];
      const pending = page.waitForResponse(r => r.url().endsWith('/api/v1/inference-sessions') && r.request().method() === 'POST');
      await page.getByRole('button', { name: 'Load Sample', exact: true }).click();
      const response = await pending;
      expect(response.status()).toBe(200);
      const data = await response.json();
      expect(data.model_name).toBe(architectures[i]);
      expect(data.output_sample.length).toBeGreaterThan(0);
      results[i].push(data);
      await expect(page.getByText('Raw Model Output', { exact: true })).toBeVisible();
      return data;
    };
    await Promise.all([infer(0), infer(1)]);
    const third = pages[2];
    await third.getByLabel('協作暱稱').fill('Third researcher');
    await third.getByRole('button', { name: '建立工作區', exact: true }).click();
    await expect(third.getByText('你是主持人', { exact: true })).toBeVisible();
    await third.getByTitle('Target compute device', { exact: true }).selectOption('cpu');
    await third.getByTitle('Model architecture', { exact: true }).selectOption('transTest');
    await third.getByRole('button', { name: 'Load Model', exact: true }).click();
    await expect(third.getByRole('button', { name: 'Load Sample', exact: true })).toBeVisible({ timeout: 60000 });
    await third.getByRole('button', { name: 'Load Sample', exact: true }).click();
    await expect(third.getByText('Raw Model Output', { exact: true })).toBeVisible({ timeout: 60000 });
    await Promise.all([infer(0), infer(1)]);
    for (const ownResults of results) expect(ownResults[1].output_sample).toEqual(ownResults[0].output_sample);
    const target = `${baseURL}/api/v1/inference-sessions/${results[0][1].session_id}`;
    expect((await contexts[1].request.get(target, { headers: { 'X-Model-Handle': handles[1] } })).status()).toBe(404);
    expect((await contexts[1].request.delete(target, { headers: { 'X-Model-Handle': handles[1] } })).status()).toBe(404);
    expect((await contexts[0].request.get(target, { headers: { 'X-Model-Handle': handles[0] } })).status()).toBe(200);
    expect(errors).toEqual([]);
  } finally {
    for (const [i, handle] of handles.entries()) {
      await contexts[i].request.delete(`${baseURL}/api/v1/models/current`, { headers: { 'X-Model-Handle': handle } });
    }
    await Promise.all(contexts.map(context => context.close()));
  }
});
