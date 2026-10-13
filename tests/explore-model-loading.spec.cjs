const { test, expect } = require('@playwright/test');

const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:3001';

test('checkpoint loads with CUDA and allows an architecture override', async ({ page }) => {
  const requests = [];
  await page.route('**/api/v1/models/load', async route => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({ json: { status: 'ok', model_name: 'transOriginal', device: 'cuda', capabilities: {}, model_handle: 'fixture-model', input_schema: { seq_len: 25, feature_dim: 10 }, layer_count: 2 } });
  });
  await page.goto(`${baseURL}/explore?checkpoint=${encodeURIComponent('/test/model.pth')}`);
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0]).toMatchObject({ checkpoint_path: '/test/model.pth', device: 'cuda' });
  expect(requests[0].manifest_spec).toBeUndefined();
  const architecture = page.getByTitle('Model architecture');
  await expect(architecture).toBeEnabled();
  await architecture.selectOption('transOriginal');
  await page.getByRole('button', { name: 'Load Model' }).click();
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[1].manifest_spec).toEqual({ name: 'transOriginal' });
});

test('empty path creates a transOriginal dummy model on CUDA', async ({ page }) => {
  let request;
  await page.route('**/api/v1/models/load', async route => {
    request = route.request().postDataJSON();
    await route.fulfill({ json: { status: 'ok', model_name: 'transOriginal', device: 'cuda', capabilities: {}, model_handle: 'fixture-model', input_schema: { seq_len: 25, feature_dim: 10 }, layer_count: 2 } });
  });
  await page.goto(`${baseURL}/explore`);
  await page.getByRole('button', { name: 'Load Model' }).click();
  await expect.poll(() => request).toBeTruthy();
  expect(request).toMatchObject({ checkpoint_path: 'dummy', manifest_spec: { name: 'transOriginal' }, device: 'cuda' });
});
