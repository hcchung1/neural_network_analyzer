const { test, expect } = require('@playwright/test');
const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:3001';

async function loadModel(page, checkpoint) {
  let count = 0;
  await page.route('**/api/v1/models/load', r => r.fulfill({ json: {
    status: 'ok', model_name: 'transTest', model_handle: `chat-model-${++count}`,
    device: 'cpu', capabilities: { has_embedding: false }, layer_count: 2,
    input_schema: { dtype: 'float32', rank: 3, batch_size: 1, seq_len: 4, feature_dim: 100,
      shape: [1, 4, 100], accepted_json_shapes: [[4, 100], [1, 4, 100]] },
  } }));
  await page.route('**/api/v1/models/current', r => r.fulfill({ json: { status: 'deleted' } }));
  await page.route('**/api/legacy/binary_samples/files?*', r => r.fulfill({ json: { success: true, files: [] } }));
  await page.goto(`${baseURL}/explore`);
  if (checkpoint) await page.getByPlaceholder(/Enter model checkpoint/).fill(checkpoint);
  await page.getByRole('button', { name: 'Load Model', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Release Model' })).toBeVisible();
  await expect(page.getByText('Loaded transTest.', { exact: true })).toBeVisible();
}

test('assistant sends page_context and displays reply without JSON wrapping', async ({ page }) => {
  await loadModel(page);
  let body;
  await page.route('**/api/v1/assistant/messages', r => {
    body = r.request().postDataJSON();
    return r.fulfill({ json: { success: true, kind: 'chat', reply: 'Model explanation' } });
  });
  await page.getByLabel('Assistant message').fill('Explain this model');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.getByText('Model explanation', { exact: true })).toBeVisible();
  expect(body.context).toBeUndefined();
  expect(body.page_context).toMatchObject({ model_path: 'transTest', selected_token: 0, current_layer: 0,
    model_input_schema: { dtype: 'float32', rank: 3, batch_size: 1, seq_len: 4, feature_dim: 100, shape: [1, 4, 100] } });
  expect(body.page_context.model_input_schema.accepted_json_shapes).toBeUndefined();
});

test('assistant describes the loaded checkpoint after editing the next path or a failed replacement', async ({ page }) => {
  await loadModel(page, '/models/A.pth');
  const paths = [];
  await page.route('**/api/v1/assistant/messages', r => {
    paths.push(r.request().postDataJSON().page_context.model_path);
    return r.fulfill({ json: { success: true, kind: 'chat', reply: `Reply ${paths.length}` } });
  });
  await page.getByPlaceholder(/Enter model checkpoint/).fill('/models/B.pth');
  await page.getByLabel('Assistant message').fill('Explain active model');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.getByText('Reply 1', { exact: true })).toBeVisible();
  await page.route('**/api/v1/models/load', r => r.fulfill({ status: 400,
    json: { error: { code: 'MODEL_BUILD_FAILED', message: 'Invalid checkpoint' } } }));
  await page.getByRole('button', { name: 'Load Model', exact: true }).click();
  await expect(page.getByText('Invalid checkpoint', { exact: true })).toBeVisible();
  await page.getByLabel('Assistant message').fill('Explain active model again');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.getByText('Reply 2', { exact: true })).toBeVisible();
  expect(paths).toEqual(['/models/A.pth', '/models/A.pth']);
});

test('assistant provider errors are readable and allow retry', async ({ page }) => {
  await loadModel(page);
  await page.route('**/api/v1/assistant/messages', r => r.fulfill({ status: 503,
    json: { detail: '分析對話服務暫時無法使用，請稍後再試。' } }));
  await page.getByLabel('Assistant message').fill('Explain');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.getByText('分析對話服務暫時無法使用，請稍後再試。', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeEnabled();
});

test('late assistant reply is discarded when a new model replaces its context', async ({ page }) => {
  await loadModel(page);
  let finish;
  await page.route('**/api/v1/assistant/messages', async r => {
    await new Promise(resolve => { finish = resolve; });
    await r.fulfill({ json: { success: true, kind: 'chat', reply: 'Old model explanation' } });
  });
  await page.getByLabel('Assistant message').fill('Explain old model');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect.poll(() => !!finish).toBe(true);
  await page.getByRole('button', { name: 'Load Model', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Load Sample', exact: true })).toBeVisible();
  const completed = page.waitForResponse(r => r.url().endsWith('/assistant/messages'));
  finish();
  await completed;
  await page.waitForTimeout(150);
  await expect(page.getByText('Old model explanation', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Explain old model', { exact: true })).toHaveCount(0);
});
