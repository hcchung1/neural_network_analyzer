const { test, expect } = require('@playwright/test');

const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:3001';
const csvPath = '/test/output/example_pred.csv';
const trainingFolder = '/test/output';
const checkpointPath = '/test/output/model_epoch_4.pth';

async function openRelatedResults(page) {
  await page.route('**/api/legacy/csv_reader/output_index/status', route => route.fulfill({ json: {
    success: true, root: trainingFolder, paths: [], indexed_count: 1, last_scanned_at: 1,
  } }));
  await page.route('**/api/legacy/csv_reader/open', route => route.fulfill({ json: {
    success: true, session_id: 'csv-test-session', display_name: 'example_pred.csv',
    display_path: csvPath, headers: ['prediction'], images: [],
    model_paths: [checkpointPath], training_folder: trainingFolder,
  } }));
  await page.route('**/api/legacy/csv_reader/page', route => route.fulfill({ json: {
    success: true, headers: ['prediction'], rows: [['1']], page: 0, page_size: 1000,
    start_row: 1, end_row: 1, has_previous: false, has_next: false,
    total_rows: 1, scanned_rows: 1, matched_rows: 1,
  } }));

  await page.goto(`${baseURL}/data`);
  await page.getByPlaceholder('輸入 server 上的 .csv 或 .zip 路徑').fill(csvPath);
  await page.getByRole('button', { name: 'Open CSV/ZIP' }).click();
  await page.getByRole('button', { name: 'Open related results for example_pred.csv' }).click();
}

test('training result action opens Runs and imports the selected folder', async ({ page }) => {
  await page.route('**/api/legacy/training/scan', async route => {
    const body = route.request().postDataJSON();
    expect(body.folder_path).toBe(trainingFolder);
    await route.fulfill({ json: { success: true, folder_path: trainingFolder, files: [
      { name: 'example_history.csv', path: `${trainingFolder}/example_history.csv`, size: 45 },
    ] } });
  });
  await page.route('**/api/legacy/training/read_file', async route => {
    expect(route.request().postDataJSON().file_path).toBe(`${trainingFolder}/example_history.csv`);
    await route.fulfill({ json: {
      success: true, content: '[TRAINING_HISTORY]\nepoch,train_loss,val_loss\n1,0.5,0.4\n',
    } });
  });
  await openRelatedResults(page);
  await page.getByRole('button', { name: /加入 Training Results/ }).click();
  await expect(page).toHaveURL(/\/runs(?:\?|$)/);
  await expect(page.locator('strong').filter({ hasText: 'example' }).first()).toBeVisible();
  await expect.poll(() => page.evaluate(() => {
    const results = JSON.parse(localStorage.getItem('training_results') || '[]');
    return results.length === 1 && results[0].folderPath === '/test/output' && results[0].data?.epochs?.[0] === 1;
  })).toBe(true);
});

test('epoch checkpoint action opens Explore and loads the selected checkpoint', async ({ page }) => {
  await page.route('**/api/v1/models/load', async route => {
    expect(route.request().postDataJSON().checkpoint_path).toBe(checkpointPath);
    await route.fulfill({ json: {
      status: 'ok', model_name: 'transTest', phase: null, device: 'cpu', capabilities: {},
      model_handle: 'related-checkpoint', input_schema: { seq_len: 4, feature_dim: 100 }, layer_count: 2,
    } });
  });
  await openRelatedResults(page);
  await page.getByRole('button', { name: /加入 Token Visualization/ }).click();
  await expect(page).toHaveURL(/\/explore(?:\?|$)/);
  await expect(page.getByPlaceholder(/Enter model checkpoint/)).toHaveValue(checkpointPath);
  await expect(page.getByText('Loaded transTest.', { exact: false })).toBeVisible();
});
