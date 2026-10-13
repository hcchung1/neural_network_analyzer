const { test, expect } = require('@playwright/test');

const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:3001';

test('returning to Data reuses open sessions and fetches only the selected file', async ({ page }) => {
  const opened = [];
  const paged = [];
  await page.route('**/api/legacy/csv_reader/output_index/status', route => route.fulfill({ json: {
    success: true, root: '/test', paths: [], indexed_count: 0, last_scanned_at: null,
  } }));
  await page.route('**/api/legacy/csv_reader/open', route => {
    const path = route.request().postDataJSON().path;
    opened.push(path);
    const name = path.split('/').pop();
    return route.fulfill({ json: {
      success: true, session_id: `session-${opened.length}`, display_name: name,
      display_path: path, headers: ['value'], images: [], model_paths: [], training_folder: null,
    } });
  });
  await page.route('**/api/legacy/csv_reader/page', route => {
    const id = route.request().postDataJSON().session_id;
    paged.push(id);
    return route.fulfill({ json: {
      success: true, headers: ['value'], rows: [[id]], page: 0, page_size: 200,
      start_row: 1, end_row: 1, has_previous: false, has_next: false,
      total_rows: 1, scanned_rows: 1, matched_rows: 1,
    } });
  });

  await page.goto(`${baseURL}/data`);
  for (const path of ['/test/first.csv', '/test/second.csv']) {
    await page.getByPlaceholder('輸入 server 上的 .csv 或 .zip 路徑').fill(path);
    await page.getByRole('button', { name: 'Open CSV/ZIP' }).click();
    await expect(page.locator('tbody tr')).toHaveCount(1);
  }
  await expect.poll(() => opened.length).toBe(2);
  await expect.poll(() => paged.length).toBe(2);

  await page.getByRole('link', { name: 'Settings', exact: true }).click();
  await expect(page).toHaveURL(/\/settings$/);
  await page.getByRole('link', { name: 'Data', exact: true }).click();
  await expect(page).toHaveURL(/\/data$/);
  await expect(page.locator('tbody')).toContainText('session-2');
  await expect.poll(() => paged.length).toBe(3);
  // Returning to Data and reloading are distinct mounts; each reads only the
  // selected session once. Do not rely on reload racing the first restore.
  await page.reload();
  await expect(page.locator('tbody tr')).toHaveCount(1);
  await expect.poll(() => paged.length).toBe(4);
  expect(opened).toEqual(['/test/first.csv', '/test/second.csv']);
  expect(paged).toEqual(['session-1', 'session-2', 'session-2', 'session-2']);

  await page.getByRole('button', { name: 'first.csv', exact: true }).first().click();
  await expect.poll(() => paged.length).toBe(5);
  expect(opened).toHaveLength(2);
  expect(paged[4]).toBe('session-1');
});

test('an empty CSV page is requested once after opening', async ({ page }) => {
  let pageRequests = 0;
  await page.route('**/api/legacy/csv_reader/output_index/status', route => route.fulfill({ json: {
    success: true, root: '/test', paths: [], indexed_count: 0, last_scanned_at: null,
  } }));
  await page.route('**/api/legacy/csv_reader/open', route => route.fulfill({ json: {
    success: true, session_id: 'empty-session', display_name: 'empty.csv',
    display_path: '/test/empty.csv', headers: ['value'], images: [],
    model_paths: [], training_folder: null,
  } }));
  await page.route('**/api/legacy/csv_reader/page', route => {
    pageRequests++;
    return route.fulfill({ json: {
      success: true, headers: ['value'], rows: [], page: 0, page_size: 200,
      start_row: 0, end_row: 0, has_previous: false, has_next: false,
      total_rows: 0, scanned_rows: 0, matched_rows: 0,
    } });
  });
  await page.goto(`${baseURL}/data`);
  await page.getByPlaceholder('輸入 server 上的 .csv 或 .zip 路徑').fill('/test/empty.csv');
  await page.getByRole('button', { name: 'Open CSV/ZIP' }).click();
  await expect(page.getByText('Rows 0-0', { exact: false })).toBeVisible();
  await page.waitForTimeout(250);
  expect(pageRequests).toBe(1);
});

test('applying a filter does not require scanning the entire CSV for a total', async ({ page }) => {
  const requests = [];
  await page.route('**/api/legacy/csv_reader/output_index/status', route => route.fulfill({ json: {
    success: true, root: '/test', paths: [], indexed_count: 0, last_scanned_at: null,
  } }));
  await page.route('**/api/legacy/csv_reader/open', route => route.fulfill({ json: {
    success: true, session_id: 'filter-session', display_name: 'filter.csv',
    display_path: '/test/filter.csv', headers: ['value'], images: [],
    model_paths: [], training_folder: null,
  } }));
  await page.route('**/api/legacy/csv_reader/page', route => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({ json: {
      success: true, headers: ['value'], rows: [['1']], page: 0, page_size: 200,
      start_row: 1, end_row: 1, has_previous: false, has_next: false,
      total_rows: null, scanned_rows: 1, matched_rows: 1,
    } });
  });
  await page.goto(`${baseURL}/data`);
  await page.getByPlaceholder('輸入 server 上的 .csv 或 .zip 路徑').fill('/test/filter.csv');
  await page.getByRole('button', { name: 'Open CSV/ZIP' }).click();
  await expect(page.locator('tbody tr')).toHaveCount(1);
  await page.getByPlaceholder('prediction = 1, actual = 0, probability > 0.7').fill('value = 1');
  await page.getByRole('button', { name: 'Apply', exact: true }).click();
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[0].page_size).toBe(200);
  expect(requests[1]).toMatchObject({ filter_text: 'value = 1', include_total: false });
});

test('a delayed stale-session reopen does not change the selected CSV', async ({ page }) => {
  const closed = [];
  const sessions = ['/test/first.csv', '/test/second.csv'].map((sourcePath, index) => ({
    sessionId: `old-${index}`, sourcePath, displayName: sourcePath.split('/').pop(),
    displayPath: sourcePath, headers: ['value'], images: [], modelPaths: [],
    trainingFolder: null, currentPage: 0, filterText: '', pinned: true,
  }));
  await page.addInitScript(saved => localStorage.setItem('csv_reader_metadata', JSON.stringify(saved)), {
    sessions, activeSourcePath: sessions[0].sourcePath, pageSize: 200,
    sidePaneOpen: false, sideMode: 'web', sideUrl: '', selectedImageId: '0', imageZoom: 1,
  });
  await page.route('**/api/legacy/csv_reader/output_index/status', route => route.fulfill({ json: {
    success: true, root: '/test', paths: [], indexed_count: 0, last_scanned_at: null,
  } }));
  await page.route('**/api/legacy/csv_reader/page', route => {
    const id = route.request().postDataJSON().session_id;
    if (id === 'old-0') return route.fulfill({ json: { success: false, error: 'Unknown CSV session' } });
    return route.fulfill({ json: {
      success: true, headers: ['value'], rows: [[id]], page: 0, page_size: 200,
      start_row: 1, end_row: 1, has_previous: false, has_next: false,
      total_rows: 1, scanned_rows: 1, matched_rows: 1,
    } });
  });
  let notifyOpen;
  const openStarted = new Promise(resolve => { notifyOpen = resolve; });
  let releaseOpen;
  const blockedOpen = new Promise(resolve => { releaseOpen = resolve; });
  await page.route('**/api/legacy/csv_reader/open', async route => {
    notifyOpen();
    await blockedOpen;
    await route.fulfill({ json: {
      success: true, session_id: 'new-first', display_name: 'first.csv',
      display_path: '/test/first.csv', headers: ['value'], images: [],
      model_paths: [], training_folder: null,
    } });
  });
  await page.route('**/api/legacy/csv_reader/close', route => {
    closed.push(route.request().postDataJSON().session_id);
    return route.fulfill({ json: { success: true } });
  });

  await page.goto(`${baseURL}/data`);
  await openStarted;
  await page.getByRole('button', { name: 'second.csv', exact: true }).first().click();
  await expect(page.getByRole('button', { name: 'second.csv', exact: true }).first()).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Close first.csv' }).click();
  releaseOpen();
  await expect(page.locator('tbody td')).toHaveText('old-1');
  await expect.poll(() => closed).toContain('new-first');
  expect(closed).toContain('old-0');
  await expect(page.getByRole('button', { name: 'first.csv', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'second.csv', exact: true }).first()).toHaveAttribute('aria-pressed', 'true');
});
