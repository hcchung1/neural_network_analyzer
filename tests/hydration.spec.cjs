const { test, expect } = require('@playwright/test');

const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:3001';
const savedResult = {
  id: 'hydration-regression', name: 'Saved training run',
  color: '#1f77b4', copyColor: '#ff7f0e', folderPath: '/saved/run',
  visible: true, data: null, phaseData: null, randomPhaseData: null,
  turnMetrics: null, randomTurnMetrics: null,
};

// Catch both React's recoverable hydration warnings and fatal runtime errors.
// Merely checking the final DOM would miss React replacing the server HTML.
test.beforeEach(async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (['error', 'warning'].includes(message.type()) &&
        /hydrat|server HTML|server-rendered|did not match/i.test(message.text())) {
      errors.push(message.text());
    }
  });
  page.hydrationErrors = errors;
});

test.afterEach(async ({ page }) => {
  expect(page.hydrationErrors).toEqual([]);
});

async function checkSidebar(page) {
  const sidebar = page.locator('aside').first();
  await expect(sidebar).toBeVisible();
  await page.getByTitle('Collapse Sidebar', { exact: true }).click();
  await expect(page.getByTitle('Expand Sidebar', { exact: true })).toBeVisible();
  await page.getByTitle('Expand Sidebar', { exact: true }).click();
  await expect(sidebar.getByRole('link', { name: 'Runs', exact: true })).toBeVisible();
}

for (const path of ['/runs', '/explore', '/compare', '/settings']) {
  test(`${path} hydrates on direct entry and reload`, async ({ page }) => {
    test.setTimeout(120000);
    const response = await page.goto(`${baseURL}${path}`);
    expect(response.status()).toBe(200);
    // The sidebar must actually be server rendered, not hidden until mount.
    expect(await response.text()).toContain('<aside');
    await checkSidebar(page);
    await page.reload();
    await checkSidebar(page);
  });
}

test('Runs restores saved results across navigation and reload without hydration errors', async ({ page }) => {
  test.setTimeout(120000);
  await page.addInitScript(result => {
    if (!localStorage.getItem('training_results')) {
      localStorage.setItem('training_results', JSON.stringify([result]));
    }
  }, savedResult);
  await page.goto(`${baseURL}/runs`);
  await expect(page.getByText(savedResult.name, { exact: true }).first()).toBeVisible();
  await checkSidebar(page);
  await page.getByRole('link', { name: 'Settings', exact: true }).click();
  await expect(page).toHaveURL(`${baseURL}/settings`);
  await page.getByRole('link', { name: 'Runs', exact: true }).click();
  await expect(page).toHaveURL(`${baseURL}/runs`);
  await expect(page.getByText(savedResult.name, { exact: true }).first()).toBeVisible();
  await page.reload();
  await expect(page.getByText(savedResult.name, { exact: true }).first()).toBeVisible();
  await checkSidebar(page);
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('training_results'))))
    .toEqual([savedResult]);
});
