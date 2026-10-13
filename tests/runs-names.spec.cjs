const { test, expect } = require('@playwright/test');

const base = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:33107';
const defaultFolder = '/workspace/Study/Mahjong-Hidden-Information-Forecast/Transformer/output/';
const name = `Transformer_${'long_run_tag_'.repeat(24)}20261007_123456`;

test.beforeEach(async ({ page }) => {
  await page.addInitScript(({ name }) => {
    localStorage.setItem('training_results', JSON.stringify([{ id: 'long', name,
      folderPath: '/output', color: '#1f77b4', copyColor: '#2ca02c', visible: false,
      data: null, phaseData: null, randomPhaseData: null, turnMetrics: null, randomTurnMetrics: null }]));
  }, { name });
});

test('long Run names fit their card, retain the date, and copy the full name', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'clipboard', {
    configurable: true, value: { writeText: async text => { window.copiedName = text; } },
  }));
  await page.goto(`${base}/runs`);
  const label = page.locator('strong').filter({ hasText: name });
  await expect(label).toBeVisible();
  for (const width of [1440, 768]) {
    await page.setViewportSize({ width, height: 900 });
    const layout = await label.evaluate(element => {
      const card = element.closest('[data-result-card]') || element.parentElement.parentElement;
      const end = element.querySelector('[data-name-suffix]');
      return { cardFits: card.scrollWidth <= card.clientWidth,
        suffix: end?.textContent, suffixFits: end ? end.getBoundingClientRect().right <= card.getBoundingClientRect().right : false };
    });
    expect(layout.cardFits).toBe(true);
    expect(layout.suffix).toBe('20261007_123456');
    expect(layout.suffixFits).toBe(true);
  }
  await page.getByRole('button', { name: '複製完整名稱', exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.copiedName)).toBe(name);
  await expect(page.getByRole('button', { name: '已複製', exact: true })).toBeVisible();
});

test('the default folder can be cleared and replaced for scanning', async ({ page }) => {
  let scannedFolder;
  await page.route('**/api/legacy/training/scan', route => {
    scannedFolder = route.request().postDataJSON().folder_path;
    return route.fulfill({ json: { success: true, folder_path: scannedFolder, files: [] } });
  });
  await page.goto(`${base}/runs`);
  const input = page.getByPlaceholder('輸入資料夾路徑', { exact: false });
  await expect(input).toHaveValue(defaultFolder);
  await input.fill('');
  await input.blur();
  await expect(input).toHaveValue('');
  await input.fill('/custom/output');
  await page.getByRole('button', { name: '掃描資料夾', exact: true }).click();
  await expect.poll(() => scannedFolder).toBe('/custom/output');
});

test('blocked copying reports failure without claiming success', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
    document.execCommand = () => false;
  });
  await page.goto(`${base}/runs`);
  await page.getByRole('button', { name: '複製完整名稱', exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: '複製失敗' })).toBeVisible();
  await expect(page.getByRole('button', { name: '已複製', exact: true })).toHaveCount(0);
});
