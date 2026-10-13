const { test, expect } = require('@playwright/test');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
test.use({ ignoreHTTPSErrors: true });

const base = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:3001';
let folder;
const tag = 'transVOG_TC_BF_SL_MG_FMT_old_single_NORM_ln_S42_TP_last_n_actions24_LM_symmetric_BS64_NE40_ACT10_20261003_235111';
const imageUrl = (kind, run = tag) => `${base}/api/v1/training-images?${new URLSearchParams({ folder, run, kind })}`;

test.beforeAll(async ({ browser }) => {
  folder = fs.mkdtempSync(path.join(os.tmpdir(), 'runs-images-')) + path.sep;
  fs.mkdirSync(`${folder}${tag}/random_feature_analysis`, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 24, height: 16 } });
  await page.setContent('<body style="background:red"></body>');
  await page.screenshot({ path: `${folder}${tag}/${tag}_all.png` });
  await page.setContent('<body style="background:blue"></body>');
  await page.screenshot({ path: `${folder}${tag}/random_feature_analysis/${tag}_all_base_vs_random_turn_metrics.png` });
  await page.close();
});
test.afterAll(() => { if (folder) fs.rmSync(folder, { recursive: true, force: true }); });

test('serves the existing Training Plot and PhaseAnalyzer PNG bytes from the scanned root', async ({ request }) => {
  for (const [kind, file] of [['training', `${tag}/${tag}_all.png`], ['phase', `${tag}/random_feature_analysis/${tag}_all_base_vs_random_turn_metrics.png`]]) {
    const response = await request.get(imageUrl(kind));
    expect(response.status()).toBe(200);
    expect(response.headers()['content-type']).toBe('image/png');
    expect((await response.body()).equals(fs.readFileSync(folder + file))).toBe(true);
    const importedName = await request.get(imageUrl(kind, `${tag}_all`));
    expect(importedName.status()).toBe(200);
    expect((await importedName.body()).equals(await response.body())).toBe(true);
  }
});

test('missing images and invalid Run names return clear errors', async ({ request }) => {
  const missing = await request.get(imageUrl('training', 'missing_20261008_000000'));
  expect(missing.status()).toBe(404);
  expect((await missing.json()).error).toMatch(/找不到/);
  expect((await request.get(imageUrl('training', '../outside'))).status()).toBe(400);
});

async function openRuns(page, name = `${tag}_all`) {
  await page.addInitScript(({ name, folder }) => {
    localStorage.setItem('training_results', JSON.stringify([{ id: 'images', name,
      folderPath: folder, color: '#1f77b4', copyColor: '#2ca02c', visible: false,
      data: null, phaseData: null, randomPhaseData: null, turnMetrics: null, randomTurnMetrics: null }]));
  }, { name, folder });
  await page.goto(`${base}/runs`);
}

test('both image buttons write image data to the clipboard and occupy the second action row', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await openRuns(page);
  const card = page.locator('[data-result-card]');
  const copyName = card.getByRole('button', { name: '複製完整名稱', exact: true });
  for (const label of ['複製訓練圖片', '複製Phase圖片']) {
    const button = card.getByRole('button', { name: label, exact: true });
    await expect(button).toBeVisible();
    expect((await button.boundingBox()).y).toBeGreaterThan((await copyName.boundingBox()).y);
    await button.click();
    await expect(card.getByRole('status').filter({ hasText: `${label}：已複製` })).toBeVisible();
    const copied = await page.evaluate(async () => {
      const items = await navigator.clipboard.read();
      const blob = await items[0].getType('image/png');
      const image = await createImageBitmap(blob);
      const dimensions = { width: image.width, height: image.height, type: blob.type };
      image.close();
      return dimensions;
    });
    expect(copied.type).toBe('image/png');
    expect(copied.width).toBe(24);
    expect(copied.height).toBe(16);
  }
});

test('a missing image hides only its own copy button for each Run', async ({ page }) => {
  const trainingOnly = 'training_only_20261011_000001';
  const phaseOnly = 'phase_only_20261011_000002';
  fs.copyFileSync(`${folder}${tag}/${tag}_all.png`, `${folder}${trainingOnly}_all.png`);
  fs.mkdirSync(`${folder}random_feature_analysis`, { recursive: true });
  fs.copyFileSync(`${folder}${tag}/random_feature_analysis/${tag}_all_base_vs_random_turn_metrics.png`, `${folder}random_feature_analysis/${phaseOnly}_all_base_vs_random_turn_metrics.png`);
  await page.addInitScript(({ folder, names }) => {
    localStorage.setItem('training_results', JSON.stringify(names.map(name => ({ id: name, name,
      folderPath: folder, color: '#1f77b4', copyColor: '#2ca02c', visible: false,
      data: null, phaseData: null, randomPhaseData: null, turnMetrics: null, randomTurnMetrics: null }))));
  }, { folder, names: [trainingOnly, phaseOnly, 'missing_20261011_000003'] });
  const checks = [];
  page.on('response', response => {
    if (response.url().includes('/api/v1/training-images?') && response.url().includes('check=1')) checks.push(response);
  });
  await page.goto(`${base}/runs`);
  await expect.poll(() => checks.length).toBe(6);
  for (const response of checks) expect(response.status()).toBe(200);
  const cards = page.locator('[data-result-card]');
  await expect(cards.nth(0).getByRole('button', { name: '複製訓練圖片', exact: true })).toBeVisible();
  await expect(cards.nth(0).getByRole('button', { name: '複製Phase圖片', exact: true })).toHaveCount(0);
  await expect(cards.nth(1).getByRole('button', { name: '複製訓練圖片', exact: true })).toHaveCount(0);
  await expect(cards.nth(1).getByRole('button', { name: '複製Phase圖片', exact: true })).toBeVisible();
  await expect(cards.nth(2).getByRole('button', { name: /複製.*圖片/ })).toHaveCount(0);
});

test('image copying still reports a file removed after the existence check', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await openRuns(page);
  const button = page.getByRole('button', { name: '複製訓練圖片', exact: true });
  await expect(button).toBeVisible();
  const file = `${folder}${tag}/${tag}_all.png`;
  fs.renameSync(file, `${file}.hidden`);
  try {
    await button.click();
    await expect(page.getByRole('status').filter({ hasText: '找不到圖片' })).toBeVisible();
    await expect(page.getByRole('status').filter({ hasText: '已複製' })).toHaveCount(0);
  } finally { fs.renameSync(`${file}.hidden`, file); }
});

test('unsupported image clipboards show how to enable image copying', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined }));
  await openRuns(page);
  await page.getByRole('button', { name: '複製Phase圖片', exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: /HTTPS.*localhost/ })).toBeVisible();
});
