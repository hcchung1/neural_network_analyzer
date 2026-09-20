const { test } = require('@playwright/test');
const { run } = require('./collaboration-flow.cjs');
test('mixed collaboration across independent browsers', async ({ page }) => {
  test.setTimeout(240000);
  await run(page, process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:3001');
});
