const { test, expect } = require('@playwright/test');
const base = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:3001';

test('real workspace heartbeat omits stable data while discussion and view stay intact', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`${base}/explore`);
  await page.getByLabel('協作暱稱').fill('Polling host');
  await page.getByRole('button', { name: '建立工作區', exact: true }).click();
  await page.getByText('你是主持人', { exact: true }).waitFor();
  await page.getByLabel('協作訊息').fill('Heartbeat keeps this discussion');
  await page.getByRole('button', { name: '傳送', exact: true }).click();
  await expect(page.getByText('Heartbeat keeps this discussion', { exact: true })).toBeVisible();
  await page.getByLabel('Token Index', { exact: true }).selectOption('5');
  for (let index = 0; index < 2; index++) {
    const response = await page.waitForResponse(async response => {
      if (response.request().method() !== 'GET' || !response.url().includes('/api/v1/workspaces/')) return false;
      return (await response.json()).unchanged === true;
    });
    const url = new URL(response.url());
    expect(url.searchParams.has('revision')).toBe(true);
    expect(url.searchParams.has('analysisRevision')).toBe(true);
    const body = await response.json();
    expect(body.members.some(member => member.name === 'Polling host' && member.online)).toBe(true);
    expect(body).not.toHaveProperty('entries');
    expect(body).not.toHaveProperty('view');
    expect(body).not.toHaveProperty('analysis');
    await expect(page.getByText('Heartbeat keeps this discussion', { exact: true })).toBeVisible();
    await expect(page.getByLabel('Token Index', { exact: true })).toHaveValue('5');
    await expect(page.getByText('● 已連線', { exact: true })).toBeVisible();
  }
  expect(errors).toEqual([]);
});
