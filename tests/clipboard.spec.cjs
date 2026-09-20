const { test, expect } = require('@playwright/test');
const fs = require('node:fs');
const ts = require('typescript');

test.beforeEach(async ({ page }) => {
  await page.setContent('<button id="copy">Copy</button><textarea id="paste"></textarea>');
  const source = fs.readFileSync(require.resolve('../lib/clipboard.ts'), 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2017 },
  });
  await page.addScriptTag({ content: `var exports = {}; ${outputText}` });
  await page.evaluate(() => {
    window.copyValue = '欄位,"quoted"\n第二列';
    document.querySelector('#copy').onclick = async () => {
      try {
        await exports.copyToClipboard(window.copyValue);
        window.copyResult = 'success';
      } catch (error) {
        window.copyResult = error.message;
      }
    };
  });
});

for (const mode of ['missing', 'denied']) {
  test(`copies exact text when Clipboard API is ${mode}`, async ({ page }) => {
    await page.evaluate((mode) => {
      Object.defineProperty(navigator, 'clipboard', {
        configurable: true,
        value: mode === 'missing' ? undefined : {
          writeText: async () => { throw new DOMException('Denied', 'NotAllowedError'); },
        },
      });
    }, mode);
    await page.click('#copy');
    await expect.poll(() => page.evaluate(() => window.copyResult)).toBe('success');
    await expect(page.locator('#copy')).toBeFocused();
    await expect(page.locator('textarea')).toHaveCount(1);
    await page.locator('#paste').focus();
    await page.keyboard.press('Control+V');
    await expect(page.locator('#paste')).toHaveValue('欄位,"quoted"\n第二列');
  });
}

test('reports blocked copying and cleans up the temporary selection', async ({ page }) => {
  await page.evaluate(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
    document.execCommand = () => false;
  });
  await page.click('#copy');
  await expect.poll(() => page.evaluate(() => window.copyResult)).toMatch(/瀏覽器/);
  await expect(page.locator('textarea')).toHaveCount(1);
  await expect(page.locator('#copy')).toBeFocused();
});
