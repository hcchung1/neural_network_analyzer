const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: './tests', testMatch: '**/*.spec.cjs', workers: 1,
  use: { headless: true, trace: 'retain-on-failure', screenshot: 'only-on-failure' },
});
