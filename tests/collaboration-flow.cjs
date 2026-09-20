// Reusable with Playwright Test or browser_run_code_unsafe: run(page).
const assert = require('node:assert/strict');

async function run(page, baseURL = 'http://127.0.0.1:3001') {
  const browser = page.context().browser();
  const guestContext = await browser.newContext();
  const guest = await guestContext.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  guest.on('pageerror', e => errors.push(e.message));
  const token = p => p.getByLabel('Token Index', { exact: true });
  const waitToken = (p, value) => p.waitForFunction(v => document.querySelector('[aria-label="Token Index"]')?.value === v, String(value));
  const identity = p => p.evaluate(() => JSON.parse(sessionStorage.getItem(`workspace:${new URLSearchParams(location.search).get('room')}`)));
  const request = (p, action) => p.evaluate(async action => {
    const id = JSON.parse(sessionStorage.getItem(`workspace:${new URLSearchParams(location.search).get('room')}`));
    const r = await fetch(`/api/v1/workspaces/${id.roomId}`, {
      method: action ? 'POST' : 'GET', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${id.token}` },
      ...(action ? { body: JSON.stringify(action) } : {}),
    });
    return { status: r.status, body: await r.json() };
  }, action);
  const send = async (p, text) => {
    await p.getByLabel('協作訊息').fill(text);
    await p.getByRole('button', { name: '傳送', exact: true }).click();
    await p.getByText(text, { exact: true }).waitFor();
  };
  try {
    await page.goto(`${baseURL}/explore`);
    await page.getByLabel('協作暱稱').fill('Alice');
    await page.getByRole('button', { name: '建立工作區', exact: true }).click();
    await page.getByText('你是主持人', { exact: true }).waitFor();
    await page.getByRole('button', { name: '分享連結', exact: true }).click();
    assert.equal(await page.getByLabel('工作區分享連結').inputValue(), page.url());
    await guest.goto(page.url());
    await guest.getByLabel('協作暱稱').fill('Bob');
    await guest.getByRole('button', { name: '加入工作區', exact: true }).click();
    await guest.getByText('獨立瀏覽', { exact: true }).waitFor();
    assert.equal(await guest.getByRole('button', { name: 'Load Model', exact: true }).isEnabled(), false);
    await token(page).selectOption('3');
    await page.waitForTimeout(1200);
    assert.equal(await token(guest).inputValue(), '0');
    await guest.getByRole('button', { name: '跟隨主持人', exact: true }).click();
    await waitToken(guest, 3);
    await token(page).selectOption('5');
    await waitToken(guest, 5);
    await page.getByRole('slider').fill('2');
    await guest.waitForFunction(() => document.querySelector('input[type=range]')?.value === '2');
    await token(guest).selectOption('2');
    await guest.getByText('獨立瀏覽', { exact: true }).waitFor();
    await token(page).selectOption('6');
    await page.waitForTimeout(1200);
    assert.equal(await token(guest).inputValue(), '2');
    await send(guest, '一起檢查這個注意力分布');
    await page.getByText('一起檢查這個注意力分布', { exact: true }).waitFor();
    await guest.getByLabel('訊息類型').selectOption('annotation');
    await send(guest, '這裡是 Token 2 / Layer 2');
    await page.getByText('這裡是 Token 2 / Layer 2', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Token 2 · Layer 2', exact: true }).click();
    await waitToken(page, 2);
    const beforeReload = await identity(guest);
    await guest.reload();
    await guest.getByText('這裡是 Token 2 / Layer 2', { exact: true }).waitFor();
    assert.deepEqual(await identity(guest), beforeReload);
    assert.equal((await request(guest, { type: 'view', view: { token: 0, layer: 0 }, analysisRevision: 0 })).status, 403);
    assert.equal((await request(guest, { type: 'load', model: { checkpoint_path: 'dummy', device: 'cpu' }, analysisRevision: 0 })).status, 403);
    const unauthenticated = await page.request.get(`${baseURL}/api/v1/workspaces/${beforeReload.roomId}`);
    assert.equal(unauthenticated.status(), 401);
    const other = await page.request.post(`${baseURL}/api/v1/workspaces`, { data: { name: 'Other room' } });
    const otherData = await other.json();
    const crossRoom = await page.request.get(`${baseURL}/api/v1/workspaces/${otherData.identity.roomId}`, { headers: { Authorization: `Bearer ${beforeReload.token}` } });
    assert.equal(crossRoom.status(), 401);

    // Only the binary-file source is a deterministic fixture. Model loading,
    // inference, room persistence and all collaboration calls use the real server.
    await page.route('**/api/legacy/binary_samples/files?*', route => route.fulfill({ json: { success: true, files: [{ name: 'collaboration-fixture.bin', size: 100 }] } }));
    await page.getByRole('button', { name: 'Load Model', exact: true }).click();
    await page.getByLabel('Binary file').waitFor({ timeout: 120000 });
    const loaded = (await request(page)).body;
    assert.equal(loaded.analysis.modelName, 'transTest');
    await guest.waitForFunction(() => document.querySelector('[aria-label="Token Index"]')?.options.length === 25);
    assert.equal((await request(page, { type: 'view', view: { token: 0, layer: 0 }, analysisRevision: 0 })).status, 409);
    const schema = loaded.analysis.inputSchema;
    await page.route('**/api/legacy/binary_samples/search', route => route.fulfill({ json: {
      success: true, compatible: true, file_name: 'collaboration-fixture.bin', record_index: 7,
      valid_len: schema.seq_len, line_number: 8, label: 0, shape: [schema.seq_len, schema.feature_dim],
      feature: Array.from({ length: schema.seq_len }, () => Array(schema.feature_dim).fill(0.01)),
    } }));
    await page.getByRole('button', { name: 'Load Sample', exact: true }).click();
    await page.getByTestId('shared-sample').waitFor({ timeout: 120000 });
    await guest.getByTestId('shared-sample').waitFor();
    await page.getByText('Output Probabilities', { exact: true }).waitFor();
    await guest.getByText('Output Probabilities', { exact: true }).waitFor();
    const hostResult = (await request(page)).body.analysis;
    const guestResult = (await request(guest)).body.analysis;
    assert.deepEqual(hostResult, guestResult);
    assert.ok(hostResult.output.length > 0);
    assert.ok(Object.keys(hostResult.attention).length > 0);
    const currentRevision = (await request(page)).body.analysisRevision;
    assert.equal((await request(page, { type: 'annotation', text: 'out of bounds', view: { token: 99999, layer: 0 }, analysisRevision: currentRevision })).status, 409);
    assert.equal((await request(page, { type: 'load', model: { checkpoint_path: '/tmp/archanalyzer-nonexistent-test-checkpoint.pth', device: 'cpu' }, analysisRevision: currentRevision })).status, 400);
    const afterFailure = (await request(page)).body;
    assert.equal(afterFailure.busy, false);
    assert.deepEqual(afterFailure.analysis, hostResult);
    assert.equal(await page.getByRole('button', { name: 'Token 2 · Layer 2 · 先前分析', exact: true }).isEnabled(), false);
    const otherState = await page.request.get(`${baseURL}/api/v1/workspaces/${otherData.identity.roomId}`, { headers: { Authorization: `Bearer ${otherData.identity.token}` } });
    assert.equal((await otherState.json()).analysisRevision, 0);

    await guestContext.setOffline(true);
    await guest.getByText('○ 重新連線中…', { exact: true }).waitFor();
    await guestContext.setOffline(false);
    await guest.getByText('● 已連線', { exact: true }).waitFor();
    await page.getByRole('button', { name: '交接主持人給 Bob', exact: true }).click();
    await guest.getByText('你是主持人', { exact: true }).waitFor();
    await page.getByText('獨立瀏覽', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Load Model', exact: true }).isEnabled(), false);
    await page.getByRole('button', { name: '跟隨主持人', exact: true }).click();
    await token(guest).selectOption('4');
    await waitToken(page, 4);
    await guestContext.setOffline(true);
    await page.getByRole('button', { name: '接任主持人', exact: true }).waitFor({ timeout: 25000 });
    await page.getByRole('button', { name: '接任主持人', exact: true }).click();
    await page.getByText('你是主持人', { exact: true }).waitFor();
    await guestContext.setOffline(false);
    await guest.getByText('獨立瀏覽', { exact: true }).waitFor();
    await page.getByRole('button', { name: '交接主持人給 Bob', exact: true }).click();
    await guest.getByText('你是主持人', { exact: true }).waitFor();
    await guest.getByRole('button', { name: '離開工作區', exact: true }).click();
    await page.getByText('你是主持人', { exact: true }).waitFor();
    await page.reload();
    await page.getByTestId('shared-sample').waitFor();
    await page.getByText('Output Probabilities', { exact: true }).waitFor();
    await guest.goto(page.url());
    await guest.getByLabel('協作暱稱').fill('Returning Bob');
    await guest.getByRole('button', { name: '加入工作區', exact: true }).click();
    await guest.getByTestId('shared-sample').waitFor();
    await guest.getByText('Output Probabilities', { exact: true }).waitFor();
    assert.deepEqual(errors, []);
    await page.setViewportSize({ width: 1600, height: 1200 });
    await page.screenshot({ path: 'test-results/collaboration-success.png', fullPage: true });
    return { roomURL: page.url(), cases: ['independent views', 'follow token/layer', 'manual unfollow', 'chat', 'anchored annotations', 'refresh persistence', 'host authorization', 'cross-room isolation', 'real inference and shared charts', 'failed analysis preserves results', 'stale revision rejection', 'offline reconnect', 'host transfer', 'offline host takeover', 'host departure', 'late join hydration'], pageErrors: errors };
  } finally { await guestContext.close(); }
}
module.exports = { run };
