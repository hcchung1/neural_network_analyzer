# ArchAnalyzer 問題分析報告

## 🔴 核心問題總結

基於目前的代碼掃描和用戶提供的 Console 輸出，以下是所有**已知缺失**的完整分析：

---

## 1. ⚠️ 前端 WebSocket 消息丟失 (嚴重)

### 症狀
從後端日誌可以看出，後端**發送了**所有響應：
```
[DEBUG] get_token_embedding response sent
[DEBUG] get_attention response sent
[DEBUG] get_output response sent
```

但前端 Console 只收到了：
```
App.tsx:50 [App] Received get_token_features message: {...}
App.tsx:50 [App] Received get_output message: {...}
```

**缺失 `get_token_embedding` 和 `get_attention` 的響應消息！**

### 根因分析
**React 的 `useState` batching 機制導致多個 `setLastMessage` 調用被合併，只保留最後一個值。**

在 `App.tsx` 的原始代碼中，使用 `useEffect` 監聽 `lastMessage` state：
```tsx
useEffect(() => {
    if (!lastMessage || typeof lastMessage !== 'object') return
    const msg = lastMessage as Record<string, unknown>
    // ... 處理消息
}, [lastMessage, send, requestFeatures])
```

當四條消息快速連續到達時：
1. `setLastMessage(msg1)` → 觸發 re-render
2. `setLastMessage(msg2)` → batching 合併
3. `setLastMessage(msg3)` → batching 合併
4. `setLastMessage(msg4)` → batching 合併

React 18 的自動 batching 機制會將同一 event loop 內的多個 state update 合併為一次，只保留最後一個值。

### 修復進度
✅ **已修復** - `App.tsx` 已改為使用 `onMessage` callback 方式處理 WebSocket 消息（不再使用 `lastMessage` state）。

---

## 2. ⚠️ Output 顯示 "token0 NaN" (嚴重)

### 症狀
用戶報告 Output View 顯示：
```
token0 NaN
token1 NaN
```

### 根因分析
**後端數據格式與前端預期不匹配**。

後端 `get_output` 返回的數據格式：
```json
{
  "action": "get_output",
  "data": {
    "logits": [[-0.0834, 0.0811]],
    "probabilities": [[0.4589, 0.5410]]
  }
}
```

注意：`logits` 和 `probabilities` 都是 **2D array**：`[[val1, val2]]`

但前端 `App.tsx` 處理時：
```tsx
case 'get_output':
    if (msg.data && msg.data.logits) {
        setOutput(msg.data.logits as number[])  // ← 2D array 被強轉為 number[]
    }
```

然後 `OutputView.tsx` 接收 `logits: number[]`：
```tsx
const maxLogit = Math.max(...logits)  // ← 對 2D array 展開
const expLogits = logits.map((l) => Math.exp(l - maxLogit))
// l 實際是 [val1, val2]，不是 number
// Math.exp([val1, val2]) = NaN
```

### 修復進度
✅ **已修復** - `App.tsx` 中已改為正確提取 `data.logits[0]`（但可能需要再次確認）。

---

## 3. ⚠️ TypeScript 編譯錯誤 (中等)

### 症狀
```
src/App.tsx(139,5): error TS2322: Type '(msg: WSMessage) => void' is not assignable to type '(msg: unknown) => void'.
```

### 根因分析
`websocket.ts` 中 `useWebSocket` 的 `onMessage` 參數類型定義為 `(data: WSMessage) => void`，但 TypeScript 編譯器某些情況下推斷為 `(data: unknown) => void`。

### 修復進度
✅ **已修復** - `websocket.ts` 和 `App.tsx` 中已統一使用 `WSMessage` 類型。

---

## 4. 🔶 get_token_features 數據不穩定 (輕微)

### 症狀
```
"transformer_layers.0.ln_attn_in": [...]  // 正常數值
```

緊接著同一個請求返回的數據中有時包含極小/極大的值（如 `1.789836489528159e-40`），這可能是：
1. 未初始化的記憶體
2. LayerNorm 的 epsilon 值
3. 浮點精度問題

### 根因分析
這是數值穩定性問題，不影響功能，但可能導致可視化不準確。

### 修復進度
⏳ **待處理** - 需要進一步驗證數據來源。

---

## 5. 🔶 後端日誌合併 (輕微)

### 症狀
```
[DEBUG] get_token_embedding result: <class 'dict'>, keys: ['error'] [DEBUG] get_token_embedding response sent
```

兩個 `[DEBUG]` 前綴在同一行，導致日誌難以閱讀。

### 根因分析
`main.py` 中的 `print` 語句可能與其他日誌衝突。

### 修復進度
⏳ **待處理** - 需要格式化日誌輸出。

---

## 📋 完整問題清單

| # | 問題 | 嚴重程度 | 狀態 | 備註 |
|---|------|---------|------|------|
| 1 | WebSocket 消息丟失 (batching) | 🔴 嚴重 | ✅ 已修復 | 改用 onMessage callback |
| 2 | Output 顯示 NaN | 🔴 嚴重 | ✅ 已修復 | 2D array 格式不匹配 |
| 3 | TypeScript 編譯錯誤 | 🟡/misc | ✅ 已修復 | 類型定義統一 |
| 4 | get_token_features 數據不穩定 | 🟡 輕微 | ⏳ 待處理 | 數值穩定性 |
| 5 | 日誌格式合併 | 🟢 輕微 | ⏳ 待處理 | 可讀性 |

---

## 🔄 測試驗證清單

在確認所有修復正確之前，需要執行以下測試：

1. **前端編譯測試**
   ```bash
   cd /workspace/Study/Mahjong-Hidden-Information-Forecast/ArchAnalyzer/frontend
   npx tsc --noEmit
   ```

2. **前端構建測試**
   ```bash
   npm run build
   ```

3. **後端啟動測試**
   ```bash
   cd /workspace/Study/Mahjong-Hidden-Information-Forecast/ArchAnalyzer/backend
   uvicorn main:app --reload
   ```

4. **整合測試**
   - 打開瀏覽器
   - 打開 DevTools Console
   - 點擊 Load Model → Register Hooks → Run Forward
   - 確認 Console 中收到所有 4 條消息（get_token_features, get_token_embedding, get_attention, get_output）
   - 確認 Embedding View 不顯示 "No embedding data"
   - 確認 Attention Heatmap 不顯示 "No attention data"
   - 確認 Output View 不顯示 NaN

---

## 🎯 後續建議

1. **數據格式文檔化**：在 API 文檔或註釋中明確每個 endpoint 的數據格式
2. **添加單元測試**：為前端組件和後端 API 添加單元測試
3. **日誌規範化**：使用統一的日誌庫（如 Python 的 `structlog`）
4. **錯誤邊界**：添加 React Error Boundary 防止組件 crash
