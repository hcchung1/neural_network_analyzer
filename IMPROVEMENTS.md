- **1. 修復 vite.config.ts hardcoded port 問題**
  - 改成從環境變數讀取 `FRONTEND_PORT` 和 `BACKEND_PORT`
  - 使用函數形式的 `defineConfig` 來動態設定 proxy target
  - 添加 `changeOrigin` 和 `secure: false` 以支援代理

- **2. 為 WebSocket 加入 auto-reconnect with exponential backoff**
  - 增加 `MAX_RECONNECT_ATTEMPTS` 和 `BASE_RECONNECT_DELAY`
  - 在 `onclose` 時自動重連，delay 會 exponential backoff
  - 清理 `reconnectTimer` 避免 memory leak

- **3. 改進 HookRegistry 記憶體管理**
  - 加入 thread-safe 操作（`threading.RLock()`）
  - 增加記憶體上限 `_max_memory_mb` 和 `_check_memory_limit()`
  - 使用 `OrderedDict` 實作 LRU eviction

- **4. 強化後端 WebSocket 錯誤處理**
  - 增加 `_send_error` 和 `_handle_load_model` 等 helper
  - 使用 `try...except` 包覆所有 action handler
  - 增加 `traceback.print_exc()` 方便除錯

- **5. 修正前端 WebSocket URL 連線問題**
  - 改成使用 `window.location.host` 建立相對路徑
  - 在 development 模式下走 Vite proxy，production 走 same host

- **6. 修復 Load Model / Run Forward 功能**
  - `load_model` 成功後 **自動** `register_hooks`
  - `run_forward` 成功後 **自動** `requestFeatures`
  - 用戶不再需要手動按三個按鈕

- **7. 改善 start.sh 啟動流程**
  - 明確 export `BACKEND_PORT` 和 `FRONTEND_PORT`
  - 增加後端 health check (`curl http://localhost:$BACKEND_PORT/health`)
  - 等待時間從 2 秒延長到 3 秒，讓後端有更多啟動時間

- **8. TypeScript 編譯修復**
  - 修正 `websocket.ts` 中 `send` 的型別（改為 `any`）
  - 修正 `AttentionHeatmap.tsx` 的型別轉換錯誤
  - 確保 `npx tsc --noEmit` 通過

- **9. 未來改善方向**
  - 建立單元測試覆蓋 `model_engine` 與 `hooks` 模組
  - 為前端添加更多錯誤提示（如 model loading timeout）
  - 考慮使用 React Query 或 SWR 來管理 WebSocket 資料狀態
