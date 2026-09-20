# ArchAnalyzer

Transformer 模型分析工作台 — Next.js App Router、FastAPI 與 PyTorch。

## 啟動方式

### 本地開發

```bash
npm install
pip install -r backend/requirements.txt
bash start.sh
```

### 透過 SSH Port Forwarding 遠端部屬

```bash
# 在遠端伺服器上啟動服務
ssh user@remote-server
bash start.sh

# 在本機終端機執行 port forwarding
ssh -L 3001:localhost:3001 user@remote-server
```

之後在本機瀏覽器開啟 `http://localhost:3001` 即可。

## 功能特色

- 🔬 **單一 Token 視覺化**: 深入檢視 Transformer 中單一 token 的資訊流
- 📊 **多維度分析**: Input Features、Embedding、Attention、Output
- 🔄 **多人協作**: 分享工作區、獨立瀏覽／跟隨主持人、共同聊天與標註
- 🎨 **層級探索**: 透過 Slider 切換不同 Transformer 層級
- 📈 **機率分布**: 視覺化最終輸出的機率分布

## 技術架構

- **Backend**: FastAPI + PyTorch + Uvicorn
- **Frontend**: Next.js + React + TypeScript
- **通訊**: REST API；協作使用增量輪詢
- **部署**: SSH Port Forwarding

## 多人混合協作

1. 在 Explore 頁面「共同協作」輸入暱稱，按「建立工作區」。
2. 按「分享連結」，其他人開啟連結並輸入暱稱加入。
3. 主持人載入共同模型與二進位樣本，所有人取得相同的分析結果。
4. 預設各自切換 token、layer。按「跟隨主持人」同步主持人的位置；
   自己手動切換位置會停止跟隨。
5. 使用協作區的聊天或「標註目前位置」共同討論。標註可跳到對應 token／layer；
   更換模型或樣本後，舊分析的標註仍保留，但不跳到新分析的同名位置。
6. 主持人可以交接給線上成員；主動離開時交接給下一位線上成員。
   主持人關閉瀏覽器、超過約 15 秒離線後，其他人可按「接任主持人」。

右側原有 AI Assistant 保持個人對話；工作區內的聊天與標註才是共享討論。
重新整理保留加入身分與共同分析。各分頁的憑證存在 `sessionStorage`；
清除瀏覽器資料或關閉分頁後，可重新以暱稱加入。

跨電腦使用時，先以所有人都能存取的伺服器 IP／網域開啟網頁，再複製連結。
`127.0.0.1` 只指向每個人自己的電腦，不可直接分享給另一台電腦。
前端預設埠為 3001、後端為 8080，可使用 `FRONTEND_PORT`、`BACKEND_PORT` 調整。
Next.js 使用 `PYTHON_SIDECAR_URL` 連線後端（預設 `http://127.0.0.1:8080/internal/v1`）。

### 儲存與部署範圍

- 工作區、共享分析與最近 500 則聊天／標註存於 `data/collaboration.sqlite3`。
  SQLite 使用 WAL 與交易，重新啟動網頁服務仍保留共同資料。
- 每個工作區最多 30 位線上參與者。約每 900 ms 輪詢共同狀態，分析未更新時
  不重傳矩陣；斷線後自動重連並取得最新狀態。
- 主持人操作由伺服器驗證。加入憑證以雜湊保存；只有房間 ID 無法直接讀取狀態，
  但持有分享連結的人可以自行加入。這是內部研究工具的連結邀請模式，沒有帳號登入。
- 協作推論使用獨立模型與 hooks，不改寫單人模式的全域模型。
  每個 Python process 同時執行一個協作推論；模型按請求載入，限制常駐 GPU 記憶體。
  無 checkpoint 的 dummy 模型是每次重新建立的測試模型，不作為可重現的實驗依據。
- 本版以單機、持久磁碟部署為範圍。多台主機需改用共享資料庫及推論佇列。
  網際網路部署前需另加登入與檔案存取政策；目前原有檔案 API 面向可信任的研究環境。

## 驗證

先啟動上述兩個服務，再執行：

```bash
npx playwright install chromium
npm run test:collaboration
npx tsc --noEmit
PYTHONPATH=backend:. python -m unittest backend.test_collaboration backend.test_csv_reader_neighbors backend.test_compatibility
```

`PLAYWRIGHT_BASE_URL` 可指定另一個網頁位置。Playwright 使用兩個獨立瀏覽器 context，
涵蓋獨立瀏覽、跟隨、手動停止跟隨、聊天、標註、重新整理、權限拒絕、跨工作區隔離、
過期操作拒絕、斷線重連、主持人交接與離開。測試以固定的合成 binary source 回應
供應樣本，模型載入、PyTorch 推論、工作區 API、資料儲存與圖表顯示均使用真實服務。
失敗時的截圖與 trace 會放在 `test-results/`。
