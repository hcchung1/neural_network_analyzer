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
- 每個工作區最多 30 位線上參與者。可見頁面約每 1 秒、背景頁面每 5 秒輪詢；
  revision 未更新時只傳心跳，presence 每 3 秒最多寫入一次，失敗會退避重試。
- 主持人操作由伺服器驗證。加入憑證以雜湊保存；只有房間 ID 無法直接讀取狀態，
  但持有分享連結的人可以自行加入。這是內部研究工具的連結邀請模式，沒有帳號登入。
- 協作推論使用獨立模型與 hooks，不改寫單人模式的全域模型。
  個人、協作、比較與 Torchlens 共用 FIFO 計算准入；協作模型使用有界常駐快取。
  無 checkpoint 的 dummy 模型在同一快取租約期間重用；到期後重新建立，不作為可重現的實驗依據。
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

## 效能、GPU 與執行環境

需要 Python 3.11+ 與 Node.js 18+。`start.sh` 在啟動前檢查 Python；可用
`ANALYZER_PYTHON=/path/to/python3.11 bash start.sh` 指定環境。

模型載入時選擇 `cuda` 可使用伺服器的 NVIDIA GPU，必須安裝支援 CUDA 的 PyTorch。
CPU 是可用的選項；沒有可用 CUDA 時會回傳可處理的錯誤。
個人分頁用各自的 `model_handle` 存取模型與分析結果；協作依 checkpoint 絕對路徑、
大小、修改時間、manifest、phase 與 device 共用常駐模型，檔案改變會建立新版本。
模型租約保護正在執行的推論；每次請求結束清除 attention，避免跨工作區殘留。

部署使用單一 Python process／單一 Uvicorn worker。個人、協作、比較與 Torchlens
共用 FIFO 計算准入，一次執行一個重型工作；額滿或等待逾時回傳 503 與 `Retry-After`。
這次優化減少重複載入、序列化及瀏覽器工作，未改用混合精度或合併推論批次。
目前環境只驗證 CPU；GPU 延遲與 VRAM 峰值需在實際部署機器量測。

以下設定在啟動 Python 前生效：

| 變數 | 預設 | 意義 |
| --- | --- | --- |
| `ANALYZER_SHARED_MODEL_TTL_SECONDS` | 300 | 協作快取閒置期限（秒） |
| `ANALYZER_MAX_SHARED_MODELS` | 2 | 協作常駐模型數量上限 |
| `ANALYZER_SHARED_MODEL_MEMORY_MB` | 2048 | 協作模型參數／buffer 預算（MiB） |
| `ANALYZER_MAX_WAITING_JOBS` | 4 | 一個執行中工作之外的等待名額 |
| `ANALYZER_COMPUTE_WAIT_SECONDS` | 120 | 計算名額等待期限（秒） |

個人模型另外使用 `ANALYZER_MODEL_TTL_SECONDS=1800`、`ANALYZER_MAX_MODELS=4`、
`ANALYZER_MODEL_MEMORY_MB=2048`。上述預算不包含建構暫存、attention、輸出及其他
CUDA 配置；調整上限時需保留額外 RAM／VRAM。

二進位樣本與 info.txt 首次建立持久 SQLite 索引，之後僅讀取指定記錄／行；
大小或修改時間改變會重建。索引保存在 `backend/data/binary_sample_index.sqlite3`，
損壞時自動重建。首次掃描二進位檔使用磁碟暫存快照，避免來源截斷造成 mmap 崩潰。
CSV 僅渲染可見列與 overscan；attention 使用 canvas，完整原始輸出可展開、複製或下載。
訓練結果按最多四個請求匯入，大型結果存 IndexedDB 並相容舊 localStorage。
資料來源清單一次取得 artifacts，健康狀態顯示實測 RTT。

新增檢查可在啟動獨立測試服務後執行：

```bash
bash tests/test_startup.sh
node --test tests/sidecar-client.test.cjs tests/collaboration-polling.test.cjs tests/performance-helpers.test.cjs tests/performance-ui-helpers.test.cjs tests/catalog-performance.test.cjs
PYTHONPATH=backend:. python3 -m unittest discover -s backend -p 'test_*.py'
npx tsc --noEmit
npm run build
PLAYWRIGHT_BASE_URL=http://127.0.0.1:3001 npx playwright test
```

實作與驗證記錄：`docs/superpowers/plans/2026-10-04-gpu-performance-and-concurrency.md`。
