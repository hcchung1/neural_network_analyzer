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
- 📊 **模型分析**: Attention、完整原始輸出與輸出 shape；Embedding 依分析服務 capability 顯示支援狀態
- 🔄 **多人協作**: 分享工作區、獨立瀏覽／跟隨主持人、共同聊天與標註
- 🎨 **層級探索**: 透過 Slider 切換不同 Transformer 層級
- 📈 **完整輸出**: 保留全部數值，不依維度猜測 logits、機率或類別名稱

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
- 個人分頁與協作推論使用各自獨立的模型與 hooks。
  個人、協作、比較、Torchlens 共用計算准入，同時執行一個重型工作。
  協作模型使用有界常駐快取，減少重複載入。
  無 checkpoint 的 dummy 模型在同一快取租約期間重用；到期後重新建立，不作為可重現的實驗依據。
- 本版以單機、持久磁碟部署為範圍。多台主機需改用共享資料庫及推論佇列。
  網際網路部署前需另加登入與檔案存取政策；目前原有檔案 API 面向可信任的研究環境。

### 個人模式與生命週期

個人模型載入後回傳隨機 `model_handle`，只保留在目前分頁記憶體。
後續推論、分析結果讀取／刪除及 Torchlens 使用 `X-Model-Handle`；缺失或到期會要求重新載入。
新模型成功載入才替換舊模型並清除圖表；失敗保留舊模型。
按 **Release Model** 可主動釋放。離開頁面或加入協作時嘗試釋放；網路中斷由 TTL 回收。
重新整理個人頁面需重新載入模型，分享需主動加入共同工作區。

部署必須使用 **單一 Python process／單一 Uvicorn worker**，不能加 `--workers 2` 或多副本負載平衡。
模型 handle、CSV session 與計算鎖皆保存在該 process；重啟後失效。
區網使用 `http://<伺服器區網IP>:3001`；`0.0.0.0` 是監聽位址。

以下環境變數在啟動 Python 前設定，修改後需重啟：

| 變數 | 預設 | 意義 |
| --- | --- | --- |
| `ANALYZER_MODEL_TTL_SECONDS` | 1800 | 個人模型閒置期限（秒） |
| `ANALYZER_MAX_MODELS` | 4 | 個人常駐模型上限 |
| `ANALYZER_MODEL_MEMORY_MB` | 2048 | 保留模型的參數／buffer 預算（MiB） |
| `ANALYZER_MAX_WAITING_JOBS` | 4 | 一個執行中工作之外的等待名額 |
| `ANALYZER_COMPUTE_WAIT_SECONDS` | 120 | 等待計算名額逾時（秒） |
| `ANALYZER_RESULT_TTL_SECONDS` | 3600 | 分析結果閒置期限（秒） |
| `ANALYZER_MAX_RESULTS` | 100 | 保留分析結果上限 |
| `ANALYZER_RESULT_MEMORY_MB` | 256 | 序列化分析結果預算（MiB） |
| `ANALYZER_CSV_TTL_SECONDS` | 1800 | CSV session 閒置期限（秒） |
| `ANALYZER_MAX_CSV_SESSIONS` | 32 | CSV session 上限，包含正在開啟／尚有讀取的關閉中 session |

模型／結果額滿會回傳 503 `MODEL_CAPACITY`／`RESULT_CAPACITY`；計算額滿或等待逾時為
503 `COMPUTE_BUSY`，附 `Retry-After`。不會為了新請求靜默驅逐其他人的有效模型或結果。
CSV 容量錯誤沿用 reader 的 `{success:false,error,error_code:"CSV_CAPACITY"}` 格式。
CSV 到期後顯示重新載入提示；關閉 session 後拒絕新讀取，進行中的讀取／圖片串流完成才刪除解壓目錄。
每 60 秒執行清理，資源存取時也會檢查期限；執行中租約不會被 TTL 清除。

參數／buffer 預算不等於推論尖峰 RAM／VRAM 上限；載入新舊模型並存、attention、Torchlens
與比較的暫存仍需額外空間。預設值不是 GPU 壓力驗收結果，應依部署機器調整。

目前個人與協作輸出都是 `raw`：`output_sample`（歷史欄位名稱）包含完整 tensor 的
row-major 展平值，`output_shape` 保留原始維度；不截成前五類，也不自動轉機率。
目前分析 API 未提供 Embedding probe，畫面會明確說明，不把架構具備 embedding 當成已捕捉資料。
AI Assistant 使用 `page_context`／`reply` 契約，模型上下文切換會清除舊對話並忽略遲到回覆。
實際 AI 回覆仍需後端既有語言模型供應商設定；自動測試替換外部供應商，不呼叫付費服務。

更新程式碼後，請在原本執行 `start.sh` 的終端機以 Ctrl+C 停止，再執行 `bash start.sh`。
Next.js 開發模式的熱更新不會重載 Python；前後端必須同時更新。

## 驗證

先啟動上述兩個服務，再執行：

```bash
npx playwright install chromium
npx playwright test
npx tsc --noEmit
PYTHONPATH=backend:. python -m unittest discover -s backend -p 'test_*.py'
node --test tests/sidecar-client.test.cjs
bash tests/test_startup.sh
npm run build
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
