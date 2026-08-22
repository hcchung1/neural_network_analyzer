# ArchAnalyzer 專案手冊（新 Session  onboarding）

> **用途**：新對話 session 可先讀此檔，快速掌握目錄結構、功能、API、持久化與近期修改脈絡。  
> **專案根目錄**（本檔所在）：  
> `/workspace/Study/Mahjong-Hidden-Information-Forecast/ArchAnalyzer`

---

## 1. 專案定位

ArchAnalyzer 是 **Mahjong Transformer 分析用全端網頁**：

| 模組 | 說明 |
|------|------|
| **Transformer 視覺化** | 載入 checkpoint、WebSocket 即時 forward、顯示 input / embedding / attention / output |
| **Training Results** | 掃描訓練輸出資料夾，比較 `*_history.csv`、Phase Analysis、Base vs Random turn metrics（Plotly） |
| **CSV Reader** | 從 [Tenhou_csvtowebsite](https://github.com/hcchung1/Tenhou_csvtowebsite) 概念移植：大檔分頁、filter、ZIP+PNG、Tenhou link 預覽 |

**技術棧**：FastAPI + WebSocket（backend）、React + Vite + TypeScript + react-plotly.js（frontend）。

**相關專案路徑**（同 repo 上層）：

- `../Transformer/` — `MahjongTransformer`、`transformer.py`、訓練輸出 `output/`
- `../PhaseAnalyzer/` — `phaseAnalyzer.py`，產生 `*_phase_summary.csv`、`*_turn_metrics.csv`、`*_base_vs_random_turn_metrics.png` 等

---

## 2. 目錄結構

```
ArchAnalyzer/
├── ARCHANALYZER_HANDBOOK.md    # 本檔
├── backend/
│   ├── main.py                 # FastAPI 入口、/ws/visualize、CORS、router 掛載
│   ├── requirements.txt
│   ├── api/
│   │   ├── model.py            # 模型相關 REST（若有的話）
│   │   ├── visualize.py        # 視覺化 REST
│   │   ├── visualize_chat.py   # LangChain Deep Agent chatbot API
│   │   ├── training.py         # scan / read_file
│   │   ├── csv_reader.py       # open / page / image / close
│   │   └── binary_samples.py   # 掃描 .bin、搜尋/解析單筆樣本、Tenhou URL
│   ├── model_engine/
│   │   └── transformer_model.py  # 單例載入 MahjongTransformer、forward、dummy fallback
│   └── hooks/
│       └── registry.py         # forward hooks、快取、attention 等中間特徵
└── frontend/
    ├── package.json
    ├── vite.config.ts          # dev proxy: /api、/ws → BACKEND_PORT（預設 8080）
    ├── src/
    │   ├── App.tsx             # 三頁切換、WS、側欄按鈕
    │   ├── utils/websocket.ts
    │   ├── components/         # 上述視覺元件 + BinarySampleLoader
    │   └── pages/
    │       ├── TrainingResultsPage.tsx
    │       └── CsvReaderPage.tsx
```

---

## 3. 前端三頁與路由（重要）

`App.tsx` 使用 `currentView: 'visualize' | 'training' | 'csvReader'`。

**頁面切換不 unmount**：三個主內容區用 `display: block | none` 切換，避免 CSV Reader / Training 狀態在切頁時消失。

全域左側欄只負責三頁導航：**Token Visualization**、**Training Results**、**CSV Reader**。模型路徑、Load Model、Register Hooks、Run Forward、Token Index 與 Layer selector 都位於 Token Visualization 主內容的 **Visualization Controls** 卡片，不會在其他頁面占用左欄。

左側欄預設固定顯示；標題列的切換按鈕可改為自動隱藏模式。自動隱藏時只保留左側窄 hover rail，滑鼠靠近或以鍵盤 focus 進入側欄就會滑出，並可按 pin 按鈕恢復固定。

---

## 4. Backend API 一覽

| Prefix | 端點 | 用途 |
|--------|------|------|
| `GET /health` | 健康檢查、`model_loaded` |
| `/api/model` | 模型載入等（見 `api/model.py`） |
| `/api/visualize` | 視覺化 REST（見 `api/visualize.py`） |
| `/api/visualize_chat` | chat / tenpai_quick（LangChain Deep Agent） |
| `POST /api/training/scan` | 遞迴掃描資料夾內 `*_history.csv`、`*_phase_summary.csv`、`*_turn_metrics.csv`、`comparison_table.csv` |
| `POST /api/training/read_file` | 讀單一 CSV 全文（UTF-8） |
| `POST /api/csv_reader/open` | 開啟 server 路徑 `.csv` / `.zip`，建立 session；回傳可辨識的相關 model / training folder |
| `POST /api/csv_reader/page` | 分頁 + filter |
| `GET /api/csv_reader/image/{session_id}/{image_index}` | ZIP 內 PNG |
| `POST /api/csv_reader/close` | 關閉 session |
| `POST /api/csv_reader/output_index/scan` | 手動重建 `Transformer/output` 內 `.csv/.zip` 的 SQLite 搜尋索引 |
| `GET /api/csv_reader/output_index/status` | 讀取 SQLite 中持久化的索引筆數與最後掃描時間，不回傳搜尋結果 |
| `GET /api/csv_reader/output_index/search?query=...` | 依資料夾／檔名搜尋索引，最多回傳 100 筆 server 絕對路徑 |
| `GET /api/binary_samples/files?directory=...` | 掃描 server 資料夾第一層的 `.bin` 檔案 |
| `POST /api/binary_samples/search` | 依檔名、record index／line number、pred id 解析單筆 binary sample |
| `WS /ws/visualize` | `load_model`、`register_hooks`、`forward`、`get_token_features`、`get_token_embedding`、`get_layer_input`、attention / output 等 |

**修改 backend 後需重啟** backend process，前端 dev 通常會透過 Vite proxy 連到 `BACKEND_PORT`。

---

## 5. WebSocket 視覺化（常見問題）

- 前端 URL：`ws(s)://{host}/ws/visualize`（`App.tsx` + `vite` proxy `/ws`）
- 側欄若長期顯示 **`WS: connecting`**：多半是 backend 未啟動、port 不對、或 proxy 未轉發 WebSocket
- 開發模式保留 React `StrictMode`。`useWebSocket` cleanup 必須以個別 `WebSocket` instance 是否仍為 `wsRef.current` 判斷是否重連；不可用跨 socket 共用的 manual-close flag。否則 StrictMode 關閉測試連線後，舊連線的延遲 `onclose` 會誤排 reconnect，形成多條幽靈連線，log 會反覆出現 `accepted`／`Client disconnected`，並讓頁面看似一直重新啟動。

**模型**：

- `TransformerModelEngine` 會嘗試從 `../Transformer/transformer.py` 載入 `MahjongTransformer`
- 自訂 `MultiHeadAttention` 若用 `F.scaled_dot_product_attention` 可能**沒有 attention weights**；專案內曾改為 eval 時重算並存 `last_attention_weights`，並在 `forward()` 後收集 — 細節見 `transformer_model.py` 與 `Transformer/transformer.py`

**Hooks**：`HookRegistry` 對 embedding / attention / ffn / output 等模組註冊 forward hook。

- `Load Model` 的 `n_layers` 取自實際 `model.transformer_layers` 長度，不依賴 `MahjongTransformer` 未保存的 `self.n_layers`；前端將 layer count 轉成 zero-based slider index，因此六個 Encoder Block 的選擇範圍是 `0–5`。
- `d_model` 由 checkpoint 建構出的模型 metadata 回傳；每個 Encoder Block 另註冊 forward pre-hook，供 `get_layer_input` 讀取所選 token 在進入該 block 前的 hidden state。

### 自訂 Feature Forward

- `Load Model` 會從 `.pth` state dict 推斷模型架構：sequence length 主要取自 `pos_encoding.pe`，feature dimension 取自 heterogeneous-token 的 `proj_init.weight` / `proj_act.weight`，或 single-projection 的 `input_projection.weight`。
- 若 checkpoint projection 與目前 runtime 全域 feature 設定不同，engine 會依 `.pth` 權重重建該 model instance 的 `proj_init` / `proj_act` 或 `input_projection`，再載入 state dict；不可用 `strict=False` 略過輸入投影權重後仍宣稱模型已正確載入。
- Backend 在 `model_meta.input_schema` 回傳 dense tensor contract：`dtype=float32`、batch size、`seq_len`、`feature_dim` 與完整 shape；若 checkpoint／模型沒有可推斷的 dense schema，前端 Custom Feature 欄位會保持 disabled 並顯示 schema unavailable。
- **Custom Feature (JSON)** 接受兩種格式：
  - 2D：`[seq_len][feature_dim]`，送出時自動補成 batch size 1。
  - 3D：`[1][seq_len][feature_dim]`。
- JSON 必須全部是有限數值。前端先檢查 JSON、batch、row count、每列 feature count；backend 轉成 `float32` tensor 後再次檢查 rank 與模型 shape，錯誤會顯示 expected / received dimensions，不執行模型 forward。
- Custom Feature 留空時，`Run Forward` 不再使用前端硬編碼的 `[1, 49, 12308]`，改由 backend 按已載入模型的 `seq_len` / `input_features` 產生隨機測試輸入。
- 2026-07-11 真實 checkpoint 驗證：`Transformer/test_0403152548/...20260403_152548__all.pth` 推斷為 add-mode `proj_init=(512,1258)`、`proj_act=(512,463)`、輸入 `float32 [1,92,1258]`，state dict 以 `strict=True` 載入；錯誤 shape 被拒絕，合法 `[92,1258]` 與 `[1,92,1258]` 都成功 forward 並產生 features、attention、output。

### Binary Sample Search 與實際盤面

- 視覺化主內容的 **Binary Sample Search** 位於 **Custom Feature (JSON)** 正上方，不占用左側控制欄；預設掃描 repo root 的 `cache_100`，也可輸入其他 server 資料夾。API 只列出該層 `.bin`，不遞迴。
- 選定檔案後可用 zero-based `record index` 取樣；若填 `line`，會在候選檔中比對 binary header 的 `line_number`。`pred id` 0/1/2 會套用與 dataset 相同的目標對手 one-hot。
- Backend 沿用 `Transformer/utils.py` 的 record-size detection、record parser 與 token packer，不自行複製 binary layout。回傳 metadata 包含 file、record、line、label、valid length 與 feature shape。
- Binary parser 會以目前載入模型的 `seq_len` 作為 physical packing length。`last_n_actions` 模式下會對應成 `seq_len - 1` 個 action slots，因此 `[49,...]` checkpoint 會重新解析最多 48 個 action，而不是先按目前 runtime 預設截成 24 個再補長度。
- New_Add cache 使用 compact init + sidecar 時，若 checkpoint contract 是舊式 dense width，backend 會把 sidecar 還原至 init token，並將 action token 放入 dense tensor；例如 `[25,986]` runtime storage 可正確轉成模型需要的 `[49,12308]`。此轉換不是單純 padding：sequence 會按模型長度重新 pack，init-only 資訊也會完整還原。
- 轉換後 shape 與目前載入模型的 `model_meta.input_schema` 完全相同時，前端才把 feature 送入 `run_forward`；無法安全轉換的 cache/config 仍會顯示 parsed/expected shape 並拒絕 forward，避免靜默截斷或不完整 padding。
- 相容 sample 載入後，前端會更新右側 Tenhou 盤面、把同一份 2D feature 以 JSON 填入 **Custom Feature (JSON)**，再送入 `run_forward`；使用者可直接檢查或修改該輸入後重新執行。重新載入模型時仍會清空舊 feature，避免沿用不相容 schema。
- **Load Sample 與視覺化同步**：`compatible=true` 時會自動 WS `run_forward`（input 為 `[1, seq, feat]` 3D batch）。載入新樣本時會先清空 Input／Embedding／Attention／Output，避免 forward 失敗或 WS 未送出時仍顯示上一筆結果。`run_forward` 前 backend 會 `registry.clear_cache()`，避免 hook cache 殘留舊 forward 的數值。
- **僅盤面變、底下圖不變** 常見原因：`compatible=false`（只更新 Tenhou、不送 feature）、未 **Load Model**、WS 未連線、或 shape mismatch；狀態列應顯示 warning／錯誤，而非靜默沿用舊圖。
- 同資料夾有 `info.txt` 時，以 binary 的 line number 讀取對應 JSON，產生與 result CSV 同格式的 `https://tenhou.net/6/#json=...&ts=0` URL。
- 右側 **Current Tenhou Board** 以 iframe 顯示目前樣本盤面，採用與 CSV Reader「檔案庫」一致的 backdrop blur、右側滑入動畫、圓角控制與陰影語彙；可關閉、從右上現代化「盤面 / Tenhou」膠囊重新展開、外部開啟。drawer 左側分隔條仍可拖曳，寬度限制 300–1200px；盤面改為 overlay drawer，不再僵硬地永久壓縮主內容。
- Current Tenhou Board 分隔條使用 Pointer Events、pointer capture 與拖曳期間的全頁透明 overlay，避免游標進入 iframe 後事件遺失。密集 `pointermove` 只記錄最新座標，每個 animation frame 直接更新一次右欄 DOM width；放開時才提交一次 React `boardWidth` state，因此不會在拖曳途中反覆 render 主視覺內容與 iframe。
- **Tenhou Board iframe 必須以 `key={boardUrl}` 重建**：Tenhou URL 的主要差異在 hash (`#json=...`)，同一 origin 的 iframe 不會因 src hash 變化而自行重載頁面。React `key={boardUrl}` 可強制在每次 boardUrl 變更時建立全新 iframe，確保切換 .bin 檔案或 record index 時盤面正確更新。
- 2026-07-11 smoke test：真實 `cache_100/labeled_part_0000001.bin` 被偵測為 34 records、record size 1424；record 0 解析為 line 1、valid length 2、`[25,986]`，並成功產生 Tenhou URL。傳入 `[92,1258]` 模型 contract 時正確回報不相容且不回傳 feature。
- 2026-07-11 本次 UI 驗證：`npm run build`（TypeScript + Vite production build）通過；Binary Sample Search DOM 順序確認位於 Custom Feature 前，相容 feature callback 會填入 textarea；盤面 resize 具備 RAF、overlay、pointerup／pointercancel／lost capture／window blur 清理路徑。
- 2026-07-11 dense conversion smoke test：真實 `cache_100/labeled_part_0000001.bin` record 0 搭配模型 contract `[49,12308]`，API 回傳 `compatible=true`、shape `[49,12308]`、conversion `new_add_compact_to_dense`，每列寬度與 49 列長度皆驗證正確。

### Input / Embedding 互動檢視

- **Input Features** 只顯示目前 Token selector 所選 token 的 raw feature row，例如 Token 0 對應 `[batch=0, token=0, :]`，不會同時混入其他 token。標題會顯示 token index 與 raw feature 數量；內容使用可展開／收合的 details 控制，方便暫時隱藏大型 Raw Input。每格保留顏色與短格式數字總覽，hover / keyboard focus 時顯示 feature index 與高精度值。
- **Embedding Vector** 是所選 raw token 經 `proj_init`／`proj_act`／`input_projection` 投影進 `d_model` 後的向量；它位於 positional/meta encoding 與 Encoder Block 0 **之前**，不是任意所選 layer 的 input/output hidden state。每個 dimension hover/focus 時會加寬、加框並顯示高精度實值。
- **Layer Input Vector** 位於 Input Features 下方、Embedding Vector 上方，顯示所選 token 在進入目前選定 Encoder Block 前的 hidden state。Layer 0 input 已包含模型在 projection 後套用的 positional/meta processing；Layer N（N > 0）input 是前一個 block 的輸出。向量維度依 checkpoint 的 `d_model` 顯示，不寫死為 512。
- Raw Input、Layer Input Vector、Embedding Vector 都獨占 visualization grid 完整寬度；Attention 與 Output 才維持兩欄排列。
- Input 與 embedding 的互動格都具備 `tabIndex` 與 `aria-label`，鍵盤使用者可取得相同的詳細數值。
- Input Features 與 Layer Input Vector 固定每列 34 格，格寬按面板可用寬度等分自適應。每列左右邊界格以朝內的 transform origin 與 tooltip 對齊方式展開，避免 hover／keyboard focus 的放大格和高精度數值被面板裁切；tooltip 保持約 17px 的實際顯示字級，方便閱讀高精度值。
- 右側「盤面」與「Chat」收合按鈕由 viewport 右緣內縮 20px，避免覆蓋主內容垂直 scrollbar。
- 2026-07-13 真實六層 checkpoint 驗證：slider metadata 為 6 blocks（index `0–5`）；Token 0/1 raw input 各自為 12,308 維且內容不同；Embedding 與 Layer 0–5 input 均為 checkpoint 推斷的 `d_model=512`。Layer 0 input 與 projection embedding 不同、Layer 1 input 與 Layer 0 input 不同。`py_compile`、`npx tsc --noEmit`、Vite production build 與 `git diff --check` 通過。

### Analysis Chat（AI 助手側欄）

- 右側 **「Chat ◀」** 按鈕（位於 board-toggle 下方 68px，綠色系）可展開 chatbot drawer（420px 寬）。
- **🔍 聽牌分析** 快速按鈕採固定三階段：後端先依 binary sample 的真實 `valid_len` 解碼盤面；LLM 在看不到 Transformer output 與 label/oracle 的條件下完成結構化盤面盲測；最後才由 deterministic backend code 加入模型比較與 oracle 評估。LLM 只取得 compact action summary，不會收到或重送完整 packed float matrix。
- `pred_id` 是相對於 observer 的對手 index（0/1/2），不是絕對座位；絕對 `target_seat` 固定以 `(player_id + pred_id + 1) % 4` 算出。API 同時驗證 `observer_seat`、`target_relative_index`、`target_seat` 與 `seat_encoding`，不完整或矛盾時拒絕盤面分析。
- checkpoint 載入後會回傳 `output_schema`。單一 binary logit 使用 sigmoid；兩個／多個 logits 使用 softmax 與明確 `positive_class_index`；已是 probability 時不重算。Output panel 與 Analysis Chat 共用此語意，schema 不明時只顯示 raw output，不猜測機率。
- **自由對話**（mode=chat）會取得後端已解碼的盤面摘要與已定義語意的模型結果；靜態規則獨立放在 SystemMessage，所有頁面資料和 client transcript 都以 user-level quoted data 傳入，避免 page context 提升為 system instruction。client 提供的 assistant history 不會被建立成可信 AIMessage。
- 輸入模型限制 messages 數量／長度、feature token 數／寬度／總值數、矩陣矩形與 finite number、dense schema、valid length、seat 關係及 logits schema。輸入錯誤使用 400/422，provider failure 使用 503；原始 exception 僅寫 server log，不回傳前端。
- ChatOpenAI 與 structured-output runnable 依 provider 設定快取重用，不再每個 request 建立及 compile Deep Agent。
- 需要 backend 已 `pip install -r requirements.txt`（含 `deepagents`、`langchain`、`langchain-openai`）且 `langchain/.env` 設定 `OPENAI_API_KEY`、`MAHJONG_AGENT_MODEL`。
- 檔案：`backend/api/visualize_chat.py`、`frontend/src/components/VisualizationChatSidebar.tsx`。

---

## 6. Training Results 頁

**檔案**：`frontend/src/pages/TrainingResultsPage.tsx`

### 掃描與合併邏輯

- Backend **遞迴** `os.walk`（Phase 常在 `output/analysis/phase_analysis_*/` 子資料夾）
- 多個檔案會 **合併為同一個成果名稱**（`normalizeResultName`）：
  - `*_history.csv`
  - `*_base_results_phase_summary.csv` / `*_random_feature_results_phase_summary.csv`
  - `*_base_results_turn_metrics.csv` / `*_random_feature_results_turn_metrics.csv`
- 分開存：`phaseData` / `randomPhaseData`、`turnMetrics` / `randomTurnMetrics`

### 圖表

1. **一般訓練曲線**（有 `data` 的結果，包含 Normal / Copy ROC AUC）
2. **Phase Analysis**（Base + Random 分色／標籤，支援 CSV 的 `auc` 欄位）
3. **Base vs Random Turn Metrics**：
   - 5 張分 metric 折線圖（Avg Probability、Accuracy、F1、ROC AUC、Positive Rate）
   - 第 5 張 **整合圖**，仿 `PhaseAnalyzer` 的 `*_base_vs_random_turn_metrics.png`（phase shading 0–24 / 24–40 / 40+、雙 Y 軸、6 條線 + count bar）

### 顯示名稱

- **上方成果列表**：完整檔名
- **圖表 legend**：只顯示時間戳 `YYYYMMDD_HHMMSS`（`getChartDisplayName`），找不到則 fallback 全名

### 顏色控制與拖曳效能

- 每筆成果有兩個 `<input type="color">`：`color` 控制 Normal / Base，`copyColor` 控制 Copy / RandomFeature。
- 顏色輸入拖曳時會高頻產生 `input` event。若每次 event 都直接更新 `results`，會同時觸發所有 Plotly 圖重繪，並將完整圖表資料 `JSON.stringify` 後同步寫入 `localStorage`，資料量大時會阻塞主執行緒。
- `ResultColorInput` 已拆成 memoized 子元件：拖曳期間只更新 input 本身，不更新頁面層級的 `results`；停止輸入 **300 ms** 後才提交最後一個顏色，失焦時則立即提交。
- `updateResultColor` 會忽略未變更的顏色，避免無效 render 與持久化。圖表和 `training_results` 只在提交後更新一次。
- 2026-07-10 瀏覽器壓力測試：同時顯示 13 張 Plotly 圖、單筆 2,000 epochs 時連續派送 121 個顏色 input events，事件處理約 9.3 ms；拖曳期間 0 次 `localStorage` 寫入，停止後 1 次，Normal 與 Copy trace 顏色皆正確更新。

### 載入上限

- **程式碼無明確筆數上限**；瓶頸在瀏覽器記憶體、`localStorage`（約 5–10MB）、Plotly 曲線數量
- 持久化 key：`localStorage` → **`training_results`**（含完整解析後的 chart 資料，大資料夾可能撐爆 localStorage）

### 圖表全螢幕

- Training Results 的每一張 Plotly 圖右上角都有 **全螢幕** 按鈕，涵蓋一般訓練曲線、Phase Analysis、4 張 Turn Metrics 與整合 Turn Metrics 圖。
- 使用瀏覽器 Fullscreen API 放大單張圖；全螢幕期間圖表高度填滿 viewport。按 **縮小**、Esc 或瀏覽器的退出全螢幕操作都會復原。
- `fullscreenchange` 後會派送 resize event，讓 `useResizeHandler` 重新計算 Plotly 尺寸；listener 會在 component unmount 時清理。
- 2026-07-11 驗證：13 個圖表位置全部使用共用 `FullscreenPlot` wrapper，TypeScript 與 Vite production build 通過。

### 已知 UI 議題（使用者回報）

- 與 CSV Reader 等頁面並存後，**前中後期合併分析圖寬度**可能被壓縮 — 檢查 Plotly `layout` / 外層 grid、`App.tsx` main 區塊寬度

---

## 7. CSV Reader 頁

**檔案**：`frontend/src/pages/CsvReaderPage.tsx`  
**Backend**：`backend/api/csv_reader.py`（邏輯對齊 Tenhou WPF：分頁讀取、filter parser、ZIP 內 CSV/PNG）

### 功能摘要

- 輸入 **server 絕對路徑** 開 `.csv` / `.zip`
- 多 tab（session）、分頁 200/500/1000/2000、filter（`=`, `>`, `,` 多條件等）
- `tenhou_link` / URL：Ctrl+點外部開啟、雙擊右側 iframe
- ZIP：右側 Training PNG 預覽、縮放
- Copy cell / row（Ctrl+C）
- 每個已開啟檔案的 tab 右側都有獨立 `×` 關閉按鈕
- 可動畫展開的「檔案庫」抽屜列出所有 session；每筆 switch 控制是否顯示於上方快速切換橫幅，隱藏不等於關閉
- **右側面板寬度可拖曳**（分隔條 8px，`col-resize`），範圍 280–1200px，預設 520px
- `Transformer/output` 快速搜尋：輸入資料夾／檔名、點結果只回填路徑欄，仍由使用者手動按 **Open CSV/ZIP**
- 若 `.zip` 內含 `.pth/.pt`，或同資料夾有檔名相符的 companion model，該 CSV tab 的 `×` 左側會顯示 `◈`；點擊後才從右側成果 drawer 顯示 **加入 Token Visualization**／**加入 Training Results**，不占用 Reader 主畫面高度

### Tenhou URL 預覽切換

- 雙擊 URL cell 會更新右側 URL 並打開 web mode。
- iframe 使用目前完整 URL 作為 React `key`；從 URL1 切換到 URL2（包括主要差異位於 hash 的 Tenhou URL）會強制建立新的 iframe，不必先按 Close。

### 檔案庫與橫幅

- 上方「檔案庫」按鈕以滑入 drawer 顯示全部已開啟 `.csv/.zip`，包含 backdrop blur、滑入與 switch 動畫。
- switch 開啟：檔案顯示在上方快速切換橫幅；關閉：只從橫幅隱藏，仍可從 drawer 選取及恢復。
- `pinned` 狀態與 session metadata 一起存入 `csv_reader_metadata`，重新整理後保留。
- drawer 中選取檔案會切換 active session 並收起 drawer；真正釋放 session 仍使用 tab 的 `×`。

### Transformer/output 快速索引

- 實際根目錄固定為 repo 的 `Transformer/output/`（需求文字中的 `Trabsformer` 視為拼字誤植），不接受前端指定其他掃描 root。
- 手動按 **重新掃描索引** 才遞迴掃描後續新增的 `.csv/.zip`；SQLite 位於 `backend/data/csv_reader_index.sqlite3`，已由 `.gitignore` 排除。
- 搜尋同時比對 relative path 與 filename，預設最多顯示 40 筆；點選結果只寫入 CSV Reader 的 path input，不會自動 open。
- SQLite 的 files table 與 `last_scanned_at` metadata 會跨瀏覽器刷新與 backend process 保留。CSV Reader mount 時呼叫 status API 還原「檔案數 + 最後掃描時間」，不再以當次 React state 判斷索引是否存在；因此已建索引刷新後不會錯誤顯示「尚未讀取索引」。
- `output_index/scan`、status/search、CSV open/page/close 都包含同步 filesystem、SQLite 或 CSV I/O，route 必須使用一般 `def`，讓 FastAPI 放進 thread pool；若寫成 `async def` 卻直接執行同步掃描，大型 output tree／CSV 會堵住 event loop，連帶讓 WebSocket 與其他 API 看似卡死。索引重建另以 process-local lock 拒絕同時重複掃描。
- 搜尋結果 dropdown 有獨立 open state。點擊搜尋區域外（包含 Filter、分頁或 table）、按 Esc、修改 query、選取結果或搜尋失敗時都會收起；outside-pointer 與 keyboard listener 只在 dropdown 開啟時註冊，關閉或 component unmount 時清理。
- 2026-07-16 驗證：目前 `Transformer/output` 重建為 1,974 筆，直接掃描約 1.20 秒；持久化 status 的筆數／timestamp 一致，重複掃描 guard 正常，backend `py_compile`、frontend TypeScript 與 `git diff --check` 通過。
- 2026-07-12 驗證：既有 SQLite status 在未重掃下回傳 1,919 筆與持久化 `last_scanned_at`，獨立 search 回傳相同 timestamp；`python -m py_compile main.py api/csv_reader.py`、`npx tsc --noEmit`、Vite production build 與 `git diff --check` 通過。

### 從 CSV Reader 加入模型／訓練成果

- ZIP session 會安全解壓 CSV、PNG 與 `.pth/.pt` 至該 session 的 temp root；關閉 session 時一併清除。若來源是一般 CSV，只有同資料夾內與 CSV stem 相符的 `.pth/.pt` 才列為 companion model，避免任意猜測。
- 相關成果不再以整列按鈕插在 Reader 上方。有成果的 tab 在關閉 `×` 左側顯示 `◈` icon；點擊後以 backdrop blur + slide-in 動畫開啟右側 drawer，模型清單在 drawer 內獨立捲動，因此大量 `.pth` 不會把 CSV table 推到頁面下方。
- 成果 drawer 綁定被點擊的 session，不要求先切換 tab；開啟時會將該 session 設為 active 並收起檔案庫 drawer。關閉 session 時若成果 drawer 正在顯示該 session，也會同步關閉。
- drawer 標題下方的 CSV display name 使用自動換行（包含沒有空白的長檔名也可斷行），不再以 ellipsis 截斷。
- drawer 先顯示 **Training Results**，再顯示 **Transformer Models**。模型主標籤只保留 checkpoint 識別資訊：`*_epoch1.pth` 顯示 `epoch1`、`*_epoch_12.pt` 顯示 `epoch12`，沒有 epoch 後綴的完整 checkpoint 顯示 `all`；原始檔名仍顯示於次要文字，完整 path 保留在 tooltip。
- Transformer Models 使用 checkpoint 數值排序而非檔名字典序：`all` 固定第一，接著為 `epoch1, epoch2, ... epoch10, ...`；相同 checkpoint 標籤才以完整 path 作穩定排序，且前端排序使用 path array copy，不修改 session 原始資料。
- 點 **加入 Token Visualization** 會把 model path 寫入左側 Model Path 並切換到 Visualization；使用者仍可確認路徑後按 **Load Model**。
- 來源資料夾中若遞迴找到 `*_history.csv`、`*_phase_summary.csv`、`*_turn_metrics.csv` 或 `comparison_table.csv`，可點 **加入 Training Results**；頁面會切換並沿用既有 scan/read/parse 流程加入圖表。
- 一般 prediction CSV 若沒有可驗證的 companion checkpoint 或 training files，不顯示匯入按鈕。

### 檔案 Tab 關閉行為

- 關閉按鈕是 tab 內獨立的 icon button，具備 `aria-label` 與完整檔名 tooltip；不需要先切換到該 tab。
- 關閉非 active tab 時維持目前檔案；關閉 active tab 時優先選取原位置右側的相鄰 tab，若沒有則選取左側；關閉最後一個 tab 後回到空白狀態。
- 前端會立即移除 tab 並更新 `csv_reader_metadata`，再呼叫 `POST /api/csv_reader/close` 清理 backend session／ZIP 暫存資料；backend 清理失敗時會在 status 顯示警告。
- 2026-07-10 端到端驗證：以三個 repo 內真實 CSV 測試 non-active、active、last-tab 關閉流程，active path 與持久化 session 數皆正確，三次 close API 均回傳 HTTP 200。

### 左右分隔條拖曳

- 分隔條使用 Pointer Events 與 `setPointerCapture()`，不再依賴父頁面 `window.mouseup`。向右拖入網頁 iframe 後放開滑鼠，父頁面仍能可靠收到 `pointerup` 並結束拖曳。
- 拖曳期間會顯示全頁透明 resize overlay，避免 iframe 或圖片接管 hit testing；`pointerup`、`pointercancel`、lost pointer capture 與瀏覽器失焦都會清理拖曳狀態。
- 每個 animation frame 使用最新的 `clientX`，直接修改右側 `<aside>` 的 DOM width，不在拖曳期間呼叫 React `setState`，因此不會反覆 render 大型 CSV table 或重建 iframe。
- 放開時才提交一次 `sidePaneWidth` state，並透過既有 metadata effect 寫入一次 `csv_reader_metadata`。
- 2026-07-10 瀏覽器驗證：真實滑鼠跨入 iframe 後放開可立即退出；1,000 個密集 pointermove events 約 53.3 ms，最終 700 px 寬度正確套用，拖曳期間 0 次持久化、放開後 1 次。

### 持久化（輕量 metadata）

- Key：**`csv_reader_metadata`**
- **只存**：路徑、每 tab 的 page/filter/`pinned`、active tab、pageSize、side pane 狀態、`sidePaneWidth`
- **不存**：rows、CSV 內容、解壓 temp
- 進頁時依 metadata **重新 open + page** 還原
- session restore 以 ref 保證只啟動一次，且在全部 open 完成前暫停 metadata 寫入；不會再因第一次 state update 的 effect cleanup 中斷並用空 sessions 覆蓋舊資料。關閉 tab 後正常 metadata effect 會寫入剩餘 sessions，因此刷新不會復活已關閉檔案。

---

## 8. 開發與驗證

```bash
# Backend（在 backend/）
python -m py_compile main.py api/training.py api/csv_reader.py api/binary_samples.py model_engine/transformer_model.py
# 啟動方式依你環境，例如：
# uvicorn main:app --host 0.0.0.0 --port 8080

# Frontend（在 frontend/）
npm install
npx tsc --noEmit
npm run build
npm run dev   # 預設 port 3001，proxy 到 8080
```

環境變數（`vite.config.ts`）：

- `BACKEND_PORT`（預設 `8080`）
- `FRONTEND_PORT`（預設 `3001`）

---

## 9. 修改紀錄（脈絡摘要）

| 主題 | 內容 |
|------|------|
| Training 曲線 legend | 圖上只顯示時間戳；列表仍全名 |
| Visualization 自訂 Feature | 從 `.pth` 推斷並重建 input projection/schema；支援 2D/3D JSON、前後端 shape 驗證，空白時依模型尺寸產生隨機 input |
| Phase Analysis 掃不到 | `training.py` 改遞迴 walk；支援 `*_turn_metrics.csv` |
| 成果列顯示三筆重複 | 合併 base/random/history 為單一成果名 |
| Random Phase 消失 | `randomPhaseData` / `randomTurnMetrics` 獨立欄位 |
| base_vs_random 圖 | 保留 4 張 + 新增第 5 張整合 Plotly（對照 `PhaseAnalyzer/phaseAnalyzer.py` 內 PNG 生成） |
| Tenhou CSV Reader | 移植至 ArchAnalyzer + sidebar 入口 |
| CSV 切 Training 會消失 | `App.tsx` 改 display 切換保留 mount |
| CSV 長期儲存 | metadata only → `csv_reader_metadata` |
| CSV 檔案無個別關閉按鈕 | 每個 tab 加入獨立 `×`；可直接關閉任意 session，active tab 依相鄰順序切換並同步 metadata/backend |
| CSV 右欄寬度 | 可拖曳 resize + 寫入 metadata |
| CSV 分隔條卡頓／無法放開 | 改用 Pointer Events + pointer capture + iframe overlay；RAF 套用最新座標並直接更新 DOM，放開時才提交 state 與 metadata |
| CSV URL1 無法直接切 URL2 | iframe 以完整 `sideUrl` 作 key；每次選新 URL 強制重建預覽 |
| CSV 新開／關閉後記憶錯誤 | 修正 restore effect 首輪 cleanup race；還原完成前禁止空 metadata 覆寫，tab 關閉後持久化剩餘清單 |
| CSV 檔案越開越多 | 新增現代化動畫檔案庫 drawer + 每 session 橫幅 switch，`pinned` 寫入 metadata |
| CSV 關聯成果匯入 | 從 ZIP／companion file 偵測模型，從支援的 training CSV 偵測成果資料夾，可送往 Visualization / Training Results |
| CSV 關聯成果清單佔滿畫面 | 移除 Reader 上方 inline 清單；改為 tab `×` 左側 `◈` + 右側動畫成果 drawer，大量模型在 drawer 內捲動 |
| 成果 drawer 長名稱難辨識 | CSV 名稱完整換行；Training Results 排在最前；模型以 `all`／`epochN` 作醒目主標籤，原始檔名與 path 仍可查閱 |
| 模型 epoch 排序混亂 | 不再使用 `epoch1, epoch10, epoch2` 字典序；固定 `all` 最前並依 epoch 阿拉伯數字升冪排列 |
| CSV output 快速搜尋 | 新增固定根目錄 SQLite index、手動重掃與 query；結果只回填 path input |
| CSV 索引刷新後顯示未讀取 | 新增 persisted status API；Reader mount 時從 SQLite 還原筆數與最後掃描時間 |
| CSV 索引結果遮住 Filter | 結果清單加入獨立 open state，點擊外部、Esc、修改 query 或選取結果時自動關閉 |
| CSV 建立索引時卡住 | 將同步 filesystem／SQLite／CSV 工作改由 FastAPI thread pool 執行；索引掃描加上重複執行 guard，避免阻塞 event loop 與並行重建 |
| Binary Sample Search 位置與 sample feature | 從左側欄移至主內容 Custom Feature 上方；相容 sample 顯示 Tenhou 盤面後，同步把 2D feature JSON 填入 Custom Feature 並 forward |
| Binary sample dense shape 轉換 | 依 checkpoint seq_len 重新 pack action tokens；New_Add compact init sidecar 還原成 dense init，真實 sample 驗證 `[25,986]` storage 可產生 `[49,12308]` model input |
| Current Tenhou Board 拖曳卡頓 | 改用 Pointer Events + pointer capture + iframe overlay；RAF 直接更新右欄 DOM width，放開時才提交 React state |
| Training 圖表全螢幕 | 13 張 Plotly 圖統一使用 FullscreenPlot；按鈕／Esc 可退出，fullscreenchange 觸發 responsive resize |
| Training 顏色拖曳卡頓 | 顏色 input 改為 memoized 本地互動 + 300 ms trailing commit；避免拖曳中重繪全部 Plotly 圖與序列化完整 `training_results` |
| Attention 權重 | MahjongTransformer MHA eval 重算 weights；backend forward 後收集 |
| 全域頁面導航 | 左欄只保留三頁導航；Visualization 的模型操作、Token Index、Layer 全數移入主內容控制卡 |
| 左側欄顯示模式 | 預設固定；可切成自動隱藏，滑鼠靠近左側 hover rail 或鍵盤 focus 時滑出 |
| Tenhou 盤面側欄 | 改為仿 CSV 檔案庫的 backdrop + slide-in overlay drawer，保留關閉、外開與拖曳調寬 |
| Input Features 可讀性 | feature hover/focus 放大，顯示 index 與高精度實值 |
| Input／Layer Input 格線 | 兩者固定每列 34 格並自適應等寬；左右邊界格向內展開，避免詳細值被裁切；詳細 tooltip 放大為易讀字級 |
| Embedding 版面與數值 | 改為獨占完整列；dimension hover/focus 加寬並顯示 backend 實際浮點值 |
| Analysis Chat 可信度 | 改為 backend decode → LLM blind board assessment → deterministic model/oracle comparison；修正 sigmoid/softmax、valid_len、target seat、prompt isolation、輸入限制與 HTTP 錯誤分類 |
| Tenhou Board iframe 切換 | 加入 `key={boardUrl}` 使 hash-only src 變動強制重建 iframe |
| Load Sample 與 forward 不同步 | WS send 改 `wsRef`；run_forward 前 clear_cache；載入樣本清空 viz state；forward 錯誤顯示於 UI |
| 網頁反覆重連／像是重啟 | WebSocket lifecycle 改以連線 instance ownership 判斷；StrictMode 舊 socket 關閉後不再建立幽靈 reconnect |
| Token / Layer 向量語意與版面 | Raw Input 只顯示所選 token 並可收合；新增 checkpoint `d_model` 對齊的 Layer Input Vector；修正六層 slider 為 0–5；三個向量 panel 全寬且右側按鈕避開 scrollbar |

---

## 10. 使用者偏好（協作時注意）

- 回覆 **簡潔、直接、少廢話**，先給結果再補必要細節
- 需要除錯時可提供 **console / 後端 log** 線索
- 側欄換頁希望 **保留頁面狀態**（已用 display 隱藏實作）

---

## 11. 新 Session 建議起手式

1. 讀本檔 + 必要時 `read_file` 上述關鍵檔
2. 確認 backend 是否在跑、`/health`、WS 是否連上
3. 改 frontend 後跑 `npx tsc --noEmit`；改 backend 後 `py_compile` 並重啟
4. Training / Phase 資料路徑範例：  
   `.../Transformer/output/`、`.../Transformer/output/analysis/`

---

*最後更新：2026-07-16（修正 StrictMode WebSocket 幽靈重連，以及 CSV／索引同步 I/O 阻塞 event loop）。*
