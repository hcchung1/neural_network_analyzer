# Transformer 單一 Token 視覺化開發方案

## 1. 目標

針對 Transformer 架構中輸入的單一 token，視覺化其 input features、經過 Embedding 後的結果，以及最終 output，以理解資訊在模型中的流動過程。

## 2. 平台評估與選擇

### 2.1 評估標準

*   **效能需求：** Transformer 模型涉及大量矩陣運算，視覺化需即時渲染高維度數據。
*   **互動性：** 使用者需能互動式地查看不同 token 與層級的資訊 (例如 hover、click)。
*   **遠端存取：** 用戶需要透過 SSH port forwarding 在遠端電腦上執行，並在本機瀏覽器查看視覺化結果。
*   **GPU 運算：** 模型推論與中間特徵提取需要 GPU 加速。
*   **開發效率：** 需能直接使用現有的 `transformer.py` 模型定義，避免重寫或轉換模型。

### 2.2 評估結果

*   **方案 A：獨立 Web 應用 (FastAPI + React)**
    *   **優點：** 跨平台、易於維護、React 生態系強大。前後端分離架構適合遠端佈署，可透過 SSH port forwarding 存取，對於需要 GPU 運算的場景尤為重要。可以透過 SSH port forwarding 在遠端伺服器上運行，本機瀏覽器查看結果。
    *   **缺點：** 需將 PyTorch 模型封裝為 API，增加架構複雜度；大型 Postal matrices 資料傳輸可能產生頻寬壓力。

*   **方案 B：獨立 Windows/Linux 桌面應用程式 (Python + PyQt)**
    *   **優點：** 直接整合現有 Python 模型，無需額外 API 層；本地運算效能高；可離線運行。
    *   **缺點：** 難以透過 SSH 遠端使用；需在本機安裝；GPU 資源受限於本機。不適合遠端 GPU 運算場景。

### 2.3 最終選擇

**推薦採用方案 A：Web 應用程式 (FastAPI + React + WebSocket)**。

理由：由於用戶需要在遠端電腦上運行並透過 SSH port forwarding 存取，Web 應用是最自然的選擇。FastAPI 作為後端可以直接載入 PyTorch 模型並使用 GPU 運算，React 前端提供豐富的互動式視覺化。前後端透過 WebSocket 進行即時通訊，有效降低大型矩陣資料傳輸的延遲。

## 3. 實行的修改方案 (針對 Web App)

### 3.1 技術架構

*   **後端 (Backend):** FastAPI + PyTorch + Uvicorn。使用 FastAPI 建立 RESTful API 與 WebSocket 服務，直接載入並執行 PyTorch 模型，支援 GPU 運算。
*   **前端 (Frontend):** React + TypeScript + D3.js/Plotly。使用 React 建構互動式 UI，透過 WebSocket 與後端進行即時通訊，使用 D3.js 或 Plotly 進行視覺化繪圖。
*   **通訊協議:** WebSocket 用於即時傳輸中間特徵與視覺化數據；REST API 用於模型初始化、設定與檔案上傳。
*   **部屬方式:** 在遠端電腦上啟動 Uvicorn 服務，透過 SSH port forwarding (e.g., `ssh -L 8000:localhost:8000 user@remote`) 在本機瀏覽器存取。

### 3.2 目錄結構建議

```
ArchAnalyzer/
├── backend/                     # FastAPI 後端
│   ├── main.py                 # FastAPI 入口與 WebSocket handler
│   ├── api/
│   │   ├── __init__.py
│   │   ├── model.py            # 模型載入與推論 API
│   │   └── visualize.py       # 視覺化數據 API
│   ├── model_engine/
│   │   ├── __init__.py
│   │   └── transformer_model.py  # 從 app.py 提取的模型定義與加載邏輯
│   ├── hooks/
│   │   ├── __init__.py
│   │   └── registry.py         # PyTorch forward hooks 註冊與管理
│   └── requirements.txt        # Python 依賴 (fastapi, uvicorn, torch, etc.)
│
├── frontend/                    # React 前端
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── App.tsx             # React 主應用
│   │   ├── components/
│   │   │   ├── TokenSelector.tsx  # Token  restoration 
│   │   │   ├── LayerSlider.tsx   # 層級滑桿
│   │   │   ├── InputFeatures.tsx  # 輸入特徵視覺化
│   │   │   ├── EmbeddingView.tsx  # Embedding 視覺化
│   │   │   ├── AttentionHeatmap.tsx # Attention 熱力圖
│   │   │   └── OutputView.tsx    # 輸出視覺化
│   │   └── utils/
│   │       └── websocket.ts   # WebSocket 連線管理
│   └── public/
│       └── index.html
└── requirements.txt            # 全域依賴
```

### 3.3 核心功能模塊

1.  **模型封裝與 Hooks (Backend):**
    *   將現有 `transformer.py` 中的 `MahjongTransformer` 模型定義提取至 `backend/model_engine/transformer_model.py`。
    *   **新增 Hook Mechanism:** 在模型的關鍵層 (e.g., Embedding Projection, MultiHeadAttention, EncoderBlock, TenpaiClassifier) 註冊 PyTorch hooks (`register_forward_hook`)，以捕獲並儲存特定 token 的中間特徵。
    *   提供 WebSocket API 供前端調用，例如 `get_token_features(batch_idx, token_idx, layer_idx)`, `get_token_embedding(batch_idx, token_idx)`。由於批次大小為 1，batch_idx 固定為 0。

2.  **視覺化模組 (Frontend):**
    *   **Input Features：** 顯示原始輸入 token 的特徵向量（例如 `F_INIT` 或 `F_ACT` 的稀疏/密集表示），以及經過 Heterogeneous Projection 前的原始分佈。
    *   **Embedding 視覺化：** 繪製 Embedding 向量在高維空間的投影 (e.g., PCA or t-SNE)，並突出顯示該單 token 的向量。若使用 Heterogeneous Tokens，則需分別顯示 `proj_init` 與 `proj_act` 的結果。
    *   **Layer-by-layer View：** 逐步顯示該 token 在 Transformer 各層 (Attention, FFN, Residual Block) 中的狀態變化。重點包括：
        *   **Token-level Attention:** 對於該 token，繪製其在各層各 head 的 Attention Score 分佈 (Heatmap)。
        *   **Hidden State Evolution:** 繪製該 token 的 hidden state 在各層的數值分佈。
    *   **Output 視覺化：** 將最終的 logit 值轉換為機率分布圖 (Bar chart)，並可選顯示與目標值的對比。

3.  **用戶介面 (UI - React):**
    *   **側邊欄:** 上傳輸入數據 (JSON/CSV)、選擇批次、特定 token index，以及模型層級 (layer slider)。
    *   **主視圖:**
        *   分區顯示 Input Features, Embedding, Intermediate Layers, Final Output。\n        *   使用 D3.js 或 Plotly 進行高效互動繪圖 (可縮放、平移的 Heatmap、Bar chart 等)。\n        *   提供 Play/Pause 按鈕模擬 token 的傳播過程。
    *   **互動性：** 點擊特定 token 時，即時透過 WebSocket 請求對應數據並更新視圖。Hover 時顯示詳細數值 (Tooltip)。

### 3.4 技術細節

*   **數據流：**
    *   前端透過 WebSocket 發送 token index, layer index 請求給後端。
    *   後端 `model_engine` 透過 hooks 捕獲指定 token 的**中間表示**。
    *   後端將數據序列化為 JSON 並透過 WebSocket 回傳給前端。
    *   前端更新圖表。
*   **性能優化：**
    *   對於大型模型，中間結果 (如 attention matrices) 可能非常龐大。採用「按需加載」策略，僅在用戶請求特定層或 token 時才計算與傳輸該部分數據。
    *   WebSocket 傳輸前對大型矩陣進行壓縮 (e.g., base64 encoding + gzip) 或降採樣。

## 4. 開發路線圖

1.  **第一階段：後端核心建置**
    *   在 `ArchAnalyzer/backend` 建立 FastAPI 專案骨架。
    *   將模型邏輯從 `app.py` 提取至 `backend/model_engine`。
    *   在模型各層 (Embedding Projection, Attention, FFN, Output) 註冊 `register_forward_hook`，確保能捕獲特定 token 的數值。
    *   建立 WebSocket endpoint，實現前後端即時通訊。

2.  **第二階段：前端視覺化開發**
    *   在 `ArchAnalyzer/frontend` 建立 React + TypeScript 專案 (建議使用 Vite)。
    *   建立基礎 UI 佈局 (側邊欄 + 主視圖)。
    *   開發 Embedding 視覺模組 (使用 PCA/t-SNE 降維展示，標註特定 token)。
    *   開發 Attention 視覺模組 (Heatmap for specific token vs all other tokens)。
    *   開發 Output 視覺模組 (Bar chart for logits)。
    *   實作 WebSocket 連線管理與數據處理。

3.  **第三階段：整合與部屬**
    *   將前後端整合，確保 WebSocket 通訊順暢。
    *   優化響應速度與記憶體使用 (e.g., 清除未使用的 cache)。
    *   加入使用者互動功能（層級滑桿、Token 選擇器、Play/Pause 動畫）。
    *   撰寫部署文件，�明如何透過 SSH port forwarding 啟動服務。

## 5. 風險評估

*   **效能瓶頸：** 大型 Transformer 模型的中間特徵可能佔用大量記憶體。需實作緩存機制或計算懶加載，避免一次加載所有層的數據。
*   **複雜度：** 視覺化深層神經網路的資訊流本質上困難，需與使用者確認可視化的具體特徵 (例如可視化 Q, K, V, attention score, context vector 等) 與維度處理方式。
*   **WebSocket 穩定性：** 大量數據傳輸可能導致 WebSocket 連線不穩定，需實作重連機制與數據分片傳輸。
*   **跨域問題：** 前後端分離開發與部署時，可能遇到 CORS 問題。後端需正確設定 CORS allowance。

---

**總結：** 建議採用 **Web 應用程式 (FastAPI + React + WebSocket)** 架構，利用 FastAPI 作為後端承載 PyTorch 模型與 GPU 運算，React 前端提供互動式視覺化。透過 SSH port forwarding 可以在遠端伺服器運行，本機瀏覽器存取，完美契合用戶的遠端開發需求。  