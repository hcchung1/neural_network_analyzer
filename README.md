# ArchAnalyzer

Transformer 單一 Token 視覺化工具 - FastAPI + React + WebSocket

## 啟動方式

### 本地開發

```bash
# 安裝後端相依套件
cd backend
pip install -r requirements.txt

# 啟動後端服務
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 在另一個終端機啟動前端
cd frontend
npm install
npm run dev
```

### 透過 SSH Port Forwarding 遠端部屬

```bash
# 在遠端伺服器上啟動服務
ssh user@remote-server
uvicorn main:app --host 0.0.0.0 --port 8000

# 在本機終端機執行 port forwarding
ssh -L 8000:localhost:8000 -L 3000:localhost:3000 user@remote-server
```

之後在本機瀏覽器開啟 `http://localhost:3000` 即可。

## 功能特色

- 🔬 **單一 Token 視覺化**: 深入檢視 Transformer 中單一 token 的資訊流
- 📊 **多維度分析**: Input Features、Embedding、Attention、Output
- 🔄 **即時互動**: WebSocket 雙向通訊，即時更新視圖
- 🎨 **層級探索**: 透過 Slider 切換不同 Transformer 層級
- 📈 **機率分布**: 視覺化最終輸出的機率分布

## 技術架構

- **Backend**: FastAPI + PyTorch + Uvicorn
- **Frontend**: React + TypeScript + Vite
- **通訊**: WebSocket + REST API
- **部署**: SSH Port Forwarding
