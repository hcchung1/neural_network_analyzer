#!/usr/bin/env bash
set -e

echo "🚀 Starting ArchAnalyzer..."

# 設定端口（可依需求修改）
BACKEND_PORT=${BACKEND_PORT:-8080}
FRONTEND_PORT=${FRONTEND_PORT:-3001}

# Export for Vite config to read
export BACKEND_PORT
export FRONTEND_PORT

echo "📡 後端端口: $BACKEND_PORT"
echo "🎨 前端端口: $FRONTEND_PORT"

# 啟動後端（背景執行）
(
  cd backend || exit 1
  echo "📦 Installing backend dependencies..."
  pip install -r requirements.txt -q 2>/dev/null || echo "Backend deps already installed"
  echo "🔥 Starting FastAPI backend on http://0.0.0.0:$BACKEND_PORT"
  # 關鍵：使用 --host 0.0.0.0 讓外部可以連線
  uvicorn main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload &
)

# 等待後端啟動
echo "⏳ Waiting for backend to start..."
sleep 3

# 測試後端是否可用
echo "🔍 Checking backend health..."
if curl -s "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
  echo "✅ Backend is healthy"
else
  echo "⚠️  Backend health check failed, but continuing..."
fi

# 啟動前端
(
  cd frontend || exit 1
  echo "📦 Installing frontend dependencies..."
  npm install 2>/dev/null || echo "Frontend deps already installed"
  echo "🎨 Starting React frontend on http://0.0.0.0:$FRONTEND_PORT"
  # 關鍵：使用 --host 0.0.0.0 讓外部可以連線
  npm run dev -- --port "$FRONTEND_PORT" --host 0.0.0.0
)
