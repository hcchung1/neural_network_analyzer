#!/usr/bin/env bash
set -e

echo "🚀 Starting ArchAnalyzer..."

# Start backend
(
n  cd backend || exit 1
  echo "📦 Installing backend dependencies..."
  pip install -r requirements.txt -q 2>/dev/null || echo "Backend deps already installed"
  echo "🔥 Starting FastAPI backend on http://localhost:8000"
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
)

# Start frontend
cd frontend || exit 1
echo "📦 Installing frontend dependencies..."
npm install || echo "Frontend deps already installed"
echo "🎨 Starting React frontend on http://localhost:3000"
npm run dev
