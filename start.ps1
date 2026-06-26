#!/usr/bin/env pwsh
# ArchAnalyzer Startup Script for Windows

$BACKEND_PORT = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { "8080" }
$FRONTEND_PORT = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { "3001" }

Write-Host "🚀 Starting ArchAnalyzer..." -ForegroundColor Cyan
Write-Host "📡 Backend port: $BACKEND_PORT" -ForegroundColor Gray
Write-Host "🎨 Frontend port: $FRONTEND_PORT" -ForegroundColor Gray

# Start backend in background
$backendJob = Start-Job {
    Set-Location backend
    Write-Host "📦 Installing backend dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt -q 2>$null
    Write-Host "🔥 Starting FastAPI backend on http://0.0.0.0:$using:BACKEND_PORT" -ForegroundColor Green
    uvicorn main:app --host 0.0.0.0 --port $using:BACKEND_PORT --reload
}

# Wait for backend to start
Start-Sleep -Seconds 2

# Start frontend
Set-Location frontend
Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Yellow
npm install 2>$null
Write-Host "🎨 Starting React frontend on http://0.0.0.0:$FRONTEND_PORT" -ForegroundColor Green
npm run dev -- --port $FRONTEND_PORT --host 0.0.0.0

# Cleanup
Stop-Job $backendJob
Remove-Job $backendJob
