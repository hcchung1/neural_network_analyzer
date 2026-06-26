#!/usr/bin/env pwsh
# ArchAnalyzer Startup Script for Windows

Write-Host "🚀 Starting ArchAnalyzer..." -ForegroundColor Cyan

# Start backend in background
$backendJob = Start-Job {
    Set-Location backend
    Write-Host "
  Installing backend dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt -q 2>$null
    Write-Host "
  Starting FastAPI backend on http://localhost:8000" -ForegroundColor Green
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
}

# Start frontend
Set-Location frontend
Write-Host "
  Installing frontend dependencies..." -ForegroundColor Yellow
npm install
Write-Host "
  Starting React frontend on http://localhost:3000" -ForegroundColor Green
npm run dev

# Cleanup
Stop-Job $backendJob
Remove-Job $backendJob
