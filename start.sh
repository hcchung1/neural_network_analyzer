#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BACKEND_HOST=${BACKEND_HOST:-0.0.0.0}
BACKEND_PORT=${BACKEND_PORT:-8080}
FRONTEND_HOST=${FRONTEND_HOST:-0.0.0.0}
FRONTEND_PORT=${FRONTEND_PORT:-3001}
SIDECAR_PID=""
NEXT_PID=""

export BACKEND_PORT
export FRONTEND_PORT
export PYTHON_SIDECAR_URL="http://127.0.0.1:$BACKEND_PORT/internal/v1"

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  if [[ -n "$SIDECAR_PID" ]] && kill -0 "$SIDECAR_PID" 2>/dev/null; then
    echo "🧹 Stopping Python Sidecar (PID $SIDECAR_PID)..."
    kill "$SIDECAR_PID" 2>/dev/null || true
  fi

  if [[ -n "$NEXT_PID" ]] && kill -0 "$NEXT_PID" 2>/dev/null; then
    echo "🧹 Stopping Next.js (PID $NEXT_PID)..."
    kill "$NEXT_PID" 2>/dev/null || true
  fi

  [[ -n "$SIDECAR_PID" ]] && wait "$SIDECAR_PID" 2>/dev/null || true
  [[ -n "$NEXT_PID" ]] && wait "$NEXT_PID" 2>/dev/null || true

  exit "$status"
}

validate_port() {
  local name=$1
  local port=$2

  if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
    echo "❌ $name must be an integer between 1 and 65535; received: $port" >&2
    exit 2
  fi
}

ensure_port_available() {
  local name=$1
  local host=$2
  local port=$3

  if ! command -v lsof >/dev/null 2>&1; then
    echo "⚠️ lsof is unavailable; skipping the $name port preflight check."
    return
  fi

  if lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | grep -q .; then
    echo "❌ $name cannot start: $host:$port is already in use." >&2
    echo "   Stop the existing process or select another port before retrying." >&2
    exit 1
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

validate_port "BACKEND_PORT" "$BACKEND_PORT"
validate_port "FRONTEND_PORT" "$FRONTEND_PORT"
ensure_port_available "Python Sidecar" "$BACKEND_HOST" "$BACKEND_PORT"
ensure_port_available "Next.js" "$FRONTEND_HOST" "$FRONTEND_PORT"

echo "🚀 Starting ArchAnalyzer Next.js + Python Sidecar Stack..."
echo "📡 Sidecar: http://$BACKEND_HOST:$BACKEND_PORT"
echo "🎨 Next.js: http://$FRONTEND_HOST:$FRONTEND_PORT"

# echo "📦 Verifying Python Sidecar..."
# (
#   cd "$PROJECT_DIR/backend"
#   python3 verify.py
# )

echo "🔥 Starting Python Sidecar..."
(
  cd "$PROJECT_DIR/backend"
  exec uvicorn main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
) &
SIDECAR_PID=$!

echo "⏳ Waiting for Sidecar health check..."
sidecar_healthy=false
for _ in {1..60}; do
  if ! kill -0 "$SIDECAR_PID" 2>/dev/null; then
    wait "$SIDECAR_PID" || true
    echo "❌ Python Sidecar exited before becoming healthy." >&2
    exit 1
  fi

  if curl --fail --silent "http://127.0.0.1:$BACKEND_PORT/internal/v1/health" >/dev/null 2>&1; then
    sidecar_healthy=true
    break
  fi
  sleep 0.5
done

if [[ "$sidecar_healthy" != true ]]; then
  echo "❌ Python Sidecar health check timed out after 30 seconds." >&2
  exit 1
fi

echo "✅ Sidecar is healthy"
echo "🎨 Starting Next.js App Router Server..."
cd "$PROJECT_DIR"

if [ -x "./node_modules/.bin/next" ]; then
  ./node_modules/.bin/next dev --hostname "$FRONTEND_HOST" --port "$FRONTEND_PORT" &
else
  npm run dev -- --hostname "$FRONTEND_HOST" --port "$FRONTEND_PORT" &
fi
NEXT_PID=$!

wait "$NEXT_PID"
