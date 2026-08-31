#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TEST_DIR=$(mktemp -d)
FAKE_BIN="$TEST_DIR/bin"
CALL_LOG="$TEST_DIR/calls.log"
SIDECAR_PID_FILE="$TEST_DIR/sidecar.pid"

cleanup() {
  if [[ -f "$SIDECAR_PID_FILE" ]]; then
    local sidecar_pid
    sidecar_pid=$(cat "$SIDECAR_PID_FILE")
    if kill -0 "$sidecar_pid" 2>/dev/null; then
      kill "$sidecar_pid" 2>/dev/null || true
    fi
  fi
  rm -rf "$TEST_DIR"
}
trap cleanup EXIT

mkdir -p "$FAKE_BIN"

cat > "$FAKE_BIN/uvicorn" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'uvicorn %s\n' "$*" >> "$TEST_CALL_LOG"
printf '%s\n' "$$" > "$TEST_SIDECAR_PID_FILE"
trap 'exit 0' TERM INT
while true; do
  sleep 1
done
EOF

cat > "$FAKE_BIN/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'curl %s\n' "$*" >> "$TEST_CALL_LOG"
exit 0
EOF

cat > "$FAKE_BIN/npm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'npm %s\n' "$*" >> "$TEST_CALL_LOG"
exit 17
EOF

chmod +x "$FAKE_BIN/uvicorn" "$FAKE_BIN/curl" "$FAKE_BIN/npm"

set +e
PATH="$FAKE_BIN:$PATH" \
  TEST_CALL_LOG="$CALL_LOG" \
  TEST_SIDECAR_PID_FILE="$SIDECAR_PID_FILE" \
  BACKEND_PORT=18080 \
  FRONTEND_PORT=13001 \
  "$PROJECT_DIR/start.sh" > "$TEST_DIR/output.log" 2>&1
status=$?
set -e

if [[ $status -ne 17 ]]; then
  echo "Expected frontend exit status 17, got $status"
  cat "$TEST_DIR/output.log"
  exit 1
fi

if ! grep -Fq "uvicorn main:app --host 0.0.0.0 --port 18080" "$CALL_LOG"; then
  echo "Sidecar did not bind to the configured public host and port"
  cat "$CALL_LOG"
  exit 1
fi

if ! grep -Fq "npm run dev -- --hostname 0.0.0.0 --port 13001" "$CALL_LOG"; then
  echo "Next.js did not bind to the configured public host and port"
  cat "$CALL_LOG"
  exit 1
fi

sidecar_pid=$(cat "$SIDECAR_PID_FILE")
if kill -0 "$sidecar_pid" 2>/dev/null; then
  echo "Sidecar process $sidecar_pid survived frontend shutdown"
  exit 1
fi

echo "startup lifecycle test passed"
