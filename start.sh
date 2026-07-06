#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
LOG_FILE="${LOG_FILE:-/tmp/secrag-${PORT}.log}"
KILL_EXISTING="${KILL_EXISTING:-1}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

if [ ! -f ".env" ]; then
  echo ".env not found. Create it first:"
  echo "  cp .env.example .env"
  echo "  then fill OPENAI_API_KEY or configure another provider."
  exit 1
fi

if lsof -iTCP:"${PORT}" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use."
  lsof -iTCP:"${PORT}" -sTCP:LISTEN -n -P
  if [ "${KILL_EXISTING}" != "1" ]; then
    echo "KILL_EXISTING=0, not stopping existing process."
    echo "Start on another port instead:"
    echo "  PORT=8002 ./start.sh"
    exit 1
  fi

  echo "Stopping existing process on port ${PORT}..."
  lsof -tiTCP:"${PORT}" -sTCP:LISTEN -n -P | xargs kill
  sleep 1

  if lsof -iTCP:"${PORT}" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    echo "Port ${PORT} is still in use after SIGTERM. Sending SIGKILL..."
    lsof -tiTCP:"${PORT}" -sTCP:LISTEN -n -P | xargs kill -9
    sleep 1
  fi
fi

echo "Starting SecRAG on http://${HOST}:${PORT}"
echo "Logs: ${LOG_FILE}"
echo
echo "UI:   http://${HOST}:${PORT}/"
echo "Docs: http://${HOST}:${PORT}/docs"
echo

exec uv run uvicorn src.api.main:app --host "${HOST}" --port "${PORT}" 2>&1 | tee "${LOG_FILE}"
