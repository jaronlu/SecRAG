#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
LOG_FILE="${LOG_FILE:-/tmp/secrag-${PORT}.log}"
KILL_EXISTING="${KILL_EXISTING:-1}"
BUILD_FRONTEND="${BUILD_FRONTEND:-auto}"  # auto|always|never

# P1-6: 可选构建 React 前端
if [ "${BUILD_FRONTEND}" = "always" ] || { [ "${BUILD_FRONTEND}" = "auto" ] && [ ! -d "frontend/dist" ] && command -v npm >/dev/null 2>&1 && [ -d "frontend" ]; }; then
  echo "Building React frontend..."
  (cd frontend && npm install --silent && npm run build)
  echo "Frontend built successfully."
fi

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

# 模型与网络环境固化（2026-11-04 网络故障定案）：
# - HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE：reranker 与 embedding 权重已本地化，
#   不设 OFFLINE 时 AutoTokenizer/from_pretrained 仍会向 huggingface.co 发在线
#   校验并挂死整条 QA 链路（表现为 503 "LLM provider unavailable"）。首次
#   下载模型需要联网时以 HF_HUB_OFFLINE=0 覆盖。
# - NO_PROXY='*'：macOS 系统代理会被 UURemote 等应用动态开关（指向 127.0.0.1
#   僵尸端口），httpx trust_env 在客户端创建时读入并冻结，导致 LLM 调用进
#   死代理；强制直连使服务端对系统代理免疫。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"

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
echo "UI:      http://${HOST}:${PORT}/"
echo "Admin:   http://${HOST}:${PORT}/admin"
echo "Docs:    http://${HOST}:${PORT}/docs"
echo "Health:  http://${HOST}:${PORT}/health"
echo "Metrics: http://${HOST}:${PORT}/metrics"
echo

exec uv run uvicorn src.api.main:app --host "${HOST}" --port "${PORT}" 2>&1 | tee "${LOG_FILE}"
