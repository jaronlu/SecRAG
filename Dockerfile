# ══════════════════════════════════════════════════════════════════════
# SecRAG Dockerfile — 多阶段构建，精简生产镜像
# ══════════════════════════════════════════════════════════════════════

# Stage 1: Builder — 安装 uv 和 Python 依赖
FROM python:3.11-slim AS builder

ENV UV_HOME=/opt/uv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PIP_NO_CACHE_DIR=1

# 安装系统依赖（编译部分 Python 包需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先复制依赖文件，利用 Docker 层缓存
COPY pyproject.toml uv.lock ./

# 安装运行时依赖（不含 dev 依赖）
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Runtime — 精简运行镜像
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HOME=/opt/uv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/data/hf_cache \
    TRANSFORMERS_CACHE=/data/hf_cache

# 运行时只需最小系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 复制虚拟环境
COPY --from=builder /app/.venv /app/.venv

# 复制项目源码
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY start.sh /app/start.sh
COPY .env.example /app/.env.example

# 复制数据目录（包含已入库的 ChromaDB 数据和示例数据）
# 注意：如果 data/ 很大，建议用 volume 挂载而非打包进镜像
COPY data/ /app/data/

# 创建数据目录（用于 volume 挂载）
RUN mkdir -p /data/chroma /data/hf_cache && \
    chmod +x /app/start.sh

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT:-8000}/health || exit 1

EXPOSE 8000

# 使用 uvicorn 直接启动（不依赖 uv，因为 .venv 已在 PATH 中）
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
