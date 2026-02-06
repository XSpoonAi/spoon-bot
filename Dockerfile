# =============================================================================
# spoon-bot Dockerfile
# Multi-stage build with uv for fast, reproducible installs
# Supports: Gateway (REST + WebSocket), Agent, CLI modes
# =============================================================================

# --------------- Stage 1: Build ---------------
FROM python:3.12-slim AS builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies (all extras for full functionality)
# Uses uv sync with frozen lockfile for reproducibility
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-extras

# Copy source code
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras


# --------------- Stage 2: Runtime ---------------
FROM python:3.12-slim AS runtime

LABEL maintainer="XSpoon Team <team@xspoon.ai>"
LABEL description="spoon-bot: Local-first AI agent with native OS tools"
LABEL version="0.1.0"

# Install runtime system dependencies
# - git: for workspace git operations
# - curl: for healthchecks
# - tini: proper PID 1 init for signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r spoonbot && useradd -r -g spoonbot -m -d /home/spoonbot spoonbot

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY --from=builder /app/spoon_bot /app/spoon_bot
COPY --from=builder /app/pyproject.toml /app/

# Copy entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
# Ensure Python doesn't buffer output (important for Docker logs)
ENV PYTHONUNBUFFERED=1
# Don't write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# ======= Default Environment Configuration =======

# --- Run Mode ---
# Options: gateway (default), agent, cli
ENV SPOON_BOT_MODE=gateway

# --- Gateway Settings ---
ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8080
ENV GATEWAY_DEBUG=false
ENV GATEWAY_WORKERS=1

# --- JWT / Auth ---
# IMPORTANT: Set a strong secret in production!
# ENV JWT_SECRET=your-secret-here
ENV JWT_ACCESS_EXPIRE_MINUTES=15

# --- LLM Provider API Keys ---
# Set at least ONE of these:
# ENV ANTHROPIC_API_KEY=
# ENV OPENAI_API_KEY=
# ENV DEEPSEEK_API_KEY=
# ENV GEMINI_API_KEY=
# ENV OPENROUTER_API_KEY=

# --- LLM Provider Settings ---
# ENV SPOON_BOT_DEFAULT_PROVIDER=anthropic
# ENV SPOON_BOT_DEFAULT_MODEL=claude-sonnet-4-20250514
# ENV BASE_URL=

# --- Web3 Configuration (optional) ---
# ENV PRIVATE_KEY=
# ENV RPC_URL=https://mainnet-1.rpc.banelabs.org
# ENV SCAN_URL=https://xt4scan.ngd.network/
# ENV CHAIN_ID=47763

# --- Agent Settings ---
ENV SPOON_BOT_MAX_ITERATIONS=20
ENV SPOON_BOT_SHELL_TIMEOUT=60
ENV SPOON_BOT_MAX_OUTPUT=10000
ENV SPOON_BOT_LOG_LEVEL=INFO

# --- Workspace ---
ENV SPOON_BOT_WORKSPACE_PATH=/data/workspace

# Create workspace directory with correct ownership
RUN mkdir -p /data/workspace/memory /data/workspace/skills \
    && chown -R spoonbot:spoonbot /data /app

# Volume for persistent workspace data
VOLUME ["/data"]

# Expose gateway port
EXPOSE 8080

# Switch to non-root user
USER spoonbot

# Health check for gateway mode
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${GATEWAY_PORT}/health || exit 1

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["tini", "--", "/app/docker-entrypoint.sh"]
