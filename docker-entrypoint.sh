#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# spoon-bot Docker Entrypoint
# Supports multiple run modes: gateway, agent, cli
# =============================================================================

MODE="${SPOON_BOT_MODE:-gateway}"

echo "============================================"
echo "  spoon-bot v0.1.0"
echo "  Mode: ${MODE}"
echo "============================================"

# Ensure workspace is initialized
WORKSPACE="${SPOON_BOT_WORKSPACE_PATH:-/data/workspace}"
mkdir -p "${WORKSPACE}/memory" "${WORKSPACE}/skills" 2>/dev/null || true

case "${MODE}" in
  gateway)
    echo "Starting Gateway (REST + WebSocket)..."
    echo "  Host: ${GATEWAY_HOST:-0.0.0.0}"
    echo "  Port: ${GATEWAY_PORT:-8080}"
    echo "  Workers: ${GATEWAY_WORKERS:-1}"
    echo "  Debug: ${GATEWAY_DEBUG:-false}"
    echo ""
    echo "Endpoints:"
    echo "  REST API:    http://${GATEWAY_HOST:-0.0.0.0}:${GATEWAY_PORT:-8080}/v1/"
    echo "  WebSocket:   ws://${GATEWAY_HOST:-0.0.0.0}:${GATEWAY_PORT:-8080}/v1/ws"
    echo "  Health:      http://${GATEWAY_HOST:-0.0.0.0}:${GATEWAY_PORT:-8080}/health"
    echo "  API Docs:    http://${GATEWAY_HOST:-0.0.0.0}:${GATEWAY_PORT:-8080}/docs"
    echo "============================================"

    # Build uvicorn command
    # Uses server.py which auto-initializes agent from env vars
    UVICORN_ARGS=(
      "spoon_bot.gateway.server:create_app"
      "--factory"
      "--host" "${GATEWAY_HOST:-0.0.0.0}"
      "--port" "${GATEWAY_PORT:-8080}"
      "--workers" "${GATEWAY_WORKERS:-1}"
      "--log-level" "$(echo "${SPOON_BOT_LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"
      "--access-log"
    )

    # Add reload flag in debug mode
    if [ "${GATEWAY_DEBUG:-false}" = "true" ]; then
      UVICORN_ARGS+=("--reload")
    fi

    exec uvicorn "${UVICORN_ARGS[@]}" "$@"
    ;;

  agent)
    echo "Starting Agent mode..."
    echo "  Workspace: ${WORKSPACE}"
    echo "============================================"

    # Pass through any extra arguments
    if [ $# -gt 0 ]; then
      exec spoon-bot agent "$@"
    else
      exec spoon-bot agent -m "${SPOON_BOT_AGENT_MESSAGE:-Hello, I am spoon-bot running in Docker!}"
    fi
    ;;

  cli)
    echo "Starting CLI mode..."
    echo "============================================"
    exec spoon-bot "$@"
    ;;

  onboard)
    echo "Running onboard setup..."
    echo "============================================"
    exec spoon-bot onboard
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    echo ""
    echo "Available modes:"
    echo "  gateway  - Start REST + WebSocket API server (default)"
    echo "  agent    - Run agent in one-shot mode"
    echo "  cli      - Run any spoon-bot CLI command"
    echo "  onboard  - Initialize workspace"
    echo ""
    echo "Set via SPOON_BOT_MODE environment variable."
    exit 1
    ;;
esac
