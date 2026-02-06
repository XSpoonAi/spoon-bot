#!/usr/bin/env bash
# =============================================================================
# spoon-bot API Integration Test Runner
#
# Usage:
#   ./run-tests.sh                    # Build, start, test, stop
#   ./run-tests.sh --skip-build       # Skip Docker build
#   ./run-tests.sh --keep-running     # Don't stop container after tests
#
# Environment variables:
#   SPOON_BOT_API_KEY  - API key for authentication (optional)
#   GATEWAY_PORT       - Port to expose (default: 18080)
#   AUTH_REQUIRED      - Set to "false" to skip auth tests
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER_NAME="spoon-bot-test"
PORT="${GATEWAY_PORT:-18080}"
SKIP_BUILD=false
KEEP_RUNNING=false

# Parse args
for arg in "$@"; do
  case $arg in
    --skip-build) SKIP_BUILD=true ;;
    --keep-running) KEEP_RUNNING=true ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

echo "╔══════════════════════════════════════════╗"
echo "║  spoon-bot API Integration Tests         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Cleanup on exit (unless --keep-running)
cleanup() {
  if [ "$KEEP_RUNNING" = false ]; then
    echo ""
    echo "🧹 Stopping test container..."
    sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# 1. Build
if [ "$SKIP_BUILD" = false ]; then
  echo "🔨 Building Docker image..."
  cd "$PROJECT_DIR"
  DOCKER_BUILDKIT=1 sudo docker build -t spoon-bot:test . 2>&1 | tail -3
  echo "   ✓ Build complete"
else
  echo "⏭  Skipping build"
fi

# 2. Stop existing container
sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# 3. Start container
echo ""
echo "🚀 Starting spoon-bot container on port $PORT..."

# Source .env file for API keys
if [ -f "$PROJECT_DIR/.env" ]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi

sudo docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$PORT:8080" \
  -e SPOON_BOT_MODE=gateway \
  -e GATEWAY_HOST=0.0.0.0 \
  -e GATEWAY_PORT=8080 \
  -e GATEWAY_AUTH_REQUIRED="${GATEWAY_AUTH_REQUIRED:-false}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  -e DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}" \
  -e GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
  -e SPOON_BOT_DEFAULT_PROVIDER="${SPOON_PROVIDER:-openai}" \
  -e SPOON_BOT_DEFAULT_MODEL="${SPOON_MODEL:-}" \
  -e JWT_SECRET="test-secret-for-jwt-signing-key" \
  -e GATEWAY_API_KEY="${GATEWAY_API_KEY:-}" \
  spoon-bot:test

echo "   Container started: $CONTAINER_NAME"

# 4. Wait for healthy
echo ""
echo "⏳ Waiting for gateway to become healthy..."
HEALTHY=false
for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    HEALTHY=true
    echo "   ✓ Gateway healthy after ${i}s"
    break
  fi
  sleep 1
done

if [ "$HEALTHY" = false ]; then
  echo "   ✗ Gateway failed to start!"
  echo ""
  echo "Container logs:"
  sudo docker logs "$CONTAINER_NAME" 2>&1 | tail -50
  exit 1
fi

# 5. Run tests
echo ""
echo "🧪 Running API tests..."
echo ""

cd "$SCRIPT_DIR"
npm install --silent 2>/dev/null

SPOON_BOT_URL="http://localhost:$PORT" \
AUTH_REQUIRED="${GATEWAY_AUTH_REQUIRED:-false}" \
SPOON_BOT_API_KEY="${GATEWAY_API_KEY:-}" \
npx jest --verbose --forceExit --detectOpenHandles 2>&1

TEST_EXIT=$?

echo ""
if [ $TEST_EXIT -eq 0 ]; then
  echo "✅ All tests passed!"
else
  echo "❌ Some tests failed (exit code: $TEST_EXIT)"
  echo ""
  echo "Container logs (last 30 lines):"
  sudo docker logs "$CONTAINER_NAME" 2>&1 | tail -30
fi

exit $TEST_EXIT
