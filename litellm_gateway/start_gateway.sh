#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export FPT_API_KEY="..."
#   ./litellm_gateway/start_gateway.sh
#
# Optional (protect local gateway):
#   export LITELLM_MASTER_KEY="my-local-key"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

HOST="${LITELLM_HOST:-0.0.0.0}"
PORT="${LITELLM_PORT:-4000}"

exec litellm \
  --config "$CONFIG_FILE" \
  --host "$HOST" \
  --port "$PORT"
