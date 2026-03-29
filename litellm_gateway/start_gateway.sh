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

# Optionally load environment variables from a .env file so users don't need to
# prefix the command with exports.
#
# Search order:
#   1) LITELLM_DOTENV_PATH (explicit)
#   2) litellm_gateway/.env
#   3) repo root .env (litellm_gateway/../.env)
DOTENV_FILE="${LITELLM_DOTENV_PATH:-}"
if [[ -z "${DOTENV_FILE}" ]]; then
  if [[ -f "$SCRIPT_DIR/.env" ]]; then
    DOTENV_FILE="$SCRIPT_DIR/.env"
  elif [[ -f "$SCRIPT_DIR/../.env" ]]; then
    DOTENV_FILE="$SCRIPT_DIR/../.env"
  fi
fi

if [[ -n "${DOTENV_FILE}" && -f "${DOTENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${DOTENV_FILE}"
  set +a
  echo "[litellm_gateway] Loaded env from: ${DOTENV_FILE}" >&2
fi

HOST="${LITELLM_HOST:-0.0.0.0}"
PORT="${LITELLM_PORT:-4000}"

if command -v litellm >/dev/null 2>&1; then
  exec litellm \
    --config "$CONFIG_FILE" \
    --host "$HOST" \
    --port "$PORT"
fi

# Fallback if the console script isn't available (e.g., some environments).
exec python -m litellm proxy \
  --config "$CONFIG_FILE" \
  --host "$HOST" \
  --port "$PORT"
