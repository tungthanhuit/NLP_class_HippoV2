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

config_enables_master_key() {
  # Detect uncommented master_key setting in config.
  grep -Eq '^[[:space:]]*master_key[[:space:]]*:' "$CONFIG_FILE"
}

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

# Guard against accidental auth mode: LiteLLM may read LITELLM_MASTER_KEY from env
# even when config does not explicitly enable master_key. Treat an empty but defined
# variable as set and unset it as well.
if [[ "${LITELLM_MASTER_KEY+x}" == "x" ]] && ! config_enables_master_key; then
  echo "[litellm_gateway] Ignoring LITELLM_MASTER_KEY from env because config does not enable general_settings.master_key." >&2
  echo "[litellm_gateway] To require client auth, set master_key in config.yaml and export OPENAI_API_KEY=<same-key> for clients." >&2
  unset LITELLM_MASTER_KEY
fi

HOST="${LITELLM_HOST:-0.0.0.0}"
PORT="${LITELLM_PORT:-4000}"

# Resolve Python interpreter in a venv-safe order:
# 1) explicit override via LITELLM_PYTHON
# 2) active virtualenv
# 3) repo-local .venv
# 4) PATH python3/python fallback
PYTHON_BIN=""
if [[ -n "${LITELLM_PYTHON:-}" ]]; then
  PYTHON_BIN="${LITELLM_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "$SCRIPT_DIR/../.venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "[litellm_gateway] ERROR: no Python interpreter found." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import litellm' >/dev/null 2>&1; then
  echo "[litellm_gateway] ERROR: litellm is not installed for: $PYTHON_BIN" >&2
  echo "[litellm_gateway] Install with: $PYTHON_BIN -m pip install \"litellm[proxy]\"" >&2
  exit 1
fi

echo "[litellm_gateway] Using Python: $PYTHON_BIN" >&2
if [[ -x "$(dirname "$PYTHON_BIN")/litellm" ]]; then
  exec "$(dirname "$PYTHON_BIN")/litellm" \
    --config "$CONFIG_FILE" \
    --host "$HOST" \
    --port "$PORT"
fi

exec "$PYTHON_BIN" -m litellm.proxy.proxy_cli \
  --config "$CONFIG_FILE" \
  --host "$HOST" \
  --port "$PORT"
