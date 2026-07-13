#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reachy Mini Live Chat — Apple Silicon setup.
# Creates a Python 3.12 uv virtualenv and installs the app + local model stack.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-.venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it: https://docs.astral.sh/uv/  (brew install uv)" >&2
  exit 1
fi

echo "==> Creating venv (${VENV}, Python ${PYTHON_VERSION})"
uv venv --python "${PYTHON_VERSION}" "${VENV}"
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

echo "==> Installing app + Mac stack (this pulls MLX / FunASR / Kokoro; large)"
uv pip install -e ".[mac,dev]" || {
  echo "Full install failed — falling back to light install (sim/stub only)." >&2
  uv pip install -e ".[dev]"
}

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Wrote .env from template — edit tokens/model ids as needed."
fi

cat <<'EOF'

==> Done.

Next:
  source .venv/bin/activate

  # Dev without any hardware or models:
  reachy-mini-live-chat --sim --stub

  # Real local pipeline (mock robot). First, start the text LLM server:
  python -m mlx_lm server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080 &
  reachy-mini-live-chat --sim

  # On the robot:
  reachy-mini-daemon &
  reachy-mini-live-chat
EOF
