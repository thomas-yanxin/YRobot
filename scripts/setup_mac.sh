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

# --- llama.cpp (serves the MiniCPM-V-4.6 VLM) ------------------------------
echo "==> Installing llama.cpp (llama-server)"
if ! command -v llama-server >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install llama.cpp || echo "brew install llama.cpp failed — install it manually." >&2
  else
    echo "Homebrew not found. Install llama.cpp manually: https://github.com/ggml-org/llama.cpp" >&2
  fi
fi

# MiniCPM-V-4.6 GGUF weights (LLM + vision projector) are fetched automatically by
# `llama-server -hf ...` on first launch (it also pulls the matching mmproj), so no
# separate download step is needed here.

cat <<EOF

==> Done.

Next:
  source ${VENV}/bin/activate

  # Dev without any hardware or models:
  reachy-mini-live-chat --sim --stub

  # Real local pipeline (mock robot). Start the MiniCPM-V-4.6 server — the first run
  # auto-downloads the model + vision projector (~2-3 GB) from Hugging Face.
  # --reasoning off disables the thinking trace (needed for low latency; vision still works):
  llama-server -hf openbmb/MiniCPM-V-4.6-gguf:Q4_K_M --reasoning off --host 0.0.0.0 --port 8080 -c 4096 &
  reachy-mini-live-chat --sim

  # (Manual weights instead of -hf:
  #   huggingface-cli download openbmb/MiniCPM-V-4.6-gguf \\
  #     MiniCPM-V-4.6-Q4_K_M.gguf mmproj-MiniCPM-V-4.6-F16.gguf --local-dir models/MiniCPM-V-4.6
  #   llama-server -m models/MiniCPM-V-4.6/MiniCPM-V-4.6-Q4_K_M.gguf \\
  #     --mmproj models/MiniCPM-V-4.6/mmproj-MiniCPM-V-4.6-F16.gguf --port 8080 -c 4096 )

  # On the robot:
  reachy-mini-daemon &
  reachy-mini-live-chat
EOF
