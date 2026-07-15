#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reachy Mini — on-robot (Wireless CM4) setup.
# The conversational brain is REMOTE (MiniCPM-o 4.5 on llama-omni-server), so the
# robot side is a thin client: the only hard deps are numpy + websockets. No torch,
# no CUDA, no local ML. This installs the app into a Python 3.12 venv. Optional
# extras (web UI, neural VAD, pillow) can be added later — each has a fallback.
#
# Run this ON the robot's CM4 (ARM64 Linux). `reachy-mini` is usually already present
# in the robot image; if so the `.[cm4]` reinstall is a no-op for it.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-.venv}"

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install "$@"
  else
    python -m pip install "$@"
  fi
}

echo "==> Creating venv (${VENV}, Python ${PYTHON_VERSION})"
if command -v uv >/dev/null 2>&1; then
  uv venv --python "${PYTHON_VERSION}" "${VENV}"
else
  "python${PYTHON_VERSION}" -m venv "${VENV}"
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

echo "==> Installing app + CM4 stack (no local ML models — brain is remote)"
pip_install -e ".[cm4,dev]" || {
  echo "Full install failed — falling back to light install (sim/stub only)." >&2
  pip_install -e ".[dev]"
}

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Wrote .env from template — set OMNI_WS_URL to your llama-omni-server."
fi

cat <<EOF

==> Done.

Next:
  source ${VENV}/bin/activate

  # 1) Point the app at your omni server (edit .env):
  #      OMNI_WS_URL=wss://<server-host>:8006
  #    (the server is llama.cpp-omni's llama-omni-server; see its deploy docs)

  # 2) Dev without any hardware or server (runs the real client against a fake omni server):
  reachy-mini-live-chat --sim --stub

  # 3) On the robot (daemon already running on the CM4):
  reachy-mini-live-chat
  #    or let the Reachy Mini app launcher discover it (entry point: reachy_mini_apps).
EOF
