#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
YROBOT_VENV="${YROBOT_VENV:-.venv}"

if command -v uv >/dev/null 2>&1; then
  uv venv --python "${PYTHON_VERSION}" "${YROBOT_VENV}"
else
  "python${PYTHON_VERSION}" -m venv "${YROBOT_VENV}"
fi

# shellcheck disable=SC1090
source "${YROBOT_VENV}/bin/activate"

if command -v uv >/dev/null 2>&1; then
  uv pip install -e .
else
  python -m pip install -e .
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo "YRobot is ready. Run: source ${YROBOT_VENV}/bin/activate && yrobot"
