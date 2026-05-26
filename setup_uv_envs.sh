#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${SCRIPT_DIR}"
DEFAULT_RUNTIME_ROOT="${PROJECT_ROOT}/.runtime"

RUNTIME_ROOT="${WTA_RUNTIME_ROOT:-${DEFAULT_RUNTIME_ROOT}}"
ENV_ROOT="${WTA_ENV_ROOT:-${RUNTIME_ROOT}/envs}"
LOG_ROOT="${WTA_LOG_ROOT:-${RUNTIME_ROOT}/logs}"
CACHE_ROOT="${WTA_CACHE_ROOT:-${RUNTIME_ROOT}/cache}"
PYTHON_SPEC="${WTA_PYTHON_SPEC:-3.12}"

if command -v module >/dev/null 2>&1; then
  module load python/3.12 >/dev/null 2>&1 || module load python/3.11.11 >/dev/null 2>&1 || true
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
UV_BIN=$(command -v uv)

mkdir -p "${ENV_ROOT}" "${LOG_ROOT}" "${CACHE_ROOT}" "${CACHE_ROOT}/bin"

"${UV_BIN}" venv --python "${PYTHON_SPEC}" "${ENV_ROOT}/core"
"${UV_BIN}" pip install --index-strategy unsafe-best-match --python "${ENV_ROOT}/core/bin/python" -r "${PROJECT_ROOT}/requirements.txt"

"${UV_BIN}" venv --python "${PYTHON_SPEC}" "${ENV_ROOT}/swift"
"${UV_BIN}" pip install --index-strategy unsafe-best-match --python "${ENV_ROOT}/swift/bin/python" -r "${PROJECT_ROOT}/requirements-swift.txt"

"${UV_BIN}" venv --python "${PYTHON_SPEC}" "${ENV_ROOT}/vllm-omni"
"${UV_BIN}" pip install --index-strategy unsafe-best-match --python "${ENV_ROOT}/vllm-omni/bin/python" -r "${PROJECT_ROOT}/requirements-vllm.txt"

ln -sfn "${ENV_ROOT}/core" "${PROJECT_ROOT}/.venv"
ln -sfn "${ENV_ROOT}/swift" "${PROJECT_ROOT}/.venv-swift"
ln -sfn "${ENV_ROOT}/vllm-omni" "${PROJECT_ROOT}/.venv-vllm-omni"

echo "Runtime root: ${RUNTIME_ROOT}"
echo "Core env: ${ENV_ROOT}/core"
echo "Swift env: ${ENV_ROOT}/swift"
echo "vLLM env: ${ENV_ROOT}/vllm-omni"
