#!/usr/bin/env bash
set -euo pipefail

# Helper to ensure the venv exists, install local hammer, and then proxy to hammer.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv_path="${repo_root}/.venv"

python_cmd=$(command -v python3.11 || command -v python3 || command -v python || true)
if [ -z "${python_cmd}" ]; then
  echo "Python 3 is required but was not found in PATH." >&2
  exit 1
fi

if [ ! -d "${venv_path}" ]; then
  echo "Creating virtual environment at ${venv_path} using ${python_cmd}."
  "${python_cmd}" -m venv "${venv_path}"
fi

"${venv_path}/bin/pip" install -U pip setuptools wheel >/dev/null
"${venv_path}/bin/pip" install -e "${repo_root}"

export PATH="${venv_path}/bin:${PATH}"

exec hammer "$@"
