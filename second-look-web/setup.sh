#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10–3.12"'
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
printf '%s\n' 'Setup complete. Model paths must be configured separately; no model was downloaded.'
