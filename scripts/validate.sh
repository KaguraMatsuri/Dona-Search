#!/bin/bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
DONA_CHECK_PYTHON="${DONA_PYTHON:-python3}"
if [[ -z "${DONA_PYTHON:-}" && -x .venv/bin/python ]]; then
    DONA_CHECK_PYTHON=.venv/bin/python
fi

"$DONA_CHECK_PYTHON" -m compileall -q dona_search scripts examples
"$DONA_CHECK_PYTHON" -c 'from pathlib import Path; from dona_search.config import default_config, validate_config; validate_config(default_config(), Path.cwd(), require_model_paths=False)'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$DONA_CHECK_PYTHON" -m pytest -q tests "$@"
