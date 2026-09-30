#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
UV_PROJECT_PYTHON="$ROOT_DIR/.venv/bin/python"
LOCAL_UNI2TS_DIR="$ROOT_DIR/_third_party/uni2ts"
REMOTE_UNI2TS_REF="uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git@7a4e77cff8d6cf480c559c6b6879ee621a492a42"

if command -v uv >/dev/null 2>&1; then
  if uv run python - <<'PY' >/dev/null 2>&1
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
import jaxtyping
print(Moirai2Forecast, Moirai2Module, jaxtyping)
PY
  then
    echo "[SKIP] Moirai2 runtime backend already present in uv environment."
    exit 0
  fi

  echo "[INSTALL] Moirai2 runtime backend for uv environment (uni2ts + jaxtyping)"
  if [ ! -x "$UV_PROJECT_PYTHON" ]; then
    echo "[INIT] Creating project uv environment"
    uv venv
  fi
  if [ -d "$LOCAL_UNI2TS_DIR" ]; then
    uv pip install --python "$UV_PROJECT_PYTHON" "$LOCAL_UNI2TS_DIR"
  else
    uv pip install --python "$UV_PROJECT_PYTHON" "$REMOTE_UNI2TS_REF"
  fi
  uv pip install --python "$UV_PROJECT_PYTHON" --no-deps "jaxtyping>=0.3,<0.4"
  exit 0
fi

if python - <<'PY' >/dev/null 2>&1
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
import jaxtyping
print(Moirai2Forecast, Moirai2Module, jaxtyping)
PY
then
  echo "[SKIP] Moirai2 runtime backend already present."
  exit 0
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Neither uv nor python found for Moirai2 backend bootstrap." >&2
  exit 1
fi

echo "[INSTALL] Moirai2 runtime backend via pip (uni2ts + jaxtyping)"
if [ -d "$LOCAL_UNI2TS_DIR" ]; then
  pip install "$LOCAL_UNI2TS_DIR"
else
  pip install "$REMOTE_UNI2TS_REF"
fi
pip install --no-deps "jaxtyping>=0.3,<0.4"
