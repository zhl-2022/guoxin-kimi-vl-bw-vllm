#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/kimi_vl_runtime_005.py" stop --task-dir "$TASK_DIR" "$@"
