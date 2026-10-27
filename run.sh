#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${KIMI_PYTHON_BIN:-python3}" "$REPOSITORY_ROOT/scripts/prepare_image_model021.py"
bash "$REPOSITORY_ROOT/scripts/apply_profile016.sh"
exec bash "$REPOSITORY_ROOT/tasks/kimi-vl-bw-vllm/run.sh" "$@"
