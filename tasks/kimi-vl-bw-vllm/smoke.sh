#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
DEFAULT_INPUTS=()
HAS_IMAGE=0
for ARG in "$@"; do case "$ARG" in --image|--image=*) HAS_IMAGE=1 ;; esac; done
if [[ "$HAS_IMAGE" == 0 ]]; then DEFAULT_INPUTS=(--image "$TASK_DIR/fixtures/red.png" --image "$TASK_DIR/fixtures/blue.png"); fi
exec "$PYTHON_BIN" "$TASK_DIR/../../scripts/recon2reason_smoke.py" --base-url "http://127.0.0.1:$KIMI_PORT" --model "$KIMI_SERVED_NAME" --log "$TASK_DIR/smoke.log" --timeout 120 --max-tokens 128 "${DEFAULT_INPUTS[@]}" "$@"
