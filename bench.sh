#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh"
RUNTIME="$ROOT/tasks/kimi-vl-bw-vllm/runtime"
RUN_ID="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUNTIME/state.json")"
export KIMI_CLIENT_MATH_HOOK="$ROOT/scripts/kimi-math-hook/sitecustomize.py"
OUT="${KIMI_BENCH_OUTPUT:-$ROOT/bench/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p -- "$ROOT/bench"
export KIMI_CLIENT_COMPAT_AUDIT="$ROOT/bench/client-compat-$(date -u +%Y%m%dT%H%M%SZ).jsonl"
export PYTHONPATH="$ROOT/scripts/kimi-client-hook:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT/scripts/t02_sglang_bench_016.py" --run \
  --client-python "${KIMI_CLIENT_PYTHON:-/root/private_data/sglang-client-004c/bin/python}" \
  --allow-client-device-discovery --model-path "$KIMI_MODEL_PATH" --served-model-name "$KIMI_SERVED_NAME" \
  --port "$KIMI_PORT" --runtime-dir "$RUNTIME" --expected-run-id "$RUN_ID" \
  --output-dir "$OUT" --timeout-seconds 5400 --total-timeout-seconds 20000 "$@"
