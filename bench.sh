#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--plan-only" ]]; then
  exec "${KIMI_PYTHON_BIN:-python3}" "$ROOT/scripts/bench_matrix021.py" --project "$ROOT" "$@"
fi
source "$ROOT/env.sh"
exec "$PYTHON_BIN" "$ROOT/scripts/bench_matrix021.py" --project "$ROOT" "$@"
