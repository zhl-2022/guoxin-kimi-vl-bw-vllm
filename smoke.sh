#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$REPOSITORY_ROOT/tasks/kimi-vl-bw-vllm/smoke.sh" "$@"
