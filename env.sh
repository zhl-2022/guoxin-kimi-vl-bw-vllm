#!/usr/bin/env bash
_TASK_REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_TASK_REPOSITORY_ROOT/tasks/kimi-vl-bw-vllm/env.sh"
if [[ ! -f "$KIMI_DATA_DIR/samples.jsonl" && -f /opt/guoxin-kimi021/data/samples.jsonl ]]; then
  export KIMI_DATA_DIR=/opt/guoxin-kimi021/data
fi
unset _TASK_REPOSITORY_ROOT
