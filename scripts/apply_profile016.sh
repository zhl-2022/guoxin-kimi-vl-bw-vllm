#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="$ROOT/patches/0001-kimi-verified-profile016.patch"
if git -C "$ROOT" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  printf '%s\n' 'Verified profile patch already applied.'
elif git -C "$ROOT" apply --check "$PATCH"; then
  git -C "$ROOT" apply "$PATCH"
else
  printf '%s\n' 'Profile source differs; refusing to apply a partial or mismatched patch.' >&2
  exit 2
fi
