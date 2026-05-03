#!/bin/bash
# Pre-commit hook: ensure every params.put*/get* key in staged Python files is
# registered in common/params_keys.h. Unregistered writes raise UnknownKeyName
# at runtime and crash the calling process (observed 2026-05-03 — PI lane
# centering crashed card.py via unregistered LaneBiasIntegral).
#
# Install:
#   ln -sf ../../tools/check_params_keys.sh .git/hooks/pre-commit
#   ln -sf ../../../../opendbc_repo/../tools/check_params_keys.sh \
#          .git/modules/opendbc/hooks/pre-commit
#
# Bypass once (use sparingly):  git commit --no-verify
set -e

# Locate params_keys.h whether running from main repo or a submodule
KEYS_FILE=""
for candidate in common/params_keys.h ../common/params_keys.h ../../common/params_keys.h; do
  if [ -f "$candidate" ]; then KEYS_FILE="$candidate"; break; fi
done
if [ -z "$KEYS_FILE" ]; then
  echo "warn: check_params_keys.sh — params_keys.h not found, skipping" >&2
  exit 0
fi

REGISTERED=$(grep -oE '\{"[A-Za-z0-9_]+",' "$KEYS_FILE" | tr -d '{",' | sort -u)

STAGED=$(git diff --cached --name-only --diff-filter=ACMR | grep '\.py$' || true)
if [ -z "$STAGED" ]; then exit 0; fi

USED=$(echo "$STAGED" | while read -r f; do
  [ -f "$f" ] && grep -hoE 'params\.(put|put_nonblocking|put_bool|get|get_bool)\("[A-Za-z0-9_]+"' "$f" || true
done | grep -oE '"[A-Za-z0-9_]+"' | tr -d '"' | sort -u)

if [ -z "$USED" ]; then exit 0; fi

UNREG=$(comm -23 <(echo "$USED") <(echo "$REGISTERED"))

if [ -n "$UNREG" ]; then
  echo "ERROR: params keys used in staged files but not registered in $KEYS_FILE:" >&2
  echo "$UNREG" | sed 's/^/  - /' >&2
  echo "" >&2
  echo "Add each missing key to common/params_keys.h before committing." >&2
  echo "Format: {\"KeyName\", {PERSISTENT, BOOL|STRING|INT, \"default\"}}" >&2
  exit 1
fi
