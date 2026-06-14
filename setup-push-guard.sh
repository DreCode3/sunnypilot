#!/bin/sh
# Lock this sunnypilot/openpilot clone so it can push ONLY to your GitHub fork (github.com/<OWNER>).
# Pushes to upstream (sunnypilot / sunnyhaibin / commaai / bluepilot) are BLOCKED. Use this until you're
# ready to contribute upstream. Idempotent — safe to re-run. Run from the repo root.
#
#   Usage:  ./setup-push-guard.sh [OWNER]      (OWNER defaults to DreCode3)
#
# Applies three layers per repo (main repo + each INITIALIZED submodule):
#   1. remote.pushDefault = your fork  -> a bare `git push` always targets your fork
#   2. every non-<OWNER> remote's PUSH url -> DISABLED://...  -> `git push upstream` fails fast
#   3. a pre-push guard hook rejects any push to a non-<OWNER> URL (covers remotes added later) + keeps Git LFS
#
# NOTE: these settings live in each repo's local .git (config + hooks) and are NOT committed, so re-run this
# script after a fresh clone or after `git submodule update --init` (to guard newly-initialized submodules).
set -eu
OWNER="${1:-DreCode3}"

write_guard_hook() {  # $1 = absolute hooks dir
  cat > "$1/pre-push" <<EOF
#!/bin/sh
# Guard (setup-push-guard.sh): this repo may push ONLY to github.com/$OWNER/*.
url="\$2"
case "\$url" in
  *github.com/$OWNER/*|*github.com:$OWNER/*) : ;;
  *) printf >&2 "\n[PUSH BLOCKED] '%s' is not your $OWNER fork.\nThis repo pushes only to github.com/$OWNER/* (override: git push --no-verify).\n\n" "\$url"; exit 1 ;;
esac
command -v git-lfs >/dev/null 2>&1 && git lfs pre-push "\$@"
exit 0
EOF
  chmod +x "$1/pre-push"
}

guard_repo() {  # $1 = repo path
  d="$1"
  for rem in $(git -C "$d" remote); do
    u=$(git -C "$d" remote get-url "$rem" 2>/dev/null || echo "")
    case "$u" in
      *github.com/"$OWNER"/*|*github.com:"$OWNER"/*)
        git -C "$d" config remote.pushDefault "$rem" ;;        # your fork -> default push target
      *)
        git -C "$d" remote set-url --push "$rem" "DISABLED://push-only-to-$OWNER-fork" ;;
    esac
  done
  write_guard_hook "$(git -C "$d" rev-parse --path-format=absolute --git-path hooks)"
  echo "  guarded: $d"
}

echo "Locking all pushes to github.com/$OWNER/* ..."
guard_repo .
# only INITIALIZED submodules: `git submodule foreach` never enters uninitialized ones (which otherwise
# resolve to the PARENT .git and would clobber its remotes).
git submodule --quiet foreach 'echo "$sm_path"' 2>/dev/null | while read -r sm; do guard_repo "$sm"; done
echo "done — verify with:  git remote -v | grep push"
