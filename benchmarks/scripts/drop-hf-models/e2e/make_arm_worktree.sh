#!/usr/bin/env bash
set -euo pipefail

HARNESS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HARNESS/env.sh"

[ $# -eq 2 ] || { echo "usage: make_arm_worktree.sh <pr1|pr2> <sha>" >&2; exit 2; }
tag=$1
sha=$2
target=$E2E_REPO_PARENT/prime-rl-run-drophf-$tag

if [ "$(git -C "$E2E_WT_MAIN" ls-tree HEAD deps/)" != "$(git -C "$E2E_WT_MAIN" ls-tree "$sha" deps/)" ]; then
  echo "$sha pins different submodules than main; seed them by hand" >&2
  exit 1
fi

git -C "$E2E_WT_MAIN" worktree add --detach "$target" "$sha"

seed() {
  local superproject=$1 reference=$2 path=$3
  rm -rf "${superproject:?}/$path"
  git clone -q --no-checkout "$reference/$path" "$superproject/$path"
  git -C "$superproject/$path" checkout -q "$(git -C "$reference/$path" rev-parse HEAD)"
  for nested in $(git -C "$reference/$path" submodule status 2>/dev/null | awk '{print $2}'); do
    seed "$superproject/$path" "$reference/$path" "$nested"
  done
}
for path in $(git -C "$E2E_WT_MAIN" submodule status | awk '{print $2}'); do
  seed "$target" "$E2E_WT_MAIN" "$path"
done
git -C "$target" submodule status
