#!/usr/bin/env bash
# Use the dedicated deploy key configured locally in this exported repository.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
expected_origin='git@github.com:John122-art/Drivor-STeervit-data-stage1.git'
if [[ "$(git remote get-url origin)" != "$expected_origin" ]]; then
    echo 'Unexpected origin; refusing to push.' >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo 'There are uncommitted changes; inspect and commit them before uploading.' >&2
    exit 1
fi
if ! remote_refs="$(git ls-remote origin)"; then
    echo 'Cannot access the repository. Add its dedicated public key and enable Allow write access:' >&2
    echo 'https://github.com/John122-art/Drivor-STeervit-data-stage1/settings/keys' >&2
    exit 1
fi
if [[ -n "$remote_refs" ]]; then
    if ! main_ref="$(git ls-remote origin refs/heads/main)"; then
        exit 1
    fi
    if [[ -z "$main_ref" ]]; then
        echo 'Remote has content but no main branch. Inspect it before choosing a target branch.' >&2
        exit 1
    fi
    git fetch origin main
    if ! git merge-base --is-ancestor origin/main HEAD; then
        if ! git merge --allow-unrelated-histories --no-edit origin/main; then
            echo 'Merge stopped for review. Resolve the reported conflicts, commit, and rerun this script.' >&2
            exit 1
        fi
    fi
fi
git push --set-upstream origin HEAD:main
local_commit="$(git rev-parse HEAD)"
remote_ref="$(git ls-remote origin refs/heads/main)"
read -r remote_commit _ <<< "$remote_ref"
if [[ "$local_commit" != "$remote_commit" ]]; then
    echo 'Remote commit differs from the uploaded commit; inspect before proceeding.' >&2
    exit 1
fi
echo "Upload verified: $local_commit"
echo 'https://github.com/John122-art/Drivor-STeervit-data-stage1'
