#!/usr/bin/env bash
# CI-style check: the unit tests on a copy of exactly the files a commit would contain (git ls-files:
# tracked + untracked-not-ignored), WITHOUT release/, configs/local.yaml, models/ or runs/, i.e. what
# an outside user has after a fresh clone. `make test` must pass there (tests needing the bundle,
# a tokenizer, model weights or catalogs skip).
#
#   scripts/dev/fresh_clone_tests.sh [DEST]        # DEST: an empty or absent scratch directory
#
# PYTHON (default: python) is the interpreter with the pinned environment. Writes only under DEST.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python}"
DEST="${1:-$(mktemp -d -t habitus-fresh-XXXXXX)}"
if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
  echo "$DEST is not empty" >&2
  exit 2
fi
mkdir -p "$DEST"
git -C "$REPO" ls-files -z --cached --others --exclude-standard | (cd "$REPO" && xargs -0 cp --parents -t "$DEST")
for p in release configs/local.yaml models runs; do
  if [ -e "$DEST/$p" ]; then
    echo "fresh copy unexpectedly holds $p" >&2
    exit 2
  fi
done
git -C "$DEST" init --quiet                      # the repository scans list files through git
cd "$DEST"
env -u HABITUS_CONFIG -u HABITUS_DATA -u HABITUS_GT -u HF_HOME -u HABITUS_LAYA_HUB -u HABITUS_CATALOGS \
    -u PIDSMAKER_ARTIFACTS -u PIDSMAKER_UPSTREAM_ARTIFACTS -u HABITUS_REQUIRE_ASSETS -u HABITUS_REQUIRE_LOCAL_ASSETS \
    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="${THREADS:-8}" \
    MKL_NUM_THREADS="${THREADS:-8}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PY" -m pytest -q -p no:cacheprovider tests/ -m "not provenance"
