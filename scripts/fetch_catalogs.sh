#!/usr/bin/env bash
# Fetch the public catalogs of the habit channel's role table at their pinned versions and check
# them (nothing is vendored: GTFOBins and LOLBAS are GPL-3.0).
#
#   scripts/fetch_catalogs.sh [DEST]
#
# DEST defaults to settings.paths.catalogs (configs/default.yaml: models/catalogs2; HABITUS_CATALOGS
# overrides it). Layout written:
#   DEST/Contents-amd64.gz        Debian bookworm main Contents-amd64 (sha256 pinned)
#   DEST/xcyclopedia.json.zip     Strontic xcyclopedia (sha256 pinned)
#   DEST/gtfobins/                GTFOBins.github.io at a pinned commit
#   DEST/lolbas/                  LOLBAS at a pinned commit
# Finally the role table is built from DEST and compared with vocab.ROLE_TABLE_SHA256.
#
# Writes happen only inside this repository and never under a read-only input (config.check_writable:
# the configured server inputs, and models / laya_hub / catalogs whenever they lie outside the repo).
# Any other DEST, e.g. a catalog checkout in another tree, is VERIFIED only: the pinned sha256s, the
# pinned commits (read with `git --no-optional-locks rev-parse`, which writes nothing) and the role
# table; nothing is fetched, checked out, created or renamed there. To fetch a fresh copy, pass a DEST
# inside the repo, e.g. `scripts/fetch_catalogs.sh models/catalogs2`, and point paths.catalogs at it.
# A clone already at its pinned commit is not fetched or checked out again.
#
# The Debian Contents file of a stable release changes with its point releases; when the current
# file no longer matches the pinned sha256, set CONTENTS_URL to a snapshot.debian.org copy of the
# pinned one. XCYCLOPEDIA_URL overrides the xcyclopedia location the same way.
set -euo pipefail

GTFOBINS_REPO="https://github.com/GTFOBins/GTFOBins.github.io.git"
GTFOBINS_COMMIT="acd524623f9c406acedd2754ebd9c2431f3675ad"
LOLBAS_REPO="https://github.com/LOLBAS-Project/LOLBAS.git"
LOLBAS_COMMIT="2056cb7662509dc998fc6ce79636aec4e6c0397f"
CONTENTS_URL="${CONTENTS_URL:-https://deb.debian.org/debian/dists/bookworm/main/Contents-amd64.gz}"
CONTENTS_SHA256="7e72214b4cef520fa07fc40ff975c60bc4501eca6308cf5cd9804c00811b776b"
XCYCLOPEDIA_URL="${XCYCLOPEDIA_URL:-https://github.com/Strontic/xcyclopedia/raw/master/strontic-xcyclopedia.json.zip}"
XCYCLOPEDIA_SHA256="6fb323a5d27637048e52e58102e98415705ef30611e6e81639d9266b44ed9a46"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

DEST="${1:-}"
if [ -z "$DEST" ]; then
  DEST="$("$PY" -c 'from habitus.config import settings; print(settings().paths.catalogs)')"
fi
case "$DEST" in /*) ;; *) DEST="$PWD/$DEST" ;; esac
# MODE write: DEST is inside the repository and passes config.check_writable; verify otherwise
MODE="$("$PY" - "$DEST" "$REPO" <<'EOF'
import os, sys
from habitus import config
dest, repo = os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])
try:
    config.check_writable(dest)
    ok = config.is_under(dest, repo)
except config.ConfigError as e:
    print(f"read-only: {e}", file=sys.stderr)
    ok = False
print("write" if ok else "verify")
EOF
)"
if [ "$MODE" = "write" ]; then
  mkdir -p "$DEST"
  echo "catalogs -> $DEST (fetch + verify)"
else
  echo "catalogs at $DEST: outside this repository or a read-only input; VERIFY only (nothing is written)"
  if [ ! -d "$DEST" ]; then
    echo "$DEST does not exist; pass a DEST inside the repository, e.g. models/catalogs2" >&2
    exit 1
  fi
fi

check_sha() {   # file expected
  local got
  got="$(sha256sum "$1" | cut -d' ' -f1)"
  if [ "$got" != "$2" ]; then
    echo "sha256 mismatch for $1: got $got, pinned $2" >&2
    return 1
  fi
}

repo_head() {   # dir -> HEAD commit (read only: no index refresh, no lock file)
  git --no-optional-locks -C "$1" rev-parse HEAD 2>/dev/null || true
}

fetch_file() {  # url dest sha
  if [ -f "$2" ] && check_sha "$2" "$3" 2>/dev/null; then
    echo "ok   $2"
    return 0
  fi
  if [ "$MODE" != "write" ]; then
    echo "$2: missing or not the pinned sha256 $3 (verify only: not fetched)" >&2
    return 1
  fi
  echo "get  $1"
  curl -fL --retry 3 -o "$2.part" "$1"
  check_sha "$2.part" "$3"
  mv "$2.part" "$2"
}

fetch_repo() {  # url commit dest
  local head
  head="$(repo_head "$3")"
  if [ "$head" = "$2" ]; then
    echo "ok   $3 @ $2"
    return 0
  fi
  if [ "$MODE" != "write" ]; then
    echo "$3 is at ${head:-<no clone>}, pinned $2 (verify only: not fetched or checked out)" >&2
    return 1
  fi
  if [ ! -d "$3/.git" ]; then
    git clone --quiet "$1" "$3"
  fi
  git -C "$3" fetch --quiet origin "$2" || git -C "$3" fetch --quiet origin
  git -C "$3" -c advice.detachedHead=false checkout --quiet "$2"
  head="$(repo_head "$3")"
  if [ "$head" != "$2" ]; then
    echo "$3 is at $head, pinned $2" >&2
    return 1
  fi
  echo "ok   $3 @ $2"
}

fetch_file "$CONTENTS_URL" "$DEST/Contents-amd64.gz" "$CONTENTS_SHA256"
fetch_file "$XCYCLOPEDIA_URL" "$DEST/xcyclopedia.json.zip" "$XCYCLOPEDIA_SHA256"
fetch_repo "$GTFOBINS_REPO" "$GTFOBINS_COMMIT" "$DEST/gtfobins"
fetch_repo "$LOLBAS_REPO" "$LOLBAS_COMMIT" "$DEST/lolbas"

HABITUS_CATALOGS="$DEST" "$PY" - <<'EOF'
from habitus.questions.habit import vocab
st = vocab.ROLE_TABLE.status()
print(f"role table sha256 {st['sha256']} (frozen {st['frozen']}): posix {st['sizes']['posix']}, "
      f"windows {st['sizes']['windows']}")
raise SystemExit(0 if st["available"] else 1)
EOF
