#!/usr/bin/env bash
# Build data/brisc2025_manifest.csv from the original BRISC 2025 download in $DATA_ROOT.
# Skip this step if you already have the manifest used in the paper (set MANIFEST=...).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ -f "$MANIFEST" ]]; then echo "manifest exists: $MANIFEST"; exit 0; fi
"$PY" -m scripts.prepare_brisc --data-root "$DATA_ROOT" --output "$MANIFEST"
