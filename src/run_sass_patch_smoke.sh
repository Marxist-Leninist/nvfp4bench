#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/gb10_2pflop_openfork_20260916/static_recon/sass_patch
GUARD=/workspace/gb10_2pflop_openfork_20260916/upstream/src/run_2pf_guarded.sh
R=$ROOT/run_sass_patch_probe
# Deliberately tiny progression. Base first proves normal cubin loading; patched25 second tests patched SASS.
# Only after both pass should the full sweep run.
"$GUARD" "$R" "$ROOT/base.cubin" 16 50 32 1
"$GUARD" "$R" "$ROOT/fill25.cubin" 20 50 32 1
