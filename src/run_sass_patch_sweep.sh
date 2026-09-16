#!/usr/bin/env bash
set -euo pipefail
# Run ONLY inside an explicit safe GPU benchmark window. Does not signal/stop any trainer.
RUNNER=${RUNNER:-./run_sass_patch_probe}
DIR=${1:-/workspace/gb10_2pflop_openfork_20260916/static_recon/sass_patch}
ITERS=${ITERS:-1000}
for block in 32 64 128; do
  for bpsm in 2 4 8 16; do
    "$RUNNER" "$DIR/base.cubin" 16 "$ITERS" "$block" "$bpsm"
    "$RUNNER" "$DIR/fill25.cubin" 20 "$ITERS" "$block" "$bpsm"
    "$RUNNER" "$DIR/fill50.cubin" 23 "$ITERS" "$block" "$bpsm"
    "$RUNNER" "$DIR/fill75.cubin" 26 "$ITERS" "$block" "$bpsm"
    "$RUNNER" "$DIR/fill100.cubin" 30 "$ITERS" "$block" "$bpsm"
  done
done
