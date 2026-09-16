#!/usr/bin/env bash
set -euo pipefail
R=/workspace/gb10_2pflop_openfork_20260916/upstream
D=/workspace/gb10_2pflop_openfork_20260916/static_recon/sass_patch_v2
S=$D/stall_exact
G=$R/src/run_2pf_guarded.sh
V=$D/run_correctness_v2
# Correctness-first, high-stall -> low-stall. No timing here.
# Each invocation independently rechecks the SG fence and CUDA/process census.
# Stop on the first loader/arithmetic failure because set -e is intentional.
for st in 12 10 8 6 4 2 1; do
  echo "=== base stall=$st ==="
  "$G" "$V" "$D/base.cubin" "$S/base_stall${st}.cubin" 0 0 8
  echo "=== fill25 stall=$st ==="
  "$G" "$V" "$D/base.cubin" "$S/fill25_stall${st}.cubin" 2 2 8
done
echo STALL_CORRECTNESS_LADDER_PASS
