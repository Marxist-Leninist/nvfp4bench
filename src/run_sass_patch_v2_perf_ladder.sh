#!/usr/bin/env bash
set -euo pipefail
R=/workspace/gb10_2pflop_openfork_20260916/upstream
D=/workspace/gb10_2pflop_openfork_20260916/static_recon/sass_patch_v2
S=$D/stall_exact
G=$R/src/run_2pf_guarded.sh
B=$D/run_bench_v2
# This is intentionally conservative: correctness+timing one candidate at a time.
# Do not execute unless the exact SG fence is active and production is absent.
run(){ local cub=$1 omma=$2 a0=$3 a1=$4; "$G" "$B" "$D/base.cubin" "$cub" "$omma" "$a0" "$a1" 128 8 700; }
# Isolate scheduler control first with no extra work.
for st in 12 10 8 7 6 4 2 1; do run "$S/base_stall${st}.cubin" 16 0 0; done
# Then add the lowest-density NOP-fill at each already-correct stall.
for st in 12 10 8 7 6 4 2 1; do run "$S/fill25_stall${st}.cubin" 20 2 2; done
