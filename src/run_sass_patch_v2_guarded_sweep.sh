#!/usr/bin/env bash
set -euo pipefail
R=/workspace/gb10_2pflop_openfork_20260916/upstream
D=/workspace/gb10_2pflop_openfork_20260916/static_recon/sass_patch_v2
G=$R/src/run_2pf_guarded.sh
V=$D/run_correctness_v2
# Phase A: arithmetic witness. Each process is guarded independently and never signals another job.
for tag in fill25 fill50 fill75 fill100; do
  read a0 a1 < <(python3 - "$D/$tag.patch.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); print(*j['added_by_bank'])
PY
)
  "$G" "$V" "$D/base.cubin" "$D/$tag.cubin" "$a0" "$a1" 8
done
# Phase B performance is intentionally a separate explicit step after Phase A passes.
echo 'CORRECTNESS_PHASE_PASS'
