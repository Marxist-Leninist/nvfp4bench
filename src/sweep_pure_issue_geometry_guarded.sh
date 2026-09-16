#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STATIC=/workspace/gb10_2pflop_openfork_20260916/static_recon/control_probe2/pure_issue_witness
MIXED=/workspace/gb10_2pflop_openfork_20260916/static_recon/control_probe2/mixed_threshold_witness/fill100_mixed5_stall7_rest8_yield0.cubin
OUT=${1:-/workspace/gb10_2pflop_openfork_20260916/static_recon/pure_issue_geometry_$(date -u +%Y%m%dT%H%M%SZ)}
ITERS=${ITERS:-700}
REPS=${REPS:-11}
mkdir -p "$OUT"
LOG="$OUT/results.jsonl"
: > "$LOG"

ones=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
fill=(2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2)

# Correctness is intentionally rechecked in the same bounded runtime window before timing.
"$HERE/run_pure_issue_guarded.sh" "$STATIC/spacing_stall7.cubin" 8 "${ones[@]}"
"$HERE/run_pure_issue_guarded.sh" "$STATIC/spacing_stall6.cubin" 8 "${ones[@]}"
"$HERE/run_pure_issue_guarded.sh" "$STATIC/fill100_stall8_yield0.cubin" 8 "${fill[@]}"
"$HERE/run_pure_issue_guarded.sh" "$MIXED" 8 "${fill[@]}"
"$HERE/run_pure_issue_guarded.sh" "$STATIC/fill100_stall7_yield0.cubin" 8 "${fill[@]}"

# candidate|OMMAs per warp per loop iteration
candidates=(
  "$STATIC/base.cubin|16"
  "$STATIC/spacing_stall7.cubin|16"
  "$STATIC/spacing_stall6.cubin|16"
  "$STATIC/fill100_stall8_yield0.cubin|31"
  "$STATIC/fill100_stall8_yield1.cubin|31"
  "$MIXED|31"
  "$STATIC/fill100_stall7_yield0.cubin|31"
  "$STATIC/fill100_stall7_yield1.cubin|31"
)

# Start with one warp per SM so warp rotation cannot hide per-warp eligibility.
# Then increase the grid warp population. Repeated equivalent warp-population points
# with different block shapes help expose block/occupancy artifacts.
geometries=(
  "32|1" "32|2" "32|4" "32|8" "32|16"
  "64|1" "64|2" "64|4" "64|8"
  "128|1" "128|2" "128|4" "128|8"
)

for c in "${candidates[@]}"; do
  IFS='|' read -r cubin opi <<<"$c"
  for g in "${geometries[@]}"; do
    IFS='|' read -r block bpsm <<<"$g"
    "$HERE/time_pure_issue_guarded.sh" "$cubin" "$opi" "$ITERS" "$block" "$bpsm" "$REPS" | tee -a "$LOG"
  done
done

python3 - "$LOG" "$OUT/summary.json" <<'PY'
import json,sys,statistics,pathlib
log,out=map(pathlib.Path,sys.argv[1:])
rows=[json.loads(x) for x in log.read_text().splitlines() if x.strip().startswith('{')]
by={}
for r in rows:
    name=pathlib.Path(r['cubin']).name
    by.setdefault(name,[]).append(r)
summary={
 'schema':'agillm.sm121.pure-issue-geometry-sweep.v1',
 'interpretation':{
   'single_warp_control':'block=32, blocks_per_sm=1 minimizes warp rotation; reduced-stall change here tests per-warp eligibility directly.',
   'population_sweep':'Increasing requested_grid_warps_per_sm tests whether any reduced-stall gain disappears as warp rotation/backend saturation takes over.',
   'throughput_rule':'Only measured OMMA/s and TFLOP/s count. Encoded stall values are not converted into a PFLOP/s prediction.'
 },
 'rows':rows,
 'by_candidate':by,
}
out.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(out)
PY

echo "GEOMETRY_SWEEP_PASS output=$OUT"
