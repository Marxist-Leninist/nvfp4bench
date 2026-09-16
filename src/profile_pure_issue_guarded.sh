#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CUBIN=${1:?candidate cubin required}
BPSM=${2:-8}
BLOCK=${3:-128}
ITERS=${4:-700}
OUT=${5:-/workspace/gb10_2pflop_openfork_20260916/static_recon/pure_issue_ncu.csv}
BIN=${PURE_ISSUE_ONCE_BIN:-/workspace/gb10_2pflop_openfork_20260916/static_recon/pure_issue_profile_once}

# These are the SM121/GB20b counters that directly corroborate the OMMA-only witness.
# The timing claim must still come from the uninstrumented CUDA-event runner.
METRICS='sm__inst_executed_pipe_tensor_subpipe_hmma.sum,sm__ops_path_tensor_src_fp4_dst_fp32_sparsity_on.sum,sm__pipe_tensor_subpipe_hmma_cycles_active.sum,sm__inst_issued.sum,sm__issue_active.sum,sm__cycles_elapsed.avg,gpu__time_duration.sum'

# Fail explicitly if the host driver refuses performance counters. Never interpret an
# empty Nsight result as zero OMMA execution.
probe=$(mktemp)
trap 'rm -f "$probe"' EXIT
ncu --query-metrics-mode all --devices 0 >"$probe" 2>&1 || true
if grep -q 'ERR_NVGPUCTRPERM' "$probe"; then
  echo 'REFUSE_NCU: NVIDIA performance counters are restricted by the host (ERR_NVGPUCTRPERM).' >&2
  echo 'Use the arithmetic witness + time_pure_issue_guarded.sh proof path on this host.' >&2
  exit 45
fi
for m in ${METRICS//,/ }; do
  if ! grep -q "^${m}[[:space:]]" "$probe"; then
    echo "REFUSE_NCU: required metric is not collectable on device 0: $m" >&2
    exit 46
  fi
done

exec "$HERE/run_2pf_guarded.sh" \
  ncu --target-processes application-only --clock-control none \
      --kernel-name pure_issue_witness --launch-count 1 \
      --metrics "$METRICS" --csv --log-file "$OUT" \
      "$BIN" "$CUBIN" "$BPSM" "$BLOCK" "$ITERS"
