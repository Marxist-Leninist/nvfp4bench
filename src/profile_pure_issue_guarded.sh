#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CUBIN=${1:?candidate cubin required}
BPSM=${2:-8}
BLOCK=${3:-128}
ITERS=${4:-700}
BIN=${PURE_ISSUE_ONCE_BIN:-/workspace/gb10_2pflop_openfork_20260916/static_recon/pure_issue_profile_once}
METRICS='sm__inst_executed_pipe_tensor,sm__inst_issued,sm__issue_active,sm__pipe_tensor_cycles_active,sm__ops_path_tensor_src_fp4_dst_fp32'
# The existing guard is the authority: explicit fence + zero CUDA/prod processes.
exec "$HERE/run_2pf_guarded.sh" ncu --target-processes application-only --kernel-name pure_issue_witness --launch-count 1 --metrics "$METRICS" --csv "$BIN" "$CUBIN" "$BPSM" "$BLOCK" "$ITERS"
