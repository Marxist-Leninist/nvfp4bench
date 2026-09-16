#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CUBIN=${1:?candidate cubin required}
OMMA_PER_WARP_ITER=${2:?OMMA count per warp per loop iteration required}
ITERS=${3:-700}
BLOCK=${4:-128}
BPSM=${5:-8}
REPS=${6:-17}
BIN=${PURE_ISSUE_TIMING_BIN:-/workspace/gb10_2pflop_openfork_20260916/static_recon/pure_issue_timing}
exec "$HERE/run_2pf_guarded.sh" "$BIN" "$CUBIN" "$OMMA_PER_WARP_ITER" "$ITERS" "$BLOCK" "$BPSM" "$REPS"
