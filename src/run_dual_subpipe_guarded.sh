#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CUBIN=${1:-/workspace/gb10_2pflop_openfork_20260916/static_recon/dual_subpipe_warp.cubin}
ITERS=${2:-700}
BPSM=${3:-8}
REPS=${4:-17}
BIN=${DUAL_SUBPIPE_RUNNER:-/workspace/gb10_2pflop_openfork_20260916/static_recon/run_dual_subpipe_probe}
exec "$HERE/run_2pf_guarded.sh" "$BIN" "$CUBIN" "$ITERS" "$BPSM" "$REPS"
