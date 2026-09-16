#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
STATIC=/workspace/gb10_2pflop_openfork_20260916/static_recon/control_probe2/pure_issue_witness
GUARD="$HERE/run_2pf_guarded.sh"
BASE="$STATIC/base.cubin"
CAND=${1:?candidate cubin required}
ITERS=${2:-8}
shift 2 || true
if [ "$#" -ne 16 ]; then
  echo "need 16 accumulator multiplicities" >&2; exit 2
fi
exec "$GUARD" "$STATIC/witness_runner" "$BASE" "$CAND" "$ITERS" "$@"
