#!/usr/bin/env bash
set -euo pipefail
# Fail-closed launcher. This script never stops/signals another process.
# Caller must explicitly set SG_2PF_BENCH_FENCE=approved after SG coordination says the GPU window is safe.
if [[ ${SG_2PF_BENCH_FENCE:-} != approved ]]; then
  echo 'REFUSE: SG_2PF_BENCH_FENCE=approved is required' >&2; exit 40
fi
mapfile -t gpupids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed '/^$/d')
if ((${#gpupids[@]})); then
  echo "REFUSE: CUDA processes already active: ${gpupids[*]}" >&2; exit 41
fi
if pgrep -af 'agillm_gb10_1pf|rpv16_target|rpv16_ar_block4|rpv16_.*paired|targetfix.*train' >/tmp/2pf_guard_procs.$$ 2>/dev/null; then
  echo 'REFUSE: production/A-B process detected:' >&2; cat /tmp/2pf_guard_procs.$$ >&2; rm -f /tmp/2pf_guard_procs.$$; exit 42
fi
rm -f /tmp/2pf_guard_procs.$$
# Recheck immediately before exec to narrow TOCTOU. We do not claim this proves future exclusivity.
sleep 0.2
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo 'REFUSE: CUDA process appeared during final guard' >&2; exit 43
fi
exec "$@"
