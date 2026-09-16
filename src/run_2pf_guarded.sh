#!/usr/bin/env bash
set -euo pipefail
# Fail-closed launcher for GB10 silicon work.
# Default mode still requires a totally empty GPU.
# Explicit handoff mode permits exactly one *already stopped* production trainer
# PID to retain its CUDA context while the benchmark runs. This script never
# sends STOP/CONT itself; src/run_2pf_handoff.sh owns that bounded transition.
if [[ ${SG_2PF_BENCH_FENCE:-} != approved ]]; then
  echo 'REFUSE: SG_2PF_BENCH_FENCE=approved is required' >&2; exit 40
fi
ALLOW=${SG_2PF_STOPPED_PROD_PID:-}

verify_allowed_stopped_prod() {
  [[ $ALLOW =~ ^[0-9]+$ ]] || { echo 'REFUSE: SG_2PF_STOPPED_PROD_PID must be numeric' >&2; exit 44; }
  [[ -r /proc/$ALLOW/stat ]] || { echo "REFUSE: allowed production PID $ALLOW is gone" >&2; exit 45; }
  local stat cmd
  stat=$(ps -o stat= -p "$ALLOW" 2>/dev/null | tr -d ' ')
  [[ $stat == *T* ]] || { echo "REFUSE: allowed production PID $ALLOW is not stopped (stat=$stat)" >&2; exit 46; }
  cmd=$(tr '\0' ' ' < /proc/$ALLOW/cmdline 2>/dev/null || true)
  [[ $cmd == *agillm_gb10_1pf* && $cmd == *train* ]] || {
    echo "REFUSE: PID $ALLOW does not look like the expected AGILLM-GB10-1PF trainer" >&2; exit 47;
  }
}

census_cuda() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed '/^$/d' || true
}

check_cuda_scope() {
  mapfile -t gpupids < <(census_cuda)
  if [[ -z $ALLOW ]]; then
    if ((${#gpupids[@]})); then
      echo "REFUSE: CUDA processes already active: ${gpupids[*]}" >&2; exit 41
    fi
  else
    verify_allowed_stopped_prod
    local p
    for p in "${gpupids[@]}"; do
      [[ $p == "$ALLOW" ]] || {
        echo "REFUSE: CUDA PID $p overlaps stopped production PID $ALLOW" >&2; exit 41;
      }
    done
  fi
}

check_cuda_scope
if [[ -z $ALLOW ]]; then
  if pgrep -af 'agillm_gb10_1pf|rpv16_target|rpv16_ar_block4|rpv16_.*paired|targetfix.*train' >/tmp/2pf_guard_procs.$$ 2>/dev/null; then
    echo 'REFUSE: production/A-B process detected:' >&2; cat /tmp/2pf_guard_procs.$$ >&2; rm -f /tmp/2pf_guard_procs.$$; exit 42
  fi
else
  # The production wrapper is expected to remain alive around its stopped child.
  # Still reject known A/B harnesses that could race to CUDA after our census.
  if pgrep -af 'rpv16_ar_block4|rpv16_.*paired|rpv16_pair2_local_ab|rpv16_ar_blockN_ab' >/tmp/2pf_guard_procs.$$ 2>/dev/null; then
    echo 'REFUSE: competing RPV A/B process detected during handoff:' >&2
    cat /tmp/2pf_guard_procs.$$ >&2; rm -f /tmp/2pf_guard_procs.$$; exit 42
  fi
fi
rm -f /tmp/2pf_guard_procs.$$
# Recheck immediately before exec to narrow TOCTOU. A stale coordination note
# is intentionally irrelevant here; current process state is the authority.
sleep 0.2
check_cuda_scope
exec "$@"
