#!/usr/bin/env bash
set -euo pipefail
# Bounded, trap-protected production -> silicon-benchmark -> production handoff.
# Requires explicit current-task approval and never kills/restarts production.
[[ ${SG_2PF_BENCH_HANDOFF:-} == approved ]] || {
  echo 'REFUSE: SG_2PF_BENCH_HANDOFF=approved is required' >&2; exit 50;
}
PROD=${1:?production PID required}; shift
(($#)) || { echo 'REFUSE: benchmark command required' >&2; exit 2; }
[[ $PROD =~ ^[0-9]+$ && -r /proc/$PROD/stat ]] || { echo "REFUSE: invalid production PID $PROD" >&2; exit 51; }
CMD=$(tr '\0' ' ' < /proc/$PROD/cmdline 2>/dev/null || true)
[[ $CMD == *agillm_gb10_1pf* && $CMD == *train* ]] || { echo "REFUSE: PID $PROD is not AGILLM-GB10-1PF trainer" >&2; exit 52; }
mapfile -t BEFORE < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed '/^$/d')
((${#BEFORE[@]} == 1)) && [[ ${BEFORE[0]} == "$PROD" ]] || {
  echo "REFUSE: pre-handoff CUDA census must contain exactly production PID $PROD; saw: ${BEFORE[*]:-none}" >&2; exit 53;
}
SAVE_DIR=${SG_2PF_PROD_SAVE_DIR:-/workspace/agillm-gb10-1pf-targetfix-active}
if find "$SAVE_DIR" -maxdepth 1 -type f -name '*.tmp' -print -quit 2>/dev/null | grep -q .; then
  echo 'REFUSE: checkpoint temporary file exists' >&2; exit 54
fi
if ls -l /proc/$PROD/fd 2>/dev/null | grep -F "$SAVE_DIR/" | grep -Eq '\.tmp|latest\.pt|checkpoint|\.pt '; then
  echo 'REFUSE: production currently has a checkpoint-like file open' >&2; exit 55
fi
LOCK=/tmp/sg_2pf_handoff.lock
exec 9>"$LOCK"
flock -n 9 || { echo 'REFUSE: another 2PF handoff is active' >&2; exit 56; }
RESUMED=0
resume_prod() {
  local rc=$?
  if [[ $RESUMED -eq 0 && -r /proc/$PROD/stat ]]; then
    kill -CONT "$PROD" 2>/dev/null || true
    RESUMED=1
  fi
  return "$rc"
}
trap resume_prod EXIT INT TERM HUP
kill -STOP "$PROD"
for _ in $(seq 1 50); do
  STAT=$(ps -o stat= -p "$PROD" 2>/dev/null | tr -d ' ' || true)
  [[ $STAT == *T* ]] && break
  sleep 0.05
done
[[ ${STAT:-} == *T* ]] || { echo "REFUSE: production did not enter stopped state (stat=${STAT:-gone})" >&2; exit 57; }
# Let any kernel submitted before SIGSTOP drain. Three quiet samples are enough
# for this micro-handoff; any other CUDA PID is an immediate abort.
quiet=0
for _ in $(seq 1 30); do
  mapfile -t NOW < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed '/^$/d')
  for p in "${NOW[@]}"; do [[ $p == "$PROD" ]] || { echo "REFUSE: competing CUDA PID $p appeared" >&2; exit 58; }; done
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 100)
  if [[ $util =~ ^[0-9]+$ ]] && ((util <= 5)); then ((quiet+=1)); else quiet=0; fi
  ((quiet >= 3)) && break
  sleep 0.1
done
((quiet >= 3)) || { echo 'REFUSE: GPU did not quiesce after stopping production' >&2; exit 59; }
export SG_2PF_BENCH_FENCE=approved
export SG_2PF_STOPPED_PROD_PID="$PROD"
"$@"
RC=$?
kill -CONT "$PROD" 2>/dev/null || true
RESUMED=1
trap - EXIT INT TERM HUP
exit "$RC"
