#!/bin/bash
# P3c GPU chain: wait for K=5 (already running) -> launch K=10 -> verify -> DONE marker.
# Launched detached (setsid) so it survives the launching shell's timeout.
# Safe: K=10 is launched ONLY if the K=5 store exists; otherwise FAILED marker.
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)" || exit 1
PY=/usr/bin/python3
LOG=logs/p3c_wrapper.log
DONE=logs/p3c_wrapper_DONE
mkdir -p logs
rm -f "$DONE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
log "wrapper started (pid $$)"

wait_done() { # $1 = the "--k N" arg to watch
  while pgrep -f "clip_finetune_kshot.py --ckpt dfn_clip --k $1" >/dev/null 2>&1; do
    sleep 60
  done
}

report() { # $1 = store path
  "$PY" - "$1" >> "$LOG" 2>&1 <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); s = d['stage2_final']
print(f"  {d.get('checkpoint')} K={d.get('K')}: {s['mean']*100:.2f}% +/- {s['std']*100:.2f}% "
      f"(n={s['n_seeds']}, monotonic={d.get('best_val_monotonic_nondecreasing')}, "
      f"selected={d.get('selected')})")
PYEOF
}

# ---- K=5 (already running) -----------------------------------------------
wait_done 5
log "K=5 process ended; checking store"
if [ -f data/p3c_clip_finetune_kshot_dfn.json ]; then
  report data/p3c_clip_finetune_kshot_dfn.json
  log "K=5 store OK; launching K=10 (detached)"
  setsid "$PY" scripts/clip_finetune_kshot.py --ckpt dfn_clip --k 10 \
    --out data/p3c_clip_finetune_kshot_dfn_k10.json </dev/null >logs/p3c_dfn_k10.log 2>&1 &
  disown
  wait_done 10
  log "K=10 process ended; checking store"
  if [ -f data/p3c_clip_finetune_kshot_dfn_k10.json ]; then
    report data/p3c_clip_finetune_kshot_dfn_k10.json
    echo DONE > "$DONE"
    log "WRAPPER DONE: both DFN K=5 and K=10 stores present"
  else
    echo FAILED_K10 > "$DONE"
    log "K=10 store MISSING (see logs/p3c_dfn_k10.log)"
  fi
else
  echo FAILED_K5 > "$DONE"
  log "K=5 store MISSING (K=10 NOT launched; see logs/p3c_dfn_k5.log)"
fi
