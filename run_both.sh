#!/usr/bin/env bash
# Sequential runner: LP++ first (creates laion feature cache), then CoOp official.
# Avoids a race on cache creation and GPU contention between the two.
set -uo pipefail
cd /home/lizhuo/Work/TobAI/Paper2-CLIPvsResNet
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs
mkdir -p "$LOGDIR"

echo "[$(date)] START lp_plus_plus.py"
python3 scripts/lp_plus_plus.py > "$LOGDIR/lp_plus_plus_$TS.log" 2>&1
RC1=$?
echo "[$(date)] lp_plus_plus.py exit=$RC1"

echo "[$(date)] START coop_official.py"
python3 scripts/coop_official.py > "$LOGDIR/coop_official_$TS.log" 2>&1
RC2=$?
echo "[$(date)] coop_official.py exit=$RC2"

echo "[$(date)] BOTH DONE rc_lp=$RC1 rc_coop=$RC2"
