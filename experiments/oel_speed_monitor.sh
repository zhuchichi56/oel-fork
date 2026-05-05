#!/bin/bash
# OEL stage 3 active speed monitor
# Alerts if step time > 90s (expected ~30-60s with offload disabled)
LOG=/tmp/oel_speed_monitor.log
> $LOG
PREV_STEP=0
PREV_TIME=$(date +%s)
echo "[$(date)] speed monitor started" >> $LOG
while true; do
  TS=$(date +%s)
  CUR_STEP=$(docker exec oel grep -aoE "training/global_step:[0-9]+" /tmp/oel_consolidate.log 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  CUR_STEP=${CUR_STEP:-0}
  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1; c++} END {printf "%.0f", s/c}')
  GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s/1024}')

  if [ "$CUR_STEP" -gt "$PREV_STEP" ] 2>/dev/null; then
    DELTA_STEPS=$((CUR_STEP - PREV_STEP))
    DELTA_TIME=$((TS - PREV_TIME))
    PER_STEP=$((DELTA_TIME / DELTA_STEPS))
    HUMAN=$(date +%H:%M:%S)
    echo "[$HUMAN] step=$CUR_STEP gpu=${GPU_MEM}GB util=${GPU_UTIL}% | $PER_STEP s/step (delta $DELTA_STEPS in ${DELTA_TIME}s)" >> $LOG
    if [ "$PER_STEP" -gt "90" ] 2>/dev/null; then
      echo "[$HUMAN] ⚠️  ALERT: step time $PER_STEP s > 90s threshold" >> $LOG
    fi
    PREV_STEP=$CUR_STEP
    PREV_TIME=$TS
  else
    HUMAN=$(date +%H:%M:%S)
    ELAPSED=$((TS - PREV_TIME))
    echo "[$HUMAN] step=$CUR_STEP (no progress for ${ELAPSED}s) gpu=${GPU_MEM}GB util=${GPU_UTIL}%" >> $LOG
  fi

  # check if done
  if grep -q "stage 3 done\|stage 3 FAILED" /tmp/oel_round1_summary.log 2>/dev/null; then
    LAST=$(tail -1 /tmp/oel_round1_summary.log)
    if echo "$LAST" | grep -q "stage 3"; then
      echo "[$(date +%H:%M:%S)] === STAGE 3 EXIT: $LAST ===" >> $LOG
      break
    fi
  fi
  sleep 60
done
