#!/bin/bash
LOG=/tmp/oel_monitor.log
> $LOG
while true; do
  TS=$(date +%H:%M:%S)
  PROC=$(docker exec oel pgrep -f "verl.trainer.main_ppo" 2>/dev/null | wc -l)
  GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {printf "%.1f", s/1024}')
  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1; c++} END {printf "%.0f", s/c}')
  D_LINES=$(docker exec oel wc -l /tmp/oel_deploy.log 2>/dev/null | awk '{print $1}')
  C_LINES=$(docker exec oel wc -l /tmp/oel_consolidate.log 2>/dev/null | awk '{print $1}')
  D_TAIL=$(docker exec oel tail -1 /tmp/oel_deploy.log 2>/dev/null | tr -d '\033' | sed 's/\[[0-9;]*m//g' | head -c 100)
  echo "[$TS] proc=$PROC gpu=${GPU_USED}GB util=${GPU_UTIL}% | deploy_lines=${D_LINES:-0} cons_lines=${C_LINES:-0}" >> $LOG
  echo "  last_deploy: ${D_TAIL}" >> $LOG
  sleep 60
done
