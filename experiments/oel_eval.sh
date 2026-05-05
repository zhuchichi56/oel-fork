#!/bin/bash
SUMMARY=/tmp/oel_round1_summary.log
log_event() { echo "[$(date)] $1" >> $SUMMARY; }
log_event "=== eval base ckpt_10 ==="

docker exec oel bash -c "
cd /root/oel/LMOps/oel
source .venv/bin/activate
export WANDB_API_KEY=2060b5a86e2951e16b1b8b5f85bd1c1b99aa02f5
export WANDB_PROJECT=oel-codeagent-zhe
export MASTER_ADDR=127.0.0.1
export OMPI_COMM_WORLD_RANK=0
ray stop --force >/dev/null 2>&1
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 >/dev/null 2>&1
bash scripts/textgame_eval_inturn.sh 'oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1,10,10,5,/mnt/zhuhe/models/Qwen3-4B-Instruct-2507,false,1024,Sokoban-v0,5,true' > /tmp/oel_eval.log 2>&1
" && log_event "eval done" || log_event "eval FAILED"
log_event "=== eval ckpt_10 result above ==="
