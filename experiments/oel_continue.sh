#!/bin/bash
LOGDIR=/tmp
SUMMARY=/tmp/oel_round1_summary.log
log_event() { echo "[$(date)] $1" >> $SUMMARY; }
log_event "=== continue from manual experience_list.txt ==="

# Stage 2: deploy collect (short version)
log_event "starting stage 2 (deploy short 20 step)"
docker exec oel bash -c "
cd /root/oel/LMOps/oel
source .venv/bin/activate
export WANDB_API_KEY=2060b5a86e2951e16b1b8b5f85bd1c1b99aa02f5
export WANDB_PROJECT=oel-codeagent-zhe
export MASTER_ADDR=127.0.0.1
export OMPI_COMM_WORLD_RANK=0
ray stop --force >/dev/null 2>&1
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 >/dev/null 2>&1
bash scripts/textgame_generate_deploy.sh \
  --model /mnt/zhuhe/models/Qwen3-4B-Instruct-2507 \
  --exp_name oel-sokoban-q3-4b-ins-round1-deploy \
  --nnodes 1 --oel_round 1 --experience_max_length 8192 \
  --textgame_name Sokoban-v0 --max_response_length 1024 \
  --textgame_max_steps 5 --textgame_no_think True \
  --total_training_steps 20 > /tmp/oel_deploy.log 2>&1
" && log_event "stage 2 done" || log_event "stage 2 FAILED"

# Stage 3: consolidate (short 20 step)
log_event "starting stage 3 (consolidate short 20 step)"
docker exec oel bash -c "
cd /root/oel/LMOps/oel
source .venv/bin/activate
export WANDB_API_KEY=2060b5a86e2951e16b1b8b5f85bd1c1b99aa02f5
export WANDB_PROJECT=oel-codeagent-zhe
export MASTER_ADDR=127.0.0.1
export OMPI_COMM_WORLD_RANK=0
bash scripts/textgame_consolidate.sh \
  --model /mnt/zhuhe/models/Qwen3-4B-Instruct-2507 \
  --exp_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1 \
  --nnodes 1 --oel_round 1 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 \
  --experience_max_length 8192 --textgame_name Sokoban-v0 \
  --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True \
  --deploy_save_dir /tmp/oel-sokoban-q3-4b-ins-round1-deploy/deploy_data \
  --exp_path /tmp/oel-sokoban-q3-4b-ins-ext-v4-selwp-round1/experience_list.txt \
  --total_training_steps 20 --save_freq 5 > /tmp/oel_consolidate.log 2>&1
" && log_event "stage 3 done" || log_event "stage 3 FAILED"

log_event "=== Round 1 base FULL DONE ==="
