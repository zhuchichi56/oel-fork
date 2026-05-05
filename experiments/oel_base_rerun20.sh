#!/bin/bash
# After compA finishes, retrain base round 1 stage 3 up to ckpt_20 (clean, no OOM)
SUMMARY=/tmp/oel_round1_summary.log
log_event() { echo "[$(date)] $1" >> $SUMMARY; }

echo "[$(date)] watcher: waiting for compA to finish..." >> $SUMMARY
while true; do
  if grep -q "Round 1 compA FULL DONE" $SUMMARY 2>/dev/null; then
    break
  fi
  sleep 60
done
log_event "compA done, starting base re-train (10 -> 20 steps)"

EXP_NAME_BASE=oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1-rerun20
MODEL_4B=/mnt/zhuhe/models/Qwen3-4B-Instruct-2507
EXP_LIST=/tmp/oel-sokoban-q3-4b-ins-ext-v4-selwp-round1/experience_list.txt
DEPLOY_DIR=/tmp/oel-sokoban-q3-4b-ins-round1-deploy/deploy_data
RAW=/mnt/zhuhe/saves/oel/raw_data/base_round1_rerun20
mkdir -p $RAW

# Re-train stage 3 to 20 step
log_event "base stage 3 re-train (target step 20)"
docker exec oel bash -c "
cd /root/oel/LMOps/oel
source .venv/bin/activate
export WANDB_API_KEY=2060b5a86e2951e16b1b8b5f85bd1c1b99aa02f5
export WANDB_PROJECT=oel-codeagent-zhe
export MASTER_ADDR=127.0.0.1
export OMPI_COMM_WORLD_RANK=0
ray stop --force >/dev/null 2>&1
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 >/dev/null 2>&1
bash scripts/textgame_consolidate.sh \
  --model $MODEL_4B \
  --exp_name $EXP_NAME_BASE \
  --nnodes 1 --oel_round 1 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 \
  --experience_max_length 8192 --textgame_name Sokoban-v0 \
  --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True \
  --deploy_save_dir $DEPLOY_DIR \
  --exp_path $EXP_LIST \
  --total_training_steps 20 --save_freq 5 > /tmp/oel_consolidate_base20.log 2>&1
" && log_event "base stage 3 (20 step) done" || log_event "base stage 3 (20 step) FAILED"

# Eval ckpt 20
log_event "base eval ckpt_20"
docker exec oel bash -c "
cd /root/oel/LMOps/oel
source .venv/bin/activate
export MASTER_ADDR=127.0.0.1
export OMPI_COMM_WORLD_RANK=0
ray stop --force >/dev/null 2>&1
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 >/dev/null 2>&1
bash scripts/textgame_eval_inturn.sh '$EXP_NAME_BASE,5,20,5,$MODEL_4B,false,1024,Sokoban-v0,5,true' > /tmp/oel_eval_base20.log 2>&1
" && log_event "base eval done" || log_event "base eval FAILED"

# Archive
docker exec oel cp /tmp/oel_consolidate_base20.log /tmp/oel_eval_base20.log $RAW/ 2>&1

ACC5=$(docker exec oel grep -aE "Held-out Envs Acc" /tmp/oel_eval_base20.log 2>/dev/null | head -1 | grep -oE "[0-9.]+$")
ACC10=$(docker exec oel grep -aE "Held-out Envs Acc" /tmp/oel_eval_base20.log 2>/dev/null | sed -n '2p' | grep -oE "[0-9.]+$")
ACC15=$(docker exec oel grep -aE "Held-out Envs Acc" /tmp/oel_eval_base20.log 2>/dev/null | sed -n '3p' | grep -oE "[0-9.]+$")
ACC20=$(docker exec oel grep -aE "Held-out Envs Acc" /tmp/oel_eval_base20.log 2>/dev/null | sed -n '4p' | grep -oE "[0-9.]+$")

cat > $RAW/RESULT.md <<RESULT_EOF
BASE_R1_RERUN20_RESULT:
  ckpt_5  acc = $ACC5
  ckpt_10 acc = $ACC10
  ckpt_15 acc = $ACC15
  ckpt_20 acc = $ACC20

stage1: reused base round 1 stage 1 (4B, 7 ckpt seeds, 100 sample/seed)
stage2: reused base round 1 stage 2 (4B, 20 deploy steps)
stage3: NEW: FSDP train 4B for full 20 steps with grad_ckpt+offload (avoid OOM)
eval: 4 checkpoints (5/10/15/20) on held-out 128 maps × 10 seeds
RESULT_EOF
log_event "FINAL base rerun20: ckpt_5=$ACC5 ckpt_10=$ACC10 ckpt_15=$ACC15 ckpt_20=$ACC20"
log_event "=== Round 1 base rerun20 FULL DONE ==="
