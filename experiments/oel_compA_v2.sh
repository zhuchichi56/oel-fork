#!/bin/bash
# compA: ONLY stage 2 + 3 + 4 (use existing 1.7B stage 1 experience)
# CRITICAL: stage 2 MUST use 4B for on-policy distillation
SUMMARY=/tmp/oel_compA_summary.log
log_event() { echo "[$(date)] $1" >> $SUMMARY; }
log_event "=== compA RESTART: stage 2/3/4 with 4B (on-policy fix) ==="

EXP_NAME_DEPLOY=oel-sokoban-q3-4b-ins-with-1b7exp-deploy
EXP_NAME_CONSOLIDATE=oel-sokoban-q3-4b-ins-with-1b7exp-r1
MODEL_4B=/mnt/zhuhe/models/Qwen3-4B-Instruct-2507
EXP_LIST=/tmp/oel-sokoban-q3-1b7-ext-v4-selwp-round1/experience_list.txt
RAW=/mnt/zhuhe/saves/oel/raw_data/compA_round1
mkdir -p $RAW

log_event "exp_list: $EXP_LIST"
docker exec oel cat $EXP_LIST 2>&1 >> $SUMMARY

# ----- Stage 2: deploy with 4B -----
log_event "stage 2 (deploy with 4B — on-policy)"
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
  --model $MODEL_4B \
  --exp_name $EXP_NAME_DEPLOY \
  --nnodes 1 --oel_round 1 --experience_max_length 8192 \
  --textgame_name Sokoban-v0 --max_response_length 1024 \
  --textgame_max_steps 5 --textgame_no_think True \
  --total_training_steps 20 > /tmp/oel_deploy_compA.log 2>&1
" && log_event "stage 2 done" || log_event "stage 2 FAILED"

# ----- Stage 3: consolidate trains 4B -----
log_event "stage 3 (consolidate trains 4B with 1.7B-extracted experience + 4B trajectory)"
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
  --exp_name $EXP_NAME_CONSOLIDATE \
  --nnodes 1 --oel_round 1 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 \
  --experience_max_length 8192 --textgame_name Sokoban-v0 \
  --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True \
  --deploy_save_dir /tmp/$EXP_NAME_DEPLOY/deploy_data \
  --exp_path $EXP_LIST \
  --total_training_steps 10 --save_freq 5 > /tmp/oel_consolidate_compA.log 2>&1
" && log_event "stage 3 done" || log_event "stage 3 FAILED"

# ----- Stage 4: eval -----
log_event "stage 4 eval"
docker exec oel bash -c "
cd /root/oel/LMOps/oel
source .venv/bin/activate
export MASTER_ADDR=127.0.0.1
export OMPI_COMM_WORLD_RANK=0
ray stop --force >/dev/null 2>&1
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 >/dev/null 2>&1
bash scripts/textgame_eval_inturn.sh '$EXP_NAME_CONSOLIDATE,10,10,5,$MODEL_4B,false,1024,Sokoban-v0,5,true' > /tmp/oel_eval_compA.log 2>&1
" && log_event "eval done" || log_event "eval FAILED"

# Archive raw data
docker exec oel cp /tmp/oel_deploy_compA.log /tmp/oel_consolidate_compA.log /tmp/oel_eval_compA.log $RAW/ 2>&1
docker exec oel cp -r /tmp/oel-sokoban-q3-1b7-ext-v4-selwp-round1 $RAW/extract_1b7/ 2>&1
docker exec oel cp -r /tmp/$EXP_NAME_DEPLOY $RAW/deploy_4b/ 2>&1

ACC=$(docker exec oel grep -aE "Held-out Envs Acc" /tmp/oel_eval_compA.log 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
log_event "FINAL: compA ckpt_10 held-out acc = ${ACC:-?}"
cat > $RAW/RESULT.md <<RESULT_EOF
COMPA_R1_RESULT: ckpt_10 held-out acc = $ACC

stage1: 1.7B Qwen3-1.7B (think mode), single ckpt seed (50), val_samples_limit=50
        avg held-out acc during extract: ~20% (vs 4B base ~10.5%)
        1.7B response avg 3 token, traj avg 10 token (4B is 600/2000)
stage2: 4B Qwen3-4B-Instruct-2507 (kept 4B for on-policy distillation), total_training_steps=20
stage3: FSDP train 4B for 10 steps with 1.7B-extracted experience + 4B-collected trajectory
        ckpt_5 + ckpt_10 saved
eval: ckpt_10 on held-out 128 maps × 10 seeds = $ACC
RESULT_EOF
log_event "=== Round 1 compA FULL DONE ==="
