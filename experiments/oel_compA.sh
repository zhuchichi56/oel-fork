#!/bin/bash
# OEL Round 1 (compression A): 1.7B replace 4B in stage 1+2; stage 3 still trains 4B
# i.e. 1.7B does rollout in extract+deploy, 4B is the trainable actor in consolidate
SUMMARY=/tmp/oel_compA_summary.log
log_event() { echo "[$(date)] $1" >> $SUMMARY; }
> $SUMMARY
log_event "=== compA Round 1: 1.7B explorer + 4B trainable ==="

EXP_NAME_EXTRACT=oel-sokoban-q3-1b7-ext-v4-selwp-round1
EXP_NAME_DEPLOY=oel-sokoban-q3-1b7-round1-deploy
EXP_NAME_CONSOLIDATE=oel-sokoban-q3-1b7-v4-selwp-lr1e-6-round1
MODEL_4B=/mnt/zhuhe/models/Qwen3-4B-Instruct-2507
MODEL_1B7=/mnt/zhuhe/models/Qwen3-1.7B

# Snapshot raw data dir
RAW=/mnt/zhuhe/saves/oel/raw_data/compA_round1
mkdir -p $RAW
log_event "raw_data dir: $RAW"

# ----- Stage 1: extract with 1.7B -----
# Use thinking version (Qwen3-1.7B is a thinking model, NOT no-think); but OEL paper used Qwen3-1.7B with think on FrozenLake.
# Following OEL paper FrozenLake config: prompt v3, no_think=False
log_event "stage 1 (extract with 1.7B)"
docker exec oel bash -c "
cd /root/oel/LMOps/oel
source .venv/bin/activate
export WANDB_API_KEY=2060b5a86e2951e16b1b8b5f85bd1c1b99aa02f5
export WANDB_PROJECT=oel-codeagent-zhe
export MASTER_ADDR=127.0.0.1
export OMPI_COMM_WORLD_RANK=0
ray stop --force >/dev/null 2>&1
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 >/dev/null 2>&1
# Single ckpt seed (50) only — fast version. val_samples_limit=50.
bash scripts/textgame_extract_inturn.sh '$EXP_NAME_EXTRACT,50,50,50,$MODEL_1B7,,v4,50,True,8192,Sokoban-v0,1024,5,True,1,' > /tmp/oel_extract_compA.log 2>&1
" && log_event "stage 1 done" || log_event "stage 1 FAILED"

# Build experience list manually (single seed)
docker exec oel bash -c "ls /tmp/$EXP_NAME_EXTRACT/global_step_50/extract_50_samples/experiences/ 2>/dev/null | sort -V | tail -1 | xargs -I{} echo /tmp/$EXP_NAME_EXTRACT/global_step_50/extract_50_samples/experiences/{} > /tmp/$EXP_NAME_EXTRACT/experience_list.txt"
docker exec oel cat /tmp/$EXP_NAME_EXTRACT/experience_list.txt 2>&1 >> $SUMMARY
log_event "experience_list built"

# ----- Stage 2: deploy with 4B (CRITICAL: must be trainable model for on-policy distillation) -----
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

# ----- Stage 3: consolidate trains 4B (CRITICAL: model is 4B, not 1.7B) -----
log_event "stage 3 (consolidate trains 4B with 1.7B-collected data)"
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
  --exp_path /tmp/$EXP_NAME_EXTRACT/experience_list.txt \
  --total_training_steps 10 --save_freq 5 > /tmp/oel_consolidate_compA.log 2>&1
" && log_event "stage 3 done" || log_event "stage 3 FAILED"

# ----- Stage 4: eval ckpt_10 -----
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
docker exec oel cp /tmp/oel_extract_compA.log /tmp/oel_deploy_compA.log /tmp/oel_consolidate_compA.log /tmp/oel_eval_compA.log $RAW/ 2>&1
docker exec oel cp -r /tmp/$EXP_NAME_EXTRACT $RAW/extract/ 2>&1
docker exec oel cp -r /tmp/$EXP_NAME_DEPLOY $RAW/deploy/ 2>&1

# Extract held-out acc
ACC=$(docker exec oel grep -aE "Held-out Envs Acc" /tmp/oel_eval_compA.log 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
log_event "FINAL: compA ckpt_10 held-out acc = ${ACC:-?}"
echo "COMPA_R1_RESULT: ckpt_10 held-out acc = $ACC" > $RAW/RESULT.md
echo "stage1: 1.7B Qwen3-1.7B, single ckpt seed (50), val_samples_limit=50" >> $RAW/RESULT.md
echo "stage2: 4B self-rollout (kept 4B for on-policy distillation), total_training_steps=20" >> $RAW/RESULT.md
echo "stage3: FSDP train 4B for 10 steps with 1.7B-collected data, ckpt_10 saved" >> $RAW/RESULT.md
echo "eval: ckpt_10 on held-out 128 maps × 10 seeds = $ACC" >> $RAW/RESULT.md
log_event "=== Round 1 compA FULL DONE ==="
