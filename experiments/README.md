# Experiment Scripts

These scripts run OEL Round 1 in two configurations on Sokoban for the paper §6.2 OEL ablation.

## Setup
- Docker: pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
- Hardware: 8× NVIDIA B200
- OEL repo at /root/oel/LMOps/oel inside container
- All ckpts written to /mnt/zhuhe/saves/oel/ (Azure blob mount)

## Pipeline scripts

### `oel_pipeline.sh` — original full pipeline (extract → deploy → consolidate)
Runs OEL paper's default 3-stage Round 1 with Qwen3-4B-Instruct-2507 self-rollout.
- Stage 1: extract experience from 10 ckpt seeds × 100 task validation
- Stage 2: deploy collect 100 trajectory steps
- Stage 3: FSDP train 100 steps with on-policy context distillation

### `oel_continue.sh` — resume Stage 2/3 from existing Stage 1 experience
Used when Stage 1 already done and we want to skip re-extraction.

### `oel_stage3_only.sh` — Stage 3 alone (re-train consolidation)
Restart consolidation from existing deploy_data + experience_list.

## Compression A experiments

### `oel_compA.sh` — full pipeline with 1.7B in stage 1+2 (WRONG, breaks on-policy)
First attempt: replaced both extract and deploy with 1.7B Qwen3-1.7B.
**This violated OEL's on-policy distillation requirement** (deploy trajectory must
come from the model being trained).

### `oel_compA_v2.sh` — corrected: 1.7B in stage 1 ONLY
Stage 1 uses 1.7B (cheap experience extraction; experience is text knowledge,
not policy-bound). Stage 2 reverts to 4B for on-policy compliance.
Stage 3 trains 4B as usual.

### `oel_base_rerun20.sh` — re-train base round 1 stage 3 to 20 steps
Used to get clean ckpt_20 after first base run OOM'd at step 10.

## Eval & monitoring

### `oel_eval.sh` — held-out 128 maps × 10 seeds eval
Eval a specific ckpt on Sokoban held-out set.

### `oel_monitor.sh` — every-60s status logger
Tracks: process count, GPU memory/util, deploy/consolidate log line counts.

### `oel_speed_monitor.sh` — step-time monitor with alert
Detects slow training (>90s/step → alert in log).

## Round 1 results

See `/mnt/zhuhe/saves/oel/raw_data/{base_round1,compA_round1}/RESULT.md` for held-out
acc per ckpt and detailed logs.

## Modifications to OEL upstream scripts

- `scripts/textgame_consolidate.sh`: toggle for `param_offload`, `optimizer_offload`,
  `enable_gradient_checkpointing`. We found `enable_gradient_checkpointing=False`
  triggers OOM in vLLM rollout (memory profiling expects clean state); keeping all
  three True is the safe config.
- `scripts/textgame_eval.sh`, `scripts/textgame_generate_deploy.sh`: redirect ckpt
  storage from `/tmp/` to `/mnt/zhuhe/saves/oel/` (large blob disk).
- All scripts: `trainer.logger=[console]` (drop wandb dependency).

## Known issues / improvements for next round

1. Stage 3 training is bottlenecked by `max_num_batched_tokens=20480` and
   `enforce_eager=True`. Bumping to 65536 + disabling enforce_eager should give
   ~2× speedup on B200.
2. `param_offload=True` is required to coexist FSDP train + vLLM rollout on same
   GPU; without it vLLM memory profiling fails.
3. Stage 3 OOM at step 10 when grad_ckpt=False; keep True.
