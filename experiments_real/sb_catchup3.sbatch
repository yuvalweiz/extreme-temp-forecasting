#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-4:00:00
#SBATCH --job-name=cu_atftfin
#SBATCH --output=cu_atftfin-%j.out
#SBATCH --error=cu_atftfin-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Catch-up 3: atft mkt finals (its pick = tuned L3e-4/D256/DO0.3 + plain), Negev+NW.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 CACHE=1 TABULAR=1 HEAD=temporalfusion
for R in Negev Northwest; do
  for SEED in 111 222 333; do
    echo "=== atftFIN_mkt $R s$SEED $(date +%H:%M)"
    REGION=$R RUN_TAG=atftFIN_mkt TARGETS="t3,t5,t10,t15" ALPHA=1 BETA=0 \
      LR_HEAD=3e-4 D_MODEL=256 DROPOUT=0.3 SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== catchup3 ALL DONE $(date)"
