#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-6:00:00
#SBATCH --job-name=std_finals
#SBATCH --output=std_finals-%j.out
#SBATCH --error=std_finals-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Battlefield-A regional finals: the TFT tabulars' Center-picked recipes (both =
# tuned hparams + plain objective) on Negev + Northwest, 3 seeds, CURRENT outputs.
# Grid winners: stf std = L3e-4 D128 DO0.3 | atft std = L1e-4 D256 DO0.3  (from Center_tune)
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 CACHE=1
for R in Negev Northwest; do
  for SEED in 111 222 333; do
    echo "=== stfSTDFIN $R s$SEED $(date +%H:%M)"
    STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_$R HEAD=temporalfusion REGION=$R RUN_TAG=stfSTDFIN \
      ALPHA=1 BETA=0 LR_HEAD=3e-4 D_MODEL=128 DROPOUT=0.3 SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
    echo "=== atftSTDFIN $R s$SEED $(date +%H:%M)"
    TABULAR=1 HEAD=temporalfusion REGION=$R RUN_TAG=atftSTDFIN \
      ALPHA=1 BETA=0 LR_HEAD=1e-4 D_MODEL=256 DROPOUT=0.3 SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== std finals ALL DONE $(date)"
