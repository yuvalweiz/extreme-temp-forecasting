#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-6:00:00
#SBATCH --job-name=stf_finals
#SBATCH --output=stf_finals-%j.out
#SBATCH --error=stf_finals-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Region finals for stf's FIXED Center pick on mkt (tuned hparams L3e-4/D256/DO0.1 +
# plain objective): Negev + Northwest, 3 seeds. (Its mk4 pick = default+plain, whose
# Negev/NW runs already exist from wave 2.)
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 CACHE=1
export STATIONVEC=1 HEAD=temporalfusion
for R in Negev Northwest; do
  for SEED in 111 222 333; do
    echo "=== stfFIN_mkt $R s$SEED $(date +%H:%M)"
    DATASET_DIR=/home/weizyuv/stationvec_$R REGION=$R RUN_TAG=stfFIN_mkt TARGETS="t3,t5,t10,t15" \
      ALPHA=1 BETA=0 LR_HEAD=3e-4 D_MODEL=256 DROPOUT=0.1 SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== stf finals ALL DONE $(date)"
