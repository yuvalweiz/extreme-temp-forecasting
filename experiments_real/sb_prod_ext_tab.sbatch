#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-8:00:00
#SBATCH --job-name=pr_exttab
#SBATCH --output=pr_exttab-%j.out
#SBATCH --error=pr_exttab-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# PRODUCTION: seeds 444/555 for featured tabular rows at their val picks (std outputs).
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 CACHE=1
for SEED in 444 555; do
  STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_Center HEAD=temporalfusion REGION=Center RUN_TAG=stfs2std_a1b0 \
    ALPHA=1 BETA=0 LR_HEAD=3e-4 D_MODEL=128 DROPOUT=0.3 SEED=$SEED CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
  TABULAR=1 HEAD=temporalfusion REGION=Center RUN_TAG=atfts2std_a1b0 \
    ALPHA=1 BETA=0 LR_HEAD=1e-4 D_MODEL=256 DROPOUT=0.3 SEED=$SEED CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
  STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_Center HEAD=lstm REGION=Center RUN_TAG=stls2std_a1b0 \
    ALPHA=1 BETA=0 LR_HEAD=3e-4 D_MODEL=256 DROPOUT=0.1 SEED=$SEED CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
  for R in Negev Northwest; do
    STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_$R HEAD=temporalfusion REGION=$R RUN_TAG=stfSTDFIN \
      ALPHA=1 BETA=0 LR_HEAD=3e-4 D_MODEL=128 DROPOUT=0.3 SEED=$SEED CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
    TABULAR=1 HEAD=temporalfusion REGION=$R RUN_TAG=atftSTDFIN \
      ALPHA=1 BETA=0 LR_HEAD=1e-4 D_MODEL=256 DROPOUT=0.3 SEED=$SEED CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
    STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_$R HEAD=lstm REGION=$R RUN_TAG=stlFIN_std \
      ALPHA=1 BETA=0 LR_HEAD=3e-4 D_MODEL=256 DROPOUT=0.1 SEED=$SEED CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== prod tab extensions ALL DONE $(date)"
