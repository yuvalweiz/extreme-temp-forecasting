#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-12:00:00
#SBATCH --job-name=winreg
#SBATCH --output=winreg-%j.out
#SBATCH --error=winreg-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Regional winter tuned runs (a1b0 winter picks at winter grid winners): ours/stf/stl,
# Negev+Northwest x 3 seeds — completes the winter deployment-selection story regionally.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 CACHE=1 COLD=1 K=1
for R in Negev Northwest; do
  for SEED in 111 222 333; do
    DATASET_DIR=/home/weizyuv/stationvec_MIN_$R PAPERW=/home/weizyuv/interp_weights/paper_opt.npz HEAD=temporalfusion REGION=$R RUN_TAG=oursWINFIN \
      ALPHA=1 BETA=0 LR_HEAD=3e-4 D_MODEL=256 DROPOUT=0.3 SEED=$SEED CKPT_ROOT=/home/weizyuv/expreal/${R}_cold $PY -u -m pipeline.train 2>&1 | tail -1
    STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_MIN_$R HEAD=temporalfusion REGION=$R RUN_TAG=stfWINFIN \
      ALPHA=1 BETA=0 LR_HEAD=3e-5 D_MODEL=256 DROPOUT=0.1 SEED=$SEED CKPT_ROOT=/home/weizyuv/expreal/${R}_cold $PY -u -m pipeline.train 2>&1 | tail -1
    STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_MIN_$R HEAD=lstm REGION=$R RUN_TAG=stlWINFIN \
      ALPHA=1 BETA=0 LR_HEAD=1e-4 D_MODEL=256 DROPOUT=0.1 SEED=$SEED CKPT_ROOT=/home/weizyuv/expreal/${R}_cold $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== winreg ALL DONE $(date)"
