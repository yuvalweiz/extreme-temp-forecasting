#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-8:00:00
#SBATCH --job-name=winfix
#SBATCH --output=winfix-%j.out
#SBATCH --error=winfix-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Winter fairness stage-2 repair (winners verified from metas, mae_cold_all currency):
# stf L3e-5/D256/DO0.1 | stl L1e-4/D256/DO0.1 | ours L3e-4/D256/DO0.3
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 REGION=Center COLD=1 K=1 CACHE=1
export CKPT_ROOT=/home/weizyuv/expreal/Center_coldtune
for SPEC in "a2b1:2:1:paper" "a1b0:1:0:all"; do
  IFS=: read ABTAG A B SEL <<< "$SPEC"
  for SEED in 111 222 333; do
    STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_MIN_Center HEAD=temporalfusion RUN_TAG=stfs2win_${ABTAG} \
      ALPHA=$A BETA=$B SELECT=$SEL LR_HEAD=3e-5 D_MODEL=256 DROPOUT=0.1 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
    STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_MIN_Center HEAD=lstm RUN_TAG=stls2win_${ABTAG} \
      ALPHA=$A BETA=$B SELECT=$SEL LR_HEAD=1e-4 D_MODEL=256 DROPOUT=0.1 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
    DATASET_DIR=/home/weizyuv/stationvec_MIN_Center PAPERW=/home/weizyuv/interp_weights/paper_opt.npz HEAD=temporalfusion RUN_TAG=ourss2win_${ABTAG} \
      ALPHA=$A BETA=$B SELECT=$SEL LR_HEAD=3e-4 D_MODEL=256 DROPOUT=0.3 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== winfix ALL DONE $(date)"
