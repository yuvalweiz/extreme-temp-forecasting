#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-6:00:00
#SBATCH --job-name=pv_tab
#SBATCH --output=pv_tab-%j.out
#SBATCH --error=pv_tab-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# FULLY-PLAIN protocol completion, tabular cells whose plain-val grid winner differs:
# stf/std L3e-5 D256 DO0.1 | atft/std L3e-5 D256 DO0.3 | atft/mkt L3e-5 D128 DO0.3 | stl/std L1e-4 D256 DO0.3
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 CACHE=1 REGION=Center
export CKPT_ROOT=/home/weizyuv/expreal/Center_tune
for SEED in 111 222 333; do
  STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_Center HEAD=temporalfusion RUN_TAG=stfp2std_a1b0 \
    ALPHA=1 BETA=0 LR_HEAD=3e-5 D_MODEL=256 DROPOUT=0.1 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
  TABULAR=1 HEAD=temporalfusion RUN_TAG=atftp2std_a1b0 \
    ALPHA=1 BETA=0 LR_HEAD=3e-5 D_MODEL=256 DROPOUT=0.3 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
  TABULAR=1 HEAD=temporalfusion RUN_TAG=atftp2mkt_a1b0 TARGETS="t3,t5,t10,t15" \
    ALPHA=1 BETA=0 LR_HEAD=3e-5 D_MODEL=128 DROPOUT=0.3 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
  STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_Center HEAD=lstm RUN_TAG=stlp2std_a1b0 \
    ALPHA=1 BETA=0 LR_HEAD=1e-4 D_MODEL=256 DROPOUT=0.3 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
done
echo "=== plainval tab ALL DONE $(date)"
