#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-6:00:00
#SBATCH --job-name=oursFINmk4
#SBATCH --output=oursFINmk4-%j.out
#SBATCH --error=oursFINmk4-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Our mk4-format regional finals: Center pick = tuned (L3e-4/D128/DO0.1) + plain
# objective; Negev + Northwest, 3 seeds.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 CACHE=1 HEAD=temporalfusion
for R in Negev Northwest; do
  for SEED in 111 222 333; do
    echo "=== oursFIN_mk4 $R s$SEED $(date +%H:%M)"
    DATASET_DIR=/home/weizyuv/stationvec_$R PAPERW=/home/weizyuv/interp_weights/paper_opt.npz \
      REGION=$R RUN_TAG=oursFIN_mk4 TARGETS="1,t5,t10,t15" ALPHA=1 BETA=0 \
      LR_HEAD=3e-4 D_MODEL=128 DROPOUT=0.1 SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== ours mk4 finals ALL DONE $(date)"
