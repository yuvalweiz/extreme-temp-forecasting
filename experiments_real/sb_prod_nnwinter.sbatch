#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-8:00:00
#SBATCH --job-name=pr_nnwin
#SBATCH --output=pr_nnwin-%j.out
#SBATCH --error=pr_nnwin-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# PRODUCTION: NN-frames WINTER (current outputs, deployed cold recipe), 3 regions x 3 seeds.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 CACHE=0 COLD=1 K=1 WINDOW=180
export FRAMEBANK=/home/weizyuv/expreal/framebank_nn FRAMEBANK_NS=/home/weizyuv/expreal/framebank_nn/norm_stats_x.npz
export HEAD=temporalfusion
for R in Center Negev Northwest; do
  for SEED in 111 222 333; do
    echo "=== nnfwin $R s$SEED $(date +%H:%M)"
    DATASET_DIR=/home/weizyuv/dsfullmin_$R REGION=$R RUN_TAG=nnfwin ALPHA=2 BETA=1 SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_cold $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== nn winter ALL DONE $(date)"
