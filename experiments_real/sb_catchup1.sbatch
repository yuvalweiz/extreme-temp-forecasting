#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-8:00:00
#SBATCH --job-name=cu_nnfmkt
#SBATCH --output=cu_nnfmkt-%j.out
#SBATCH --error=cu_nnfmkt-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Catch-up 1: finish nnf-mkt stage 2 (winner L3e-5/D256/DO0.1): a2b1 s333 + a3b15 + a1b0.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 CACHE=0 REGION=Center WINDOW=180
export FRAMEBANK=/home/weizyuv/expreal/framebank_nn FRAMEBANK_NS=/home/weizyuv/expreal/framebank_nn/norm_stats_x.npz
export HEAD=temporalfusion
run() { echo "=== nnfs2mkt_$1 s$5 $(date +%H:%M)"; RUN_TAG=nnfs2mkt_$1 TARGETS="t3,t5,t10,t15" ALPHA=$2 BETA=$3 SELECT=$4 SEED=$5 \
  LR_HEAD=3e-5 D_MODEL=256 DROPOUT=0.1 CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1; }
run a2b1 2 1 paper 333
for S in 111 222 333; do run a3b15 3 1.5 paper $S; done
for S in 111 222 333; do run a1b0 1 0 all $S; done
echo "=== catchup1 ALL DONE $(date)"
