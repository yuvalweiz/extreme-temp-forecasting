#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00
#SBATCH --job-name=p2nnf
#SBATCH --output=p2nnf-%j.out
#SBATCH --error=p2nnf-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# P2 sensitivity for the NN variant: regional finals + 444/555 at its P2 recipe.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 CACHE=0 WINDOW=180
export FRAMEBANK=/home/weizyuv/expreal/framebank_nn FRAMEBANK_NS=/home/weizyuv/expreal/framebank_nn/norm_stats_x.npz HEAD=temporalfusion
for R in Negev Northwest; do
  for S in 111 222 333 444 555; do
    echo "=== nnfP2FIN $R s$S $(date +%H:%M)"
    REGION=$R RUN_TAG=nnfP2FIN ALPHA=1 BETA=0 LR_HEAD=3e-5 D_MODEL=128 DROPOUT=0.1 SEED=$S \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
for S in 444 555; do
  REGION=Center RUN_TAG=nnfp2std_a1b0 ALPHA=1 BETA=0 LR_HEAD=3e-5 D_MODEL=128 DROPOUT=0.1 SEED=$S \
    CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
done
echo "=== p2 nnf ALL DONE $(date)"
