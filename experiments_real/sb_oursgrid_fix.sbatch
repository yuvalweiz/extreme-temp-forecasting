#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-6:00:00
#SBATCH --job-name=ogridfix
#SBATCH --output=ogridfix-%j.out
#SBATCH --error=ogridfix-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Complete the ours winter grid (skips existing configs).
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 REGION=Center COLD=1 K=1 CACHE=1
export DATASET_DIR=/home/weizyuv/stationvec_MIN_Center PAPERW=/home/weizyuv/interp_weights/paper_opt.npz HEAD=temporalfusion
for LR in 3e-4 1e-4 3e-5; do
  for DM in 128 256; do
    for DO in 0.1 0.3; do
      TAG="ourstgwinL${LR}D${DM}O${DO}"; TAG=${TAG//./p}; TAG=${TAG//-/}
      ls -d /home/weizyuv/expreal/Center_coldtune/${TAG}__* >/dev/null 2>&1 && { echo "skip $TAG"; continue; }
      echo "=== $TAG $(date +%H:%M)"
      RUN_TAG=$TAG ALPHA=2 BETA=1 SELECT=paper LR_HEAD=$LR D_MODEL=$DM DROPOUT=$DO SEED=111 \
        CKPT_ROOT=/home/weizyuv/expreal/Center_coldtune $PY -u -m pipeline.train 2>&1 | tail -1
    done
  done
done
echo "=== ours grid fix ALL DONE $(date)"
