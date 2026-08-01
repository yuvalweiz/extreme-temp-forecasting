#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-8:00:00
#SBATCH --job-name=nnfeat
#SBATCH --output=nnfeat-%j.out
#SBATCH --error=nnfeat-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Featured NN pipeline rows (Framing A, deployed protocol): default hparams, per-region
# deployed (a,b), SELECT=paper, seeds 111-555. Skips existing run dirs.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 CACHE=0 WINDOW=180 HEAD=temporalfusion
export FRAMEBANK=/home/weizyuv/expreal/framebank_nn FRAMEBANK_NS=/home/weizyuv/expreal/framebank_nn/norm_stats_x.npz
run() { R=$1; A=$2; B=$3; S=$4
  ls -d /home/weizyuv/expreal/${R}_stable/nnfb__*__a${A}__b${B}__s${S} >/dev/null 2>&1 && { echo "skip $R a$A b$B s$S"; return; }
  echo "=== nnfb $R a$A b$B s$S $(date +%H:%M)"
  REGION=$R RUN_TAG=nnfb ALPHA=$A BETA=$B SEED=$S CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1; }
for S in 444 555; do run Center 2 1 $S; done
for S in 111 222 333 444 555; do run Negev 3 1.5 $S; done
for S in 444 555; do run Northwest 1 0 $S; done
echo "=== nn featured rows ALL DONE $(date)"
