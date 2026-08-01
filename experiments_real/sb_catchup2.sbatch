#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00
#SBATCH --job-name=cu_chain
#SBATCH --output=cu_chain-%j.out
#SBATCH --error=cu_chain-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Catch-up 2: the dead 4090 chain, on cluster. NW NN-anchor; NN default candidates
# (nnmk4/nnmktpm/nnmk4pm/nnfbpm); Negev nnmkt.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 CACHE=0 WINDOW=180 HEAD=temporalfusion
export FRAMEBANK=/home/weizyuv/expreal/framebank_nn FRAMEBANK_NS=/home/weizyuv/expreal/framebank_nn/norm_stats_x.npz
run() { R=$1; TAG=$2; TGT=$3; A=$4; B=$5; SEL=$6; S=$7
  ls -d /home/weizyuv/expreal/${R}_stable/${TAG}__*s${S} >/dev/null 2>&1 && { echo "skip $TAG $R s$S"; return; }
  echo "=== $TAG $R s$S $(date +%H:%M)"
  TGTENV=""; [ -n "$TGT" ] && TGTENV="TARGETS=$TGT"
  env $TGTENV REGION=$R RUN_TAG=$TAG ALPHA=$A BETA=$B SELECT=$SEL SEED=$S \
    CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1; }
for S in 111 222 333; do run Northwest nnfb "" 1 0 paper $S; done
for S in 111 222 333; do run Center nnmk4 "1,t5,t10,t15" 2 1 paper $S; done
for S in 111 222 333; do run Center nnmktpm "t3,t5,t10,t15" 1 0 all $S; done
for S in 111 222 333; do run Center nnmk4pm "1,t5,t10,t15" 1 0 all $S; done
for S in 111 222 333; do run Center nnfbpm "" 1 0 all $S; done
for S in 111 222 333; do run Negev nnmkt "t3,t5,t10,t15" 3 1.5 paper $S; done
echo "=== catchup2 ALL DONE $(date)"
