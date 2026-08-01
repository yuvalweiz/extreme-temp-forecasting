#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-8:00:00
#SBATCH --job-name=stfwin
#SBATCH --output=stfwin-%j.out
#SBATCH --error=stfwin-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Winter per-station Tab-TFT ablation row: matched protocol (deployed cold recipe,
# SELECT=paper, K=1 a2b1), 3 regions x 5 seeds, std cold outputs.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 CACHE=1 COLD=1 K=1
export STATIONVEC=1 HEAD=temporalfusion
for R in Center Negev Northwest; do
  for SEED in 111 222 333 444 555; do
    echo "=== stfwin $R s$SEED $(date +%H:%M)"
    DATASET_DIR=/home/weizyuv/stationvec_MIN_$R REGION=$R RUN_TAG=stfwin ALPHA=2 BETA=1 SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_cold $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== stf winter std ALL DONE $(date)"
