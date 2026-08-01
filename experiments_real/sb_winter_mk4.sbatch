#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# WINTER top-k wave (rank-free format, cold targets = means of k COLDEST days):
# deployed cold recipe (K=1, a2b1) per family, 3 regions x 3 seeds. Env: FAM.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 CACHE=1 COLD=1 K=1
declare -A SN=( [Center]="Ce" [Negev]="Ne" [Northwest]="No" )
for R in Center Negev Northwest; do
  case $FAM in
    ours) EXTRA="DATASET_DIR=/home/weizyuv/stationvec_MIN_$R PAPERW=/home/weizyuv/interp_weights/paper_opt.npz HEAD=temporalfusion";;
    stf)  EXTRA="STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_MIN_$R HEAD=temporalfusion";;
    stl)  EXTRA="STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_MIN_$R HEAD=lstm";;
    atft) EXTRA="TABULAR=1 DATASET_DIR=/home/weizyuv/dsfullmin_$R HEAD=temporalfusion";;
  esac
  for SEED in 111 222 333; do
    echo "=== w${FAM}_mk4 ${SN[$R]} s$SEED $(date +%H:%M)"
    env $EXTRA REGION=$R RUN_TAG=w${FAM}_mk4 TARGETS="1,t5,t10,t15" ALPHA=2 BETA=1 SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_cold $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== winter ${FAM} ALL DONE $(date)"
