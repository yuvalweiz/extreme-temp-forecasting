#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00
#SBATCH --job-name=p2fin
#SBATCH --output=p2fin-%j.out
#SBATCH --error=p2fin-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# P2 (fully-plain) sensitivity: regional finals + 444/555 for the P2-picked recipes.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 CACHE=1
run() { env $1 REGION=$2 RUN_TAG=$3 ALPHA=1 BETA=0 LR_HEAD=$4 D_MODEL=$5 DROPOUT=$6 SEED=$7 \
  CKPT_ROOT=$8 $PY -u -m pipeline.train 2>&1 | tail -1; }
ATFT="TABULAR=1 HEAD=temporalfusion"; STL_E="STATIONVEC=1 HEAD=lstm"; STF_E="STATIONVEC=1 HEAD=temporalfusion"
for R in Negev Northwest; do
  for S in 111 222 333 444 555; do
    echo "=== p2 finals $R s$S $(date +%H:%M)"
    run "$ATFT" $R atftP2FIN 3e-5 256 0.3 $S /home/weizyuv/expreal/${R}_stable
    run "$STL_E DATASET_DIR=/home/weizyuv/stationvec_$R" $R stlP2FIN 1e-4 256 0.3 $S /home/weizyuv/expreal/${R}_stable
    run "$STF_E DATASET_DIR=/home/weizyuv/stationvec_$R" $R stfP2FIN 3e-5 256 0.1 $S /home/weizyuv/expreal/${R}_stable
  done
done
for S in 444 555; do
  run "$ATFT" Center atftp2std_a1b0 3e-5 256 0.3 $S /home/weizyuv/expreal/Center_tune
  run "$STL_E DATASET_DIR=/home/weizyuv/stationvec_Center" Center stlp2std_a1b0 1e-4 256 0.3 $S /home/weizyuv/expreal/Center_tune
  run "$STF_E DATASET_DIR=/home/weizyuv/stationvec_Center" Center stfp2std_a1b0 3e-5 256 0.1 $S /home/weizyuv/expreal/Center_tune
done
echo "=== p2 finals ALL DONE $(date)"
