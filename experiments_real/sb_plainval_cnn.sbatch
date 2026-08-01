#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00
#SBATCH --job-name=pv_cnn
#SBATCH --output=pv_cnn-%j.out
#SBATCH --error=pv_cnn-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# FULLY-PLAIN protocol completion, CNN cells: ours/std L1e-4 D256 DO0.1 | ours/mkt L3e-5 D128 DO0.3
# | nnf/std L3e-5 D128 DO0.1 | nnf/mkt L1e-4 D256 DO0.3
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 REGION=Center WINDOW=180
export CKPT_ROOT=/home/weizyuv/expreal/Center_tune
OURS="DATASET_DIR=/home/weizyuv/stationvec_Center PAPERW=/home/weizyuv/interp_weights/paper_opt.npz HEAD=temporalfusion CACHE=1"
NNF="FRAMEBANK=/home/weizyuv/expreal/framebank_nn FRAMEBANK_NS=/home/weizyuv/expreal/framebank_nn/norm_stats_x.npz HEAD=temporalfusion CACHE=0"
for SEED in 111 222 333; do
  env $OURS RUN_TAG=oursp2std_a1b0 ALPHA=1 BETA=0 LR_HEAD=1e-4 D_MODEL=256 DROPOUT=0.1 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
  env $OURS RUN_TAG=oursp2mkt_a1b0 TARGETS="t3,t5,t10,t15" ALPHA=1 BETA=0 LR_HEAD=3e-5 D_MODEL=128 DROPOUT=0.3 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
  env $NNF RUN_TAG=nnfp2std_a1b0 ALPHA=1 BETA=0 LR_HEAD=3e-5 D_MODEL=128 DROPOUT=0.1 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
  env $NNF RUN_TAG=nnfp2mkt_a1b0 TARGETS="t3,t5,t10,t15" ALPHA=1 BETA=0 LR_HEAD=1e-4 D_MODEL=256 DROPOUT=0.3 SEED=$SEED $PY -u -m pipeline.train 2>&1 | tail -1
done
echo "=== plainval cnn ALL DONE $(date)"
