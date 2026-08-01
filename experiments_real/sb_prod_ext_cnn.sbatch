#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00
#SBATCH --job-name=pr_extcnn
#SBATCH --output=pr_extcnn-%j.out
#SBATCH --error=pr_extcnn-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# PRODUCTION: seeds 444/555 for featured CNN rows (std outputs): nnf-tuned + ours-tuned,
# all regions. Center runs go to Center_tune (join s2std_a1b0 sets); regions to *_stable FIN tags.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SELECT=all SAVE_TOPK=5 WINDOW=180
NNENV="FRAMEBANK=/home/weizyuv/expreal/framebank_nn FRAMEBANK_NS=/home/weizyuv/expreal/framebank_nn/norm_stats_x.npz HEAD=temporalfusion CACHE=0"
for SEED in 444 555; do
  env $NNENV REGION=Center RUN_TAG=nnfs2std_a1b0 ALPHA=1 BETA=0 LR_HEAD=3e-4 D_MODEL=128 DROPOUT=0.1 SEED=$SEED \
    CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
  for R in Negev Northwest; do
    env $NNENV REGION=$R RUN_TAG=nnfFIN_std ALPHA=1 BETA=0 LR_HEAD=3e-4 D_MODEL=128 DROPOUT=0.1 SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
OURSENV="PAPERW=/home/weizyuv/interp_weights/paper_opt.npz HEAD=temporalfusion CACHE=1"
for SEED in 444 555; do
  env $OURSENV DATASET_DIR=/home/weizyuv/stationvec_Center REGION=Center RUN_TAG=ourss2std_a1b0 ALPHA=1 BETA=0 LR_HEAD=3e-5 D_MODEL=128 DROPOUT=0.1 SEED=$SEED \
    CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
  for R in Negev Northwest; do
    env $OURSENV DATASET_DIR=/home/weizyuv/stationvec_$R REGION=$R RUN_TAG=oursFIN_std ALPHA=1 BETA=0 LR_HEAD=3e-5 D_MODEL=128 DROPOUT=0.1 SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/${R}_stable $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== prod cnn extensions ALL DONE $(date)"
