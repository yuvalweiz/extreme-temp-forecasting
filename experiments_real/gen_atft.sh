#!/bin/bash
# Article-style tabular control (region-mean series) with the TFT head, matched tuning.
# Hot: champion alpha/beta per region. Cold: default (a2 b1) like csp/cft controls.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
emit () { # $1=jobname $2=body
  local F=/home/weizyuv/slurm_real/$1.sbatch
  { echo '#!/bin/bash'
    echo '#SBATCH --chdir=/home/weizyuv'
    echo '#SBATCH --partition=main'
    echo '#SBATCH --gpus=rtx_4090:1'
    echo '#SBATCH --cpus-per-task=8'
    echo '#SBATCH --mem=64G'
    echo '#SBATCH --time=0-06:00:00'
    echo "#SBATCH --job-name=$1"
    echo "#SBATCH --output=/home/weizyuv/slurm_real/$1-%j.out"
    echo "#SBATCH --error=/home/weizyuv/slurm_real/$1-%j.out"
    echo 'module load anaconda'
    echo 'export PYTHONPATH=/home/weizyuv/artsrc'
    echo "$2"
    echo "$PY -u -m pipeline.train"
  } > "$F"
  sbatch "$F"
}
for SEED in 111 222 333; do
  # HOT article-style tabular w/ TFT head
  emit atft_Ce_s$SEED "export REGION=Center RUN_TAG=atft TABULAR=1 CACHE=1 DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 SEED=$SEED ALPHA=2 BETA=1 CKPT_ROOT=\"/home/weizyuv/expreal/Center_stable\"
export DATASET_DIR=/home/weizyuv/dsfull_Center
  emit atft_Ne_s$SEED "export REGION=Negev RUN_TAG=atft TABULAR=1 CACHE=1 DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 SEED=$SEED ALPHA=3 BETA=1.5 CKPT_ROOT=\"/home/weizyuv/expreal/Negev_stable\"
export DATASET_DIR=/home/weizyuv/dsfull_Negev
  emit atft_No_s$SEED "export REGION=Northwest RUN_TAG=atft TABULAR=1 CACHE=1 DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 SEED=$SEED ALPHA=1 BETA=0 CKPT_ROOT=\"/home/weizyuv/expreal/Northwest_stable\"
export DATASET_DIR=/home/weizyuv/dsfull_Northwest
  # COLD article-style tabular w/ TFT head
  for RS in Center:Ce Negev:Ne Northwest:No; do
    R=${RS%%:*}; S=${RS##*:}
    emit catft_${S}_s$SEED "export COLD=1 REGION=$R RUN_TAG=${S}_catft TABULAR=1 CACHE=1 DETERMINISTIC=1 WARMUP=2 K=1 SEED=$SEED EPOCHS=120 PATIENCE=30 SAVE_TOPK=5 CKPT_ROOT=\"/home/weizyuv/expreal/${R}_cold\"
export DATASET_DIR=/home/weizyuv/dsfullmin_${R}
  done
done
