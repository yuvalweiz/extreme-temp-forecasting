#!/bin/bash
# Extreme-squeeze fleet: pinball tail-quantile loss, extreme oversampling, LOSO-ktuned kernel.
# Hot per-region champion alpha/beta; cold default. All guarded-protocol members (val-only admission later).
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
emit () {
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
    echo "$3"
    echo "$PY -u -m pipeline.train"
  } > "$F"
  sbatch "$F" > /dev/null && echo -n .
}
declare -A AB=( [Center]="ALPHA=2 BETA=1" [Negev]="ALPHA=3 BETA=1.5" [Northwest]="ALPHA=1 BETA=0" )
declare -A SH=( [Center]=Ce [Negev]=Ne [Northwest]=No )
for R in Center Negev Northwest; do
  S=${SH[$R]}
  HOTBASE="export REGION=$R CACHE=1 DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 ${AB[$R]} CKPT_ROOT=/home/weizyuv/expreal/${R}_stable"
  COLDBASE="export COLD=1 REGION=$R CACHE=1 DETERMINISTIC=1 WARMUP=2 K=1 EPOCHS=120 PATIENCE=30 SAVE_TOPK=5 CKPT_ROOT=/home/weizyuv/expreal/${R}_cold"
  for SEED in 111 222 333; do
    emit pb_${S}_s$SEED "$HOTBASE RUN_TAG=${S}_pb07 PINBALL=0.7 SEED=$SEED" \
      "export DATASET_DIR=/home/weizyuv/stationvec_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
    emit kt_${S}_s$SEED "$HOTBASE RUN_TAG=${S}_kt SEED=$SEED" \
      "export DATASET_DIR=/home/weizyuv/stationvec_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt_ktuned.npz"
    emit cpb_${S}_s$SEED "$COLDBASE RUN_TAG=${S}_cpb03 PINBALL=0.3 SEED=$SEED" \
      "export DATASET_DIR=/home/weizyuv/stationvec_MIN_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
    emit ckt_${S}_s$SEED "$COLDBASE RUN_TAG=${S}_ckt SEED=$SEED" \
      "export DATASET_DIR=/home/weizyuv/stationvec_MIN_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt_ktuned.npz"
  done
  for SEED in 111 222; do
    emit os_${S}_s$SEED "$HOTBASE RUN_TAG=${S}_os3 EXT_OVERSAMPLE=3 SEED=$SEED" \
      "export DATASET_DIR=/home/weizyuv/stationvec_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
    emit pbo_${S}_s$SEED "$HOTBASE RUN_TAG=${S}_pbo PINBALL=0.7 EXT_OVERSAMPLE=3 SEED=$SEED" \
      "export DATASET_DIR=/home/weizyuv/stationvec_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
    emit cos_${S}_s$SEED "$COLDBASE RUN_TAG=${S}_cos3 EXT_OVERSAMPLE=3 SEED=$SEED" \
      "export DATASET_DIR=/home/weizyuv/stationvec_MIN_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
  done
done
echo
