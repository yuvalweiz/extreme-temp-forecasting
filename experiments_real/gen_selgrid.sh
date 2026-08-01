#!/bin/bash
# SELECTION-CRITERION experiment: retrain the champion recipe with SELECT=all / SELECT=ext
# (vs the existing SELECT=paper runs). Hot + cold, 3 regions, 3 seeds. Lands overnight.
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
  for SEL in all ext; do
    for SEED in 111 222 333; do
      emit sel${SEL}_${S}_s$SEED \
        "export REGION=$R RUN_TAG=${S}_sel${SEL} SELECT=$SEL CACHE=1 DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 SEED=$SEED ${AB[$R]} CKPT_ROOT=/home/weizyuv/expreal/${R}_ablate" \
        "export DATASET_DIR=/home/weizyuv/stationvec_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
      emit csel${SEL}_${S}_s$SEED \
        "export COLD=1 REGION=$R RUN_TAG=${S}_csel${SEL} SELECT=$SEL CACHE=1 DETERMINISTIC=1 WARMUP=2 K=1 EPOCHS=120 PATIENCE=30 SAVE_TOPK=5 SEED=$SEED CKPT_ROOT=/home/weizyuv/expreal/${R}_cold" \
        "export DATASET_DIR=/home/weizyuv/stationvec_MIN_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
    done
  done
done
echo
