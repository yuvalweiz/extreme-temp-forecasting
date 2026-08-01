#!/bin/bash
# Cold anchor-extreme closers: alpha-3 members (hot's winning move) + KED seed depth.
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
for RS in Center:Ce Negev:Ne Northwest:No; do
  R=${RS%%:*}; Sh=${RS##*:}
  for SEED in 111 222 333 444 555; do
    emit ca3_${Sh}_s$SEED \
      "export COLD=1 REGION=$R RUN_TAG=${Sh}_ca3 CACHE=1 DETERMINISTIC=1 WARMUP=2 K=1 SEED=$SEED EPOCHS=120 PATIENCE=30 SAVE_TOPK=5 ALPHA=3 CKPT_ROOT=/home/weizyuv/expreal/${R}_cold" \
      "export DATASET_DIR=/home/weizyuv/stationvec_MIN_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
  done
  for SEED in 444 555; do
    emit cw_${Sh}_cked_s$SEED \
      "export COLD=1 REGION=$R RUN_TAG=${Sh}_cked CACHE=1 DETERMINISTIC=1 WARMUP=2 K=1 SEED=$SEED EPOCHS=120 PATIENCE=30 SAVE_TOPK=5 CKPT_ROOT=/home/weizyuv/expreal/${R}_cold" \
      "export DATASET_DIR=/home/weizyuv/stationvec_MIN_$R
export PAPERW=/home/weizyuv/interp_weights/kriging_ked.npz"
    emit cw_${Sh}_ckeda3_s$SEED \
      "export COLD=1 REGION=$R RUN_TAG=${Sh}_ckeda3 CACHE=1 DETERMINISTIC=1 WARMUP=2 K=1 SEED=$SEED EPOCHS=120 PATIENCE=30 SAVE_TOPK=5 ALPHA=3 CKPT_ROOT=/home/weizyuv/expreal/${R}_cold" \
      "export DATASET_DIR=/home/weizyuv/stationvec_MIN_$R
export PAPERW=/home/weizyuv/interp_weights/kriging_ked.npz"
  done
done
echo
