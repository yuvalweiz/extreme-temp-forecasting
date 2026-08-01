#!/bin/bash
# Cold closer wave: apply the hot-winning machinery to cold.
# Members per region: csp+SAVE_TOPK (5 seeds), cked KED frames (3), ckeda3 KED+ALPHA=3 (3)
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
for REG in Center:Ce Negev:Ne Northwest:No; do
  R=${REG%%:*}; S=${REG##*:}
  emit () { # $1=tag $2=seed $3=extra_exports
    local TAG=$1 SEED=$2 EXTRA=$3
    local F=/home/weizyuv/slurm_real/cw_${S}_${TAG}_s${SEED}.sbatch
    cat > "$F" <<EOF
#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=main
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --job-name=cw${S}${TAG}${SEED}
#SBATCH --output=/home/weizyuv/slurm_real/cw_${S}_${TAG}_s${SEED}-%j.out
#SBATCH --error=/home/weizyuv/slurm_real/cw_${S}_${TAG}_s${SEED}-%j.out
module load anaconda
export PYTHONPATH=/home/weizyuv/artsrc
export COLD=1 REGION=$R CACHE=1 DETERMINISTIC=1 WARMUP=2 K=1 SEED=$SEED EPOCHS=120 PATIENCE=30 CKPT_ROOT="/home/weizyuv/expreal/${R}_cold"
export DATASET_DIR=/home/weizyuv/stationvec_MIN_${R}
export RUN_TAG=${S}_${TAG} SAVE_TOPK=5
$EXTRA
$PY -u -m pipeline.train
EOF
    sbatch "$F"
  }
  for SEED in 111 222 333 444 555; do
    emit ctpk $SEED 'export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz'
  done
  for SEED in 111 222 333; do
    emit cked $SEED 'export PAPERW=/home/weizyuv/interp_weights/kriging_ked.npz'
    emit ckeda3 $SEED 'export PAPERW=/home/weizyuv/interp_weights/kriging_ked.npz
export ALPHA=3'
  done
done
