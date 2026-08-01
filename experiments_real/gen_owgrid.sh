#!/bin/bash
# Output-weight (u) grid search for EVERY model family:
#  hot anchor (Negev/NW; Center already done locally), cold (3 regions; geometric = existing csp),
#  hot soft 3-output (Center, TARGETS=3,7,15).
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
emit () { # $1=jobname $2=env1 $3=env2
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
CFG4="oweq:1,1,1,1 owgeo:1,0.5,0.25,0.125 owsteep:1,0.25,0.0625,0.0156 owanc:1,0.1,0.1,0.1"
CFG3="soweq:1,1,1 sowgeo:1,0.5,0.25 sowsteep:1,0.25,0.0625 sowanc:1,0.1,0.1"
for SEED in 111 222; do
  # HOT anchor grid — Negev (a3 b1.5) and Northwest (a1 b0); Center covered locally
  for C in $CFG4; do
    T=${C%%:*}; W=${C##*:}
    emit ${T}_Ne_s$SEED \
      "export REGION=Negev RUN_TAG=ng_${T} OUT_WEIGHTS=$W CACHE=1 DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 SEED=$SEED ALPHA=3 BETA=1.5 CKPT_ROOT=/home/weizyuv/expreal/Negev_ablate" \
      "export DATASET_DIR=/home/weizyuv/stationvec_Negev
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
    emit ${T}_No_s$SEED \
      "export REGION=Northwest RUN_TAG=nw_${T} OUT_WEIGHTS=$W CACHE=1 DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 SEED=$SEED ALPHA=1 BETA=0 CKPT_ROOT=/home/weizyuv/expreal/Northwest_ablate" \
      "export DATASET_DIR=/home/weizyuv/stationvec_Northwest
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
  done
  # COLD grid — eq/steep/anc (geometric = the existing csp default runs)
  for C in oweq:1,1,1,1 owsteep:1,0.25,0.0625,0.0156 owanc:1,0.1,0.1,0.1; do
    T=${C%%:*}; W=${C##*:}
    for RS in Center:Ce Negev:Ne Northwest:No; do
      R=${RS%%:*}; Sh=${RS##*:}
      emit c${T}_${Sh}_s$SEED \
        "export COLD=1 REGION=$R RUN_TAG=${Sh}_c${T} OUT_WEIGHTS=$W CACHE=1 DETERMINISTIC=1 WARMUP=2 K=1 SEED=$SEED EPOCHS=120 PATIENCE=30 SAVE_TOPK=5 CKPT_ROOT=/home/weizyuv/expreal/${R}_cold" \
        "export DATASET_DIR=/home/weizyuv/stationvec_MIN_$R
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
    done
  done
  # HOT soft-model grid (3 outputs, ranks 3/7/15) — Center
  for C in $CFG3; do
    T=${C%%:*}; W=${C##*:}
    emit ${T}_Ce_s$SEED \
      "export REGION=Center RUN_TAG=${T} TARGETS=3,7,15 OUT_WEIGHTS=$W CACHE=1 DETERMINISTIC=1 WARMUP=2 SELECT=paper SAVE_TOPK=5 SEED=$SEED ALPHA=2 BETA=1 CKPT_ROOT=/home/weizyuv/expreal/Center_ablate" \
      "export DATASET_DIR=/home/weizyuv/stationvec_Center
export PAPERW=/home/weizyuv/interp_weights/paper_opt.npz"
  done
done
echo
