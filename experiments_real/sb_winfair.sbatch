#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-12:00:00
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# WINTER deployment-selection: grid + objective candidates on COLD std targets, Center.
# Env: FAM in {stf, stl, ours}.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 REGION=Center COLD=1 K=1
case $FAM in
  stf)  ENV="STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_MIN_Center HEAD=temporalfusion CACHE=1";;
  stl)  ENV="STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_MIN_Center HEAD=lstm CACHE=1";;
  ours) ENV="DATASET_DIR=/home/weizyuv/stationvec_MIN_Center PAPERW=/home/weizyuv/interp_weights/paper_opt.npz HEAD=temporalfusion CACHE=1";;
esac
for LR in 3e-4 1e-4 3e-5; do
  for DM in 128 256; do
    for DO in 0.1 0.3; do
      TAG="${FAM}tgwinL${LR}D${DM}O${DO}"; TAG=${TAG//./p}; TAG=${TAG//-/}
      echo "=== $TAG $(date +%H:%M)"
      env $ENV RUN_TAG=$TAG ALPHA=2 BETA=1 SELECT=paper LR_HEAD=$LR D_MODEL=$DM DROPOUT=$DO SEED=111 \
        CKPT_ROOT=/home/weizyuv/expreal/Center_coldtune $PY -u -m pipeline.train 2>&1 | tail -1
    done
  done
done
read LR DM DO <<< $($PY - <<PYEOF
import json, glob
best=None
for d in glob.glob("/home/weizyuv/expreal/Center_coldtune/${FAM}tgwinL*__*"):
    m=json.load(open(d+"/meta.json")); v=m["val"]["mae_mean"]
    cfg=d.split("/")[-1].split("__")[0].replace("${FAM}tgwin","")
    if best is None or v<best[0]: best=(v,cfg)
c=best[1]
lr={"3e4":"3e-4","1e4":"1e-4","3e5":"3e-5"}[c.split("D")[0][1:]]
print(lr, c.split("D")[1].split("O")[0], c.split("O")[1].replace("p","."))
PYEOF
)
echo "=== $FAM winter grid winner: LR=$LR D=$DM DO=$DO (plain-val ranked)"
for ABSPEC in "a2b1:2:1:paper" "a1b0:1:0:all"; do
  IFS=: read ABTAG A B SEL <<< "$ABSPEC"
  for SEED in 111 222 333; do
    echo "=== ${FAM}s2win_${ABTAG} s$SEED $(date +%H:%M)"
    env $ENV RUN_TAG=${FAM}s2win_${ABTAG} ALPHA=$A BETA=$B SELECT=$SEL LR_HEAD=$LR D_MODEL=$DM DROPOUT=$DO SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/Center_coldtune $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== ${FAM} winter fairness ALL DONE $(date)"
