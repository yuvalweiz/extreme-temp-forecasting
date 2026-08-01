#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-12:00:00
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Generic fairness cell: stage-1 grid + in-job val winner + stage-2 (a,b)/objective,
# Center. Env: FAM in {ours, stl, nnf}; MODE in {std, mk4, mkt}.
#   ours = kernel-synthesis pipeline | stl = per-station Tab-LSTM | nnf = ours on the
#   LEARNED-interpolation frame bank. std = current anchor+soft outputs; mk4/mkt = top-k.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 REGION=Center
case $FAM in
  ours) export DATASET_DIR=/home/weizyuv/stationvec_Center PAPERW=/home/weizyuv/interp_weights/paper_opt.npz HEAD=temporalfusion CACHE=1;;
  stl)  export STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_Center HEAD=lstm CACHE=1;;
  nnf)  export FRAMEBANK=/home/weizyuv/expreal/framebank_nn FRAMEBANK_NS=/home/weizyuv/expreal/framebank_nn/norm_stats_x.npz HEAD=temporalfusion CACHE=0 WINDOW=180;;
esac
TGTENV=""
[ "$MODE" = mk4 ] && TGTENV="TARGETS=1,t5,t10,t15"
[ "$MODE" = mkt ] && TGTENV="TARGETS=t3,t5,t10,t15"
for LR in 3e-4 1e-4 3e-5; do
  for DM in 128 256; do
    for DO in 0.1 0.3; do
      TAG="${FAM}tg${MODE}L${LR}D${DM}O${DO}"; TAG=${TAG//./p}; TAG=${TAG//-/}
      echo "=== $TAG $(date +%H:%M)"
      env $TGTENV RUN_TAG=$TAG ALPHA=2 BETA=1 SELECT=paper LR_HEAD=$LR D_MODEL=$DM DROPOUT=$DO SEED=111 \
        CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
    done
  done
done
read LR DM DO <<< $($PY - <<PYEOF
import json, glob
best=None
for d in glob.glob("/home/weizyuv/expreal/Center_tune/${FAM}tg${MODE}L*__*"):
    m=json.load(open(d+"/meta.json")); v=min(m["topk_scores"].values())
    cfg=d.split("/")[-1].split("__")[0].replace("${FAM}tg${MODE}","")
    if best is None or v<best[0]: best=(v,cfg)
c=best[1]
lr={"3e4":"3e-4","1e4":"1e-4","3e5":"3e-5"}[c.split("D")[0][1:]]
print(lr, c.split("D")[1].split("O")[0], c.split("O")[1].replace("p","."))
PYEOF
)
echo "=== $FAM $MODE grid winner: LR=$LR D=$DM DO=$DO"
for ABSPEC in "a2b1:2:1:paper" "a3b15:3:1.5:paper" "a1b0:1:0:all"; do
  IFS=: read ABTAG A B SEL <<< "$ABSPEC"
  for SEED in 111 222 333; do
    echo "=== ${FAM}s2${MODE}_${ABTAG} s$SEED $(date +%H:%M)"
    env $TGTENV RUN_TAG=${FAM}s2${MODE}_${ABTAG} ALPHA=$A BETA=$B SELECT=$SEL LR_HEAD=$LR D_MODEL=$DM DROPOUT=$DO SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== ${FAM} ${MODE} fairness ALL DONE $(date)"
