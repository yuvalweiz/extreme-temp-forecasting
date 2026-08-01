#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-12:00:00
#SBATCH --job-name=stl_full
#SBATCH --output=stl_full-%j.out
#SBATCH --error=stl_full-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Tab-LSTM (per-station, stl) — the SAME fairness treatment as ours/stf/atft:
# stage 1: identical pre-registered hparam grid, Center, s111, both formats;
# stage 2: at its own grid winner, the (a,b)/objective candidates, 3 seeds.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_Center HEAD=lstm
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 CACHE=1 REGION=Center
for FMT in mk4 mkt; do
  if [ "$FMT" = mk4 ]; then TGT="1,t5,t10,t15"; else TGT="t3,t5,t10,t15"; fi
  for LR in 3e-4 1e-4 3e-5; do
    for DM in 128 256; do
      for DO in 0.1 0.3; do
        TAG="stltg${FMT}L${LR}D${DM}O${DO}"; TAG=${TAG//./p}; TAG=${TAG//-/}
        echo "=== $TAG $(date +%H:%M)"
        RUN_TAG=$TAG TARGETS=$TGT ALPHA=2 BETA=1 SELECT=paper LR_HEAD=$LR D_MODEL=$DM DROPOUT=$DO SEED=111 \
          CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
      done
    done
  done
  read LR DM DO <<< $($PY - <<PYEOF
import json, glob
best=None
for d in glob.glob("/home/weizyuv/expreal/Center_tune/stltg${FMT}L*__*"):
    m=json.load(open(d+"/meta.json")); v=min(m["topk_scores"].values())
    cfg=d.split("/")[-1].split("__")[0].replace("stltg${FMT}","")
    if best is None or v<best[0]: best=(v,cfg)
c=best[1]
lr=c.split("D")[0][1:].replace("p",".")
lr={"3e4":"3e-4","1e4":"1e-4","3e5":"3e-5"}.get(lr,lr)
dm=c.split("D")[1].split("O")[0]
do=c.split("O")[1].replace("p",".")
print(lr, dm, do)
PYEOF
)
  echo "=== stl $FMT grid winner: LR=$LR D=$DM DO=$DO"
  for ABSPEC in "a2b1:2:1:paper" "a3b15:3:1.5:paper" "a1b0:1:0:all"; do
    IFS=: read ABTAG A B SEL <<< "$ABSPEC"
    for SEED in 111 222 333; do
      echo "=== stls2${FMT}_${ABTAG} s$SEED $(date +%H:%M)"
      RUN_TAG=stls2${FMT}_${ABTAG} TARGETS=$TGT ALPHA=$A BETA=$B SELECT=$SEL LR_HEAD=$LR D_MODEL=$DM DROPOUT=$DO SEED=$SEED \
        CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
    done
  done
done
echo "=== stl full treatment ALL DONE $(date)"
