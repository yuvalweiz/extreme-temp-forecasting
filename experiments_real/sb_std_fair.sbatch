#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-12:00:00
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
# Fairness protocol on the CURRENT paper outputs (anchor+soft, no TARGETS override):
# stage 1 grid + in-job winner selection + stage 2 (a,b)/objective candidates, Center.
# Env: FAM in {atft, stf}. Same pre-registered grid/candidates as the top-k program.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 CACHE=1 REGION=Center
case $FAM in
  stf)  export STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_Center HEAD=temporalfusion;;
  atft) export TABULAR=1 HEAD=temporalfusion;;
esac
for LR in 3e-4 1e-4 3e-5; do
  for DM in 128 256; do
    for DO in 0.1 0.3; do
      TAG="${FAM}tgstdL${LR}D${DM}O${DO}"; TAG=${TAG//./p}; TAG=${TAG//-/}
      echo "=== $TAG $(date +%H:%M)"
      RUN_TAG=$TAG ALPHA=2 BETA=1 SELECT=paper LR_HEAD=$LR D_MODEL=$DM DROPOUT=$DO SEED=111 \
        CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
    done
  done
done
read LR DM DO <<< $($PY - <<PYEOF
import json, glob
best=None
for d in glob.glob("/home/weizyuv/expreal/Center_tune/${FAM}tgstdL*__*"):
    m=json.load(open(d+"/meta.json")); v=min(m["topk_scores"].values())
    cfg=d.split("/")[-1].split("__")[0].replace("${FAM}tgstd","")
    if best is None or v<best[0]: best=(v,cfg)
c=best[1]
lr=c.split("D")[0][1:]
lr={"3e4":"3e-4","1e4":"1e-4","3e5":"3e-5"}[lr]
print(lr, c.split("D")[1].split("O")[0], c.split("O")[1].replace("p","."))
PYEOF
)
echo "=== $FAM std grid winner: LR=$LR D=$DM DO=$DO"
for ABSPEC in "a2b1:2:1:paper" "a3b15:3:1.5:paper" "a1b0:1:0:all"; do
  IFS=: read ABTAG A B SEL <<< "$ABSPEC"
  for SEED in 111 222 333; do
    echo "=== ${FAM}s2std_${ABTAG} s$SEED $(date +%H:%M)"
    RUN_TAG=${FAM}s2std_${ABTAG} ALPHA=$A BETA=$B SELECT=$SEL LR_HEAD=$LR D_MODEL=$DM DROPOUT=$DO SEED=$SEED \
      CKPT_ROOT=/home/weizyuv/expreal/Center_tune $PY -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== ${FAM} std fairness ALL DONE $(date)"
