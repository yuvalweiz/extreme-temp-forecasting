#!/bin/bash
# 4090 WORK CHAIN — keeps the local GPU busy after the current jobs finish.
# Waits for fx_nnf_mkt-LOCAL + Negev NN seeds, then runs the NN-frames candidate
# completion sequentially (skips any run whose dir already exists). nohup-safe.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
LOG=/home/weizyuv/expreal/LOCAL4090.log
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }
log "chain armed: waiting for current 4090 work"
while pgrep -f "pipeline.train" >/dev/null 2>&1 || ! grep -q "ALL DONE" /home/weizyuv/fx_nnf_mkt-LOCAL.out 2>/dev/null; do
  sleep 300
done
log "4090 free — starting chain"
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 CACHE=0 WINDOW=180
export FRAMEBANK=/home/weizyuv/expreal/framebank_nn
export FRAMEBANK_NS=/home/weizyuv/expreal/framebank_nn/norm_stats_x.npz
run() {  # run <region> <tag> <targets("" for std)> <alpha> <beta> <select> <seed>
  R=$1; TAG=$2; TGT=$3; A=$4; B=$5; SEL=$6; SEED=$7
  ls -d /home/weizyuv/expreal/${R}_stable/${TAG}__*s${SEED} >/dev/null 2>&1 && { log "skip $TAG $R s$SEED (exists)"; return; }
  log "run $TAG $R s$SEED"
  TGTENV=""; [ -n "$TGT" ] && TGTENV="TARGETS=$TGT"
  env $TGTENV REGION=$R RUN_TAG=$TAG ALPHA=$A BETA=$B SELECT=$SEL SEED=$SEED \
    CKPT_ROOT=/home/weizyuv/expreal/${R}_stable HEAD=temporalfusion \
    $PY -u -m pipeline.train 2>&1 | tail -1 >> $LOG
}
# A) North-West NN anchor (completes multi-region evidence; NW deployed weights (1,0))
for S in 111 222 333; do run Northwest nnfb "" 1 0 paper $S; done
# B) NN mk4 default+custom (candidate missing for nnf/mk4)
for S in 111 222 333; do run Center nnmk4 "1,t5,t10,t15" 2 1 paper $S; done
# C) NN mkt default+plain
for S in 111 222 333; do run Center nnmktpm "t3,t5,t10,t15" 1 0 all $S; done
# D) NN mk4 default+plain
for S in 111 222 333; do run Center nnmk4pm "1,t5,t10,t15" 1 0 all $S; done
# E) NN std default+plain (old outputs)
for S in 111 222 333; do run Center nnfbpm "" 1 0 all $S; done
# F) Negev NN mkt (regional top-k evidence for the NN candidate)
for S in 111 222 333; do run Negev nnmkt "t3,t5,t10,t15" 3 1.5 paper $S; done
log "CHAIN COMPLETE"
