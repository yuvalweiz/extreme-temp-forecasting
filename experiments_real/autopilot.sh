#!/bin/bash
# AUTOPILOT — session-independent driver for the output-format campaign.
# Runs under nohup on the login node; survives Claude-session death and author logout.
# Never cancels anything. Idempotent via marker files in ~/expreal/.autopilot/.
# Log: ~/expreal/AUTOPILOT.log
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
MARK=/home/weizyuv/expreal/.autopilot
LOG=/home/weizyuv/expreal/AUTOPILOT.log
mkdir -p $MARK
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }
log "autopilot started (pid $$)"

# helper: true when no experiment jobs are left in the queue (3 consecutive checks)
queue_empty() {
  local n=0
  for i in 1 2 3; do
    out=$(squeue -u weizyuv -h -o '%j' 2>/dev/null) || return 1
    echo "$out" | grep -vq '^pycharm_srv$' && return 1
    sleep 60
  done
  return 0
}

# Phase 1: wait for the cluster queue to drain (all fairness cells, waves, winter)
until queue_empty; do sleep 600; done
log "queue drained"

# Phase 2: verdict snapshot
$PY /home/weizyuv/expreal/verdict.py > /home/weizyuv/expreal/VERDICT_TABLES.txt 2>&1
log "verdict written (phase 2)"

# Phase 3: submit region finals for each family's picked recipe (mkt), if missing
if [ ! -f $MARK/finals_submitted ]; then
  $PY - <<'PYEOF' >> /home/weizyuv/expreal/AUTOPILOT.log 2>&1
import json, glob, subprocess, os
import numpy as np
E="/home/weizyuv/expreal"; C=f"{E}/Center_stable"; T=f"{E}/Center_tune"
def runs(pats):
    out=[]
    for p in pats:
        for d in glob.glob(p):
            try: m=json.load(open(d+"/meta.json"))
            except Exception: continue
            t=m.get("test_topk") or m.get("test")
            out.append((m["val"]["mae_mean"], t["mae_mean"]))
    return out
def stat(rs): return (len(rs), float(np.mean([r[0] for r in rs]))) if rs else None
def gridwin(fam):
    best=None
    for d in glob.glob(f"{T}/{fam}tgmktL*__*"):
        try: m=json.load(open(d+"/meta.json"))
        except Exception: continue
        v=min(m["topk_scores"].values()); tag=d.split("/")[-1].split("__")[0]
        if best is None or v<best[0]: best=(v,tag)
    return best[1] if best else None
def hp_from_tag(tag, fam):
    c=tag.replace(f"{fam}tgmkt","")
    lr={"3e4":"3e-4","1e4":"1e-4","3e5":"3e-5"}[c.split("D")[0][1:]]
    return lr, c.split("D")[1].split("O")[0], c.split("O")[1].replace("p",".")
FAM_ENV={"ours":"DATASET_DIR=/home/weizyuv/stationvec_{R} PAPERW=/home/weizyuv/interp_weights/paper_opt.npz HEAD=temporalfusion",
         "stf":"STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_{R} HEAD=temporalfusion",
         "stl":"STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_{R} HEAD=lstm",
         "atft":"TABULAR=1 HEAD=temporalfusion"}
for fam in ["ours","stf","atft","stl"]:
    gw=gridwin(fam)
    cands={"def+custom":(runs([f"{C}/{fam}_mkt__*"]),("1e-4","256","0.1"),"2 1 paper"),
           "def+plain":(runs([f"{C}/{fam}pm_mkt__*"]),("1e-4","256","0.1"),"1 0 all")}
    if gw:
        hp=hp_from_tag(gw,fam)
        cands["tun+custom"]=(runs([f"{T}/{gw}__*",f"{T}/{fam}s2mkt_a2b1__*"]),hp,"2 1 paper")
        cands["tun+a3b15"]=(runs([f"{T}/{fam}s2mkt_a3b15__*"]),hp,"3 1.5 paper")
        cands["tun+plain"]=(runs([f"{T}/{fam}s2mkt_a1b0__*"]),hp,"1 0 all")
    avail={k:v for k,v in cands.items() if v[0]}
    pickk=min(avail,key=lambda k:stat(avail[k][0])[1])
    (lr,dm,do),ab=avail[pickk][1],avail[pickk][2]
    a,b,sel=ab.split()
    # already-covered picks: default recipes have Negev/NW wave runs
    if pickk=="def+custom" or pickk=="def+plain":
        print(f"[finals] {fam}: pick {pickk} — regional runs already exist, skip submit"); continue
    body=f"""#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-6:00:00
#SBATCH --job-name=FIN_{fam}
#SBATCH --output=FIN_{fam}-%j.out
#SBATCH --error=FIN_{fam}-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
PY={os.environ.get('PY','/home/weizyuv/.conda/envs/timesfm311/bin/python')}
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5 CACHE=1
for R in Negev Northwest; do
  for SEED in 111 222 333; do
    echo "=== {fam}FIN_mkt $R s$SEED"
    env {FAM_ENV[fam].replace('{R}','$R')} REGION=$R RUN_TAG={fam}FIN_mkt TARGETS="t3,t5,t10,t15" \\
      ALPHA={a} BETA={b} SELECT={sel} LR_HEAD={lr} D_MODEL={dm} DROPOUT={do} SEED=$SEED \\
      CKPT_ROOT=/home/weizyuv/expreal/${{R}}_stable /home/weizyuv/.conda/envs/timesfm311/bin/python -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== {fam} finals ALL DONE"
"""
    p=f"/home/weizyuv/expreal/.autopilot/FIN_{fam}.sbatch"
    open(p,"w").write(body)
    r=subprocess.run(["sbatch",p],capture_output=True,text=True)
    print(f"[finals] {fam}: pick {pickk} (lr={lr} d={dm} do={do} ab={a},{b}) -> {r.stdout.strip() or r.stderr.strip()}")
PYEOF
  touch $MARK/finals_submitted
  log "region finals submitted"
fi

# Phase 4: wait for finals, then final verdict
sleep 120
until queue_empty; do sleep 600; done
$PY /home/weizyuv/expreal/verdict.py > /home/weizyuv/expreal/VERDICT_TABLES.txt 2>&1
log "FINAL verdict written to expreal/VERDICT_TABLES.txt — campaign data complete"
