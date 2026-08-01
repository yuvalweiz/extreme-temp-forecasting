#!/bin/bash
# AUTOPILOT 2 — conditional finals for cells whose selection lands late:
# (ours,std) (stl,std) (nnf,std) (nnf,mk4) (nnf,mkt). If a cell's validation pick is a
# TUNED recipe (no regional runs exist), submit its Negev/NW finals. Runs after the main
# queue drains twice (post first-autopilot finals). Idempotent via markers + run-dir checks.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
MARK=/home/weizyuv/expreal/.autopilot
LOG=/home/weizyuv/expreal/AUTOPILOT.log
log() { echo "[$(date '+%m-%d %H:%M')] AP2 $*" >> $LOG; }
log "autopilot2 started (pid $$)"
queue_empty() {
  for i in 1 2 3; do
    out=$(squeue -u weizyuv -h -o '%j' 2>/dev/null) || return 1
    echo "$out" | grep -vq '^pycharm_srv$' && return 1
    sleep 90
  done
  return 0
}
# wait for main autopilot's finals to be submitted AND queue to drain
until [ -f $MARK/finals_submitted ]; do sleep 600; done
sleep 300
until queue_empty; do sleep 600; done
log "queue drained (post-finals)"
if [ ! -f $MARK/ap2_submitted ]; then
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
def gridwin(fam, mode):
    best=None
    for d in glob.glob(f"{T}/{fam}tg{mode}L*__*"):
        try: m=json.load(open(d+"/meta.json"))
        except Exception: continue
        v=min(m["topk_scores"].values()); tag=d.split("/")[-1].split("__")[0]
        if best is None or v<best[0]: best=(v,tag)
    return best[1] if best else None
def hp(tag, fam, mode):
    c=tag.replace(f"{fam}tg{mode}","")
    lr={"3e4":"3e-4","1e4":"1e-4","3e5":"3e-5"}[c.split("D")[0][1:]]
    return lr, c.split("D")[1].split("O")[0], c.split("O")[1].replace("p",".")
BASE={"ours":"DATASET_DIR=/home/weizyuv/stationvec_$R PAPERW=/home/weizyuv/interp_weights/paper_opt.npz HEAD=temporalfusion CACHE=1",
      "stl":"STATIONVEC=1 DATASET_DIR=/home/weizyuv/stationvec_$R HEAD=lstm CACHE=1",
      "nnf":"FRAMEBANK=/home/weizyuv/expreal/framebank_nn FRAMEBANK_NS=/home/weizyuv/expreal/framebank_nn/norm_stats_x.npz HEAD=temporalfusion CACHE=0 WINDOW=180"}
TGT={"std":"","mk4":"1,t5,t10,t15","mkt":"t3,t5,t10,t15"}
for fam,mode in [("ours","std"),("stl","std"),("nnf","std"),("nnf","mk4"),("nnf","mkt")]:
    gw=gridwin(fam,mode)
    cands={}
    if mode!="std":
        cands["def+custom"]=(runs([f"{C}/{'nnmkt' if (fam,mode)==('nnf','mkt') else ('nnmk4' if (fam,mode)==('nnf','mk4') else fam+'_'+mode)}__*"]),None,"2 1 paper")
        pmtag={"nnf_mkt":"nnmktpm","nnf_mk4":"nnmk4pm"}.get(f"{fam}_{mode}", f"{fam}pm_{mode}")
        cands["def+plain"]=(runs([f"{C}/{pmtag}__*"]),None,"1 0 all")
    else:
        deftag={"ours":None,"stl":"stl","nnf":"nnfb"}[fam]
        if deftag: cands["def+custom"]=(runs([f"{C}/{deftag}__*"]),None,"2 1 paper")
        if fam=="nnf": cands["def+plain"]=(runs([f"{C}/nnfbpm__*"]),None,"1 0 all")
    if gw:
        h=hp(gw,fam,mode)
        cands["tun+custom"]=(runs([f"{T}/{gw}__*",f"{T}/{fam}s2{mode}_a2b1__*"]),h,"2 1 paper")
        cands["tun+a3b15"]=(runs([f"{T}/{fam}s2{mode}_a3b15__*"]),h,"3 1.5 paper")
        cands["tun+plain"]=(runs([f"{T}/{fam}s2{mode}_a1b0__*"]),h,"1 0 all")
    avail={k:v for k,v in cands.items() if v[0]}
    if not avail: print(f"[AP2] {fam}/{mode}: nothing landed, skip"); continue
    pickk=min(avail,key=lambda k:np.mean([x[0] for x in avail[k][0]]))
    hps,ab=avail[pickk][1],avail[pickk][2]
    if hps is None:
        print(f"[AP2] {fam}/{mode}: pick {pickk} (default recipe) — no new finals needed"); continue
    if glob.glob(f"{E}/Negev_stable/{fam}FIN_{mode}__*"):
        print(f"[AP2] {fam}/{mode}: finals already exist, skip"); continue
    lr,dm,do=hps; a,b,sel=ab.split()
    tgt=TGT[mode]; tgtline=f'TARGETS="{tgt}"' if tgt else ""
    body=f"""#!/bin/bash
#SBATCH --chdir=/home/weizyuv
#SBATCH --partition=gpu
#SBATCH --time=0-8:00:00
#SBATCH --job-name=AP2_{fam}_{mode}
#SBATCH --output=AP2_{fam}_{mode}-%j.out
#SBATCH --error=AP2_{fam}_{mode}-%j.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G
export PYTHONPATH=/home/weizyuv/artsrc
export DETERMINISTIC=1 WARMUP=2 SAVE_TOPK=5
for R in Negev Northwest; do
  for SEED in 111 222 333; do
    echo "=== {fam}FIN_{mode} $R s$SEED"
    env {BASE[fam]} REGION=$R RUN_TAG={fam}FIN_{mode} {tgtline} ALPHA={a} BETA={b} SELECT={sel} \\
      LR_HEAD={lr} D_MODEL={dm} DROPOUT={do} SEED=$SEED \\
      CKPT_ROOT=/home/weizyuv/expreal/${{R}}_stable /home/weizyuv/.conda/envs/timesfm311/bin/python -u -m pipeline.train 2>&1 | tail -1
  done
done
echo "=== AP2 {fam} {mode} ALL DONE"
"""
    p=f"{E}/.autopilot/AP2_{fam}_{mode}.sbatch"; open(p,"w").write(body)
    r=subprocess.run(["sbatch",p],capture_output=True,text=True)
    print(f"[AP2] {fam}/{mode}: pick {pickk} tuned {lr}/{dm}/{do} ab={a},{b} -> {r.stdout.strip() or r.stderr.strip()}")
PYEOF
  touch $MARK/ap2_submitted
  log "conditional finals submitted"
fi
sleep 120
until queue_empty; do sleep 600; done
$PY /home/weizyuv/expreal/verdict.py > /home/weizyuv/expreal/VERDICT_TABLES.txt 2>&1
log "AP2 final verdict written — ALL DATA COMPLETE"
