import pandas as pd, numpy as np, glob
import itertools as it
B="/home/weizyuv/Models Evaluations/Predictions VS Actuals/Summer"
FR="/home/weizyuv/Deep Learning Models/Cluster {} /dataset".replace(" /dataset","")
def thrf(reg):
    d=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/dataset_FULL_h180_next30_DOM_1_7_14_21_28"
    return float(np.load(f"{d}/norm_stats_extremes_full_yNONE_v1.npz",allow_pickle=True)["hot_p90"])
COLS=["tag","true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15","pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]
def sa(rd,tag,split,topk=False):
    suf="_topk" if topk else ""
    fs=glob.glob(f"/home/weizyuv/expreal/{rd}/{tag}__*_s*/preds_{split}{suf}.csv")
    if not fs: return None
    return pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag",as_index=False).mean()
def ac(d,thr):
    m=d["true_m1_hot"]>=thr; ae=(d["pred_m1_hot"]-d["true_m1_hot"]).abs()
    return np.array([ae.mean(), ae[m].mean()])
def sc_(d,thr):
    m=d["true_m1_hot"]>=thr; se,see=[],[]
    for p in ["p3","p7","p15"]:
        e=(d[f"pred_m1_{p}"]-d[f"true_m1_{p}"]).abs(); se.append(e.mean()); see.append(e[m].mean())
    return np.array([np.mean(se), np.mean(see)])
def article_tab(reg,thr):
    th=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_hot_israel_wide_{reg}.csv")[0])
    ts=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_soft_israel_wide_{reg}.csv")[0])
    hc=[c for c in th.columns if "pred" in c.lower()][0]
    scm={p:[c for c in ts.columns if "pred" in c.lower() and c.lower().endswith(p)][0] for p in ["p3","p7","p15"]}
    d=th[["tag","true_m1_hot",hc]].merge(ts[["tag","true_m1_p3","true_m1_p7","true_m1_p15"]+list(scm.values())],on="tag")
    d=d.rename(columns={hc:"pred_m1_hot",scm["p3"]:"pred_m1_p3",scm["p7"]:"pred_m1_p7",scm["p15"]:"pred_m1_p15"})
    return np.concatenate([ac(d,thr),sc_(d,thr)])
POOLS={"Center":("Center_stable",["opt3","optm3","A3_opt","optm","selpb_base","selpb_indom","opa3_opt","opa3_optm","ked","ftv_optm3","ftv_opt3"]),
       "Negev":("Negev_stable",["ng3_opt","ng3_optm","ng_opt","ng_optm","ng_selsp","ng_cropN","ng_cropNm","ng_ked","ng_keda3","ftv_ngked"]),
       "Northwest":("Northwest_stable",["nw3_opt","nw3_optm","nw_opt","nw_optm","nw_selsp","nw_ked","nw_softw","ftv_nwopt"])}
def build(rd,tags,thr):
    pool={}
    for tg in tags:
        for tk in [False,True]:
            v=sa(rd,tg,"val",tk); t=sa(rd,tg,"test",tk)
            if v is not None and t is not None: pool[f"{tg}{'_t3' if tk else ''}"]=(v,t)
    for a_,b_ in list(it.combinations(list(pool),2)):
        for w in [0.3,0.5,0.7]:
            dv=pool[a_][0].merge(pool[b_][0],on=COLS[:5],suffixes=("_a","_b"))
            dt=pool[a_][1].merge(pool[b_][1],on=COLS[:5],suffixes=("_a","_b"))
            for c in COLS[5:]:
                dv[c]=w*dv[c+"_a"]+(1-w)*dv[c+"_b"]; dt[c]=w*dt[c+"_a"]+(1-w)*dt[c+"_b"]
            pool[f"{a_}+{b_}@{w}"]=(dv[COLS].copy(),dt[COLS].copy())
    return pool
def crit_pick(pool,thr,kind,target):
    fn=ac if target=="anc" else sc_
    stats={n:fn(v,thr) for n,(v,t) in pool.items()}
    if kind=="mm": return min(stats,key=lambda n:stats[n].mean())
    if kind=="ext": return min(stats,key=lambda n:stats[n][1])
    if kind=="constrained":
        best=min(s[0] for s in stats.values())
        elig={n:s for n,s in stats.items() if s[0]<=best+0.05}
        return min(elig,key=lambda n:elig[n][1])
    if kind=="w2":  return min(stats,key=lambda n:(stats[n][0]+2*stats[n][1])/3)
CRITS=["mm","ext","constrained","w2"]
print(f"{'criterion':13}"+ "".join(f"{r:>11}" for r in POOLS)+f"{'TOTAL':>8}")
for kind in CRITS:
    tot=0; row=[]
    for reg,(rd,tags) in POOLS.items():
        thr=thrf(reg); ref=article_tab(reg,thr); pool=build(rd,tags,thr)
        ap=crit_pick(pool,thr,kind,"anc"); sp=crit_pick(pool,thr,kind,"soft")
        ours=np.concatenate([ac(pool[ap][1],thr),sc_(pool[sp][1],thr)])
        w=int((ours<ref).sum()); tot+=w; row.append(w)
    print(f"{kind:13}"+ "".join(f"{w:>11}" for w in row)+f"{tot:>8}/12")

print("\n=== DETAIL: ext-criterion picks, missing cells ===")
for reg,(rd,tags) in POOLS.items():
    thr=thrf(reg); ref=article_tab(reg,thr); pool=build(rd,tags,thr)
    ap=crit_pick(pool,thr,"ext","anc"); sp=crit_pick(pool,thr,"ext","soft")
    a=ac(pool[ap][1],thr); s=sc_(pool[sp][1],thr)
    cells=["anc_all","anc_ext","soft_all","soft_ext"]; ours=np.concatenate([a,s])
    miss=[f"{cells[i]} ({ours[i]:.3f} vs {ref[i]:.3f}, gap {ours[i]-ref[i]:+.3f})" for i in range(4) if ours[i]>=ref[i]]
    print(f"{reg}: missing -> {miss if miss else 'NONE (4/4)'}")
