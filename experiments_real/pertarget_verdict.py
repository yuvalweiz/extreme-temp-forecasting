import pandas as pd, numpy as np, glob
import itertools as it
B="/home/weizyuv/Models Evaluations/Predictions VS Actuals/Summer"
def thrf(reg):
    d=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/dataset_FULL_h180_next30_DOM_1_7_14_21_28"
    return float(np.load(f"{d}/norm_stats_extremes_full_yNONE_v1.npz",allow_pickle=True)["hot_p90"])
COLS=["tag","true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15","pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]
def sa(rd,tag,split,topk=False):
    suf="_topk" if topk else ""
    fs=glob.glob(f"/home/weizyuv/expreal/{rd}/{tag}__*_s*/preds_{split}{suf}.csv")
    if not fs: return None
    return pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag",as_index=False).mean()
def anc_cells(d,thr):
    m=d["true_m1_hot"]>=thr; ae=(d["pred_m1_hot"]-d["true_m1_hot"]).abs()
    return np.array([ae.mean(), ae[m].mean()])
def soft_cells(d,thr):
    m=d["true_m1_hot"]>=thr; se,see=[],[]
    for p in ["p3","p7","p15"]:
        e=(d[f"pred_m1_{p}"]-d[f"true_m1_{p}"]).abs(); se.append(e.mean()); see.append(e[m].mean())
    return np.array([np.mean(se), np.mean(see)])
def article_tab(reg,thr):
    th=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_hot_israel_wide_{reg}.csv")[0])
    ts=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_soft_israel_wide_{reg}.csv")[0])
    hc=[c for c in th.columns if "pred" in c.lower()][0]
    sc={p:[c for c in ts.columns if "pred" in c.lower() and c.lower().endswith(p)][0] for p in ["p3","p7","p15"]}
    d=th[["tag","true_m1_hot",hc]].merge(ts[["tag","true_m1_p3","true_m1_p7","true_m1_p15"]+list(sc.values())],on="tag")
    d=d.rename(columns={hc:"pred_m1_hot",sc["p3"]:"pred_m1_p3",sc["p7"]:"pred_m1_p7",sc["p15"]:"pred_m1_p15"})
    return np.concatenate([anc_cells(d,thr),soft_cells(d,thr)])
POOLS={"Center":("Center_stable",["opt3","optm3","A3_opt","optm","selpb_base","selpb_indom","opa3_opt","opa3_optm","ked"]),
       "Negev":("Negev_stable",["ng3_opt","ng3_optm","ng_opt","ng_optm","ng_selsp","ng_cropN","ng_cropNm","ng_ked"]),
       "Northwest":("Northwest_stable",["nw3_opt","nw3_optm","nw_opt","nw_optm","nw_selsp","nw_ked"])}
tot=0
for reg,(rd,tags) in POOLS.items():
    thr=thrf(reg)
    ref=article_tab(reg,thr)
    pool={}
    for tg in tags:
        for tk in [False,True]:
            v=sa(rd,tg,"val",tk); t=sa(rd,tg,"test",tk)
            if v is not None and t is not None: pool[f"{tg}{'_t3' if tk else ''}"]=(v,t)
    names=list(pool)
    for a_,b_ in it.combinations(names,2):
        for w in [0.3,0.5,0.7]:
            dv=pool[a_][0].merge(pool[b_][0],on=COLS[:5],suffixes=("_a","_b"))
            dt=pool[a_][1].merge(pool[b_][1],on=COLS[:5],suffixes=("_a","_b"))
            for c in ["pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]:
                dv[c]=w*dv[c+"_a"]+(1-w)*dv[c+"_b"]; dt[c]=w*dt[c+"_a"]+(1-w)*dt[c+"_b"]
            pool[f"{a_}+{b_}@{w}"]=(dv[COLS].copy(),dt[COLS].copy())
    # per-target val picks: anchor pick by val anchor MeanMAE; soft pick by val soft MeanMAE
    anc_pick=min(pool,key=lambda n: anc_cells(pool[n][0],thr).mean())
    soft_pick=min(pool,key=lambda n: soft_cells(pool[n][0],thr).mean())
    a=anc_cells(pool[anc_pick][1],thr); s=soft_cells(pool[soft_pick][1],thr)
    ours=np.concatenate([a,s]); wins=int((ours<ref).sum()); tot+=wins
    print(f"{reg}: anchor_pick={anc_pick[:38]} soft_pick={soft_pick[:38]}")
    print(f"  OURS  anc {a[0]:.3f}/{a[1]:.3f} soft {s[0]:.3f}/{s[1]:.3f}")
    print(f"  ARTtab anc {ref[0]:.3f}/{ref[1]:.3f} soft {ref[2]:.3f}/{ref[3]:.3f}  -> beats {wins}/4")
print(f"\nTOTAL: {tot}/12 cells")
