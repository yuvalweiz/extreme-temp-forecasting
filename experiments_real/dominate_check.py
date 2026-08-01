import pandas as pd, numpy as np, glob
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
def cells(d,thr):
    m=d["true_m1_hot"]>=thr
    ae=(d["pred_m1_hot"]-d["true_m1_hot"]).abs()
    se,see=[],[]
    for p in ["p3","p7","p15"]:
        e=(d[f"pred_m1_{p}"]-d[f"true_m1_{p}"]).abs(); se.append(e.mean()); see.append(e[m].mean())
    return np.array([ae.mean(), ae[m].mean(), np.mean(se), np.mean(see)])
def article_tab(reg,thr):
    th=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_hot_israel_wide_{reg}.csv")[0])
    ts=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_soft_israel_wide_{reg}.csv")[0])
    hc=[c for c in th.columns if "pred" in c.lower()][0]
    sc={p:[c for c in ts.columns if "pred" in c.lower() and c.lower().endswith(p)][0] for p in ["p3","p7","p15"]}
    d=th[["tag","true_m1_hot",hc]].merge(ts[["tag","true_m1_p3","true_m1_p7","true_m1_p15"]+list(sc.values())],on="tag")
    d=d.rename(columns={hc:"pred_m1_hot",sc["p3"]:"pred_m1_p3",sc["p7"]:"pred_m1_p7",sc["p15"]:"pred_m1_p15"})
    return cells(d,thr)
POOLS={"Center":("Center_stable",["opt3","optm3","A3_opt","optm","selpb_base","selpb_indom","opa3_opt","opa3_optm","ked"]),
       "Negev":("Negev_stable",["ng3_opt","ng3_optm","ng_opt","ng_optm","ng_selsp","ng_cropN","ng_cropNm","ng_ked"]),
       "Northwest":("Northwest_stable",["nw3_opt","nw3_optm","nw_opt","nw_optm","nw_selsp","nw_ked"])}
for reg,(rd,tags) in POOLS.items():
    thr=thrf(reg)
    ref=article_tab(reg,thr)
    print(f"\n=== {reg} | ARTICLE tab cells: {np.round(ref,3)} ===")
    results=[]
    for tg in tags:
        for tk in [False,True]:
            t=sa(rd,tg,"test",tk); v=sa(rd,tg,"val",tk)
            if t is None or v is None: continue
            c=cells(t,thr); wins=int((c<ref).sum())
            vscore=cells(v,thrf(reg)).mean()   # balanced 4-cell val score (val-only)
            results.append((vscore,f"{tg}{'_t3' if tk else ''}",c,wins))
    results.sort()
    for vs,n,c,w in results[:8]:
        star=" <<<< 4/4 DOMINATES" if w==4 else ""
        print(f"  val={vs:.3f} {n:18} test={np.round(c,3)} beats {w}/4{star}")
    vp=results[0]
    print(f"  VAL-PICK: {vp[1]} -> beats {vp[3]}/4 on test")
