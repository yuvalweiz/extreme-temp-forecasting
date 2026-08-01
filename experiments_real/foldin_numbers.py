"""FOLD-IN PACKAGE: every number needed to swap the article to the new model.
Emits, per region and season, for the FROZEN picks (final_hot_verdict PICKS + cold csp5):
  anchor + soft: MAE and RMSE, all-sample / extreme / Mean;  per-rank all-case MAE
  (p1,p3,p7,p15); 3-region means; pooled significance vs Prophet, SARIMAX and the
  article Tab-LSTM (paper protocol: moving-block bootstrap B=10,000, blocks 4/2).
Output: ~/expreal/foldin_package.txt (human) — regenerate any time.
"""
import pandas as pd, numpy as np, glob, json

B=10_000; rng=np.random.default_rng(42)
COLS=["tag","true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15",
      "pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]
def bootp(d,block):
    d=np.asarray(d,float); n=len(d)
    if n<block+2: return np.nan
    nb=int(np.ceil(n/block)); st=rng.integers(0,n-block+1,size=(B,nb))
    idx=(st[:,:,None]+np.arange(block)[None,None,:]).reshape(B,-1)[:,:n]
    m=d[idx].mean(1); return min(1.0,2*min((m<=0).mean(),(m>=0).mean()))
def hthr(reg):
    z=np.load(f"/home/weizyuv/Deep Learning Models/Cluster {reg}/dataset_FULL_h180_next30_DOM_1_7_14_21_28/norm_stats_extremes_full_yNONE_v1.npz",allow_pickle=True)
    for k in ["hot_p90","p90","hot_thr"]:
        if k in z: return float(z[k])
def cthr(reg):
    d=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/Winter Models/dataset_FULL_MIN_h180_next30_DOM_1_7_14_21_28"
    return float(np.load(f"{d}/norm_stats_extremes_full_MIN.npz",allow_pickle=True)["cold_p10"])
def ens(pat,tk):
    fs=glob.glob(f"{pat}/preds_test{'_topk' if tk else ''}.csv")
    return pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag",as_index=False).mean(), len(fs)
PICKS={"Center":dict(anchor=("opt3",1),soft=("opa3_opt",1)),
       "Negev":dict(anchor=("ng_keda3",1),soft=("ng3_opt",0)),
       "Northwest":dict(anchor=("No_pbo",1),soft=("nw_ked",1))}
def stats(err, m):
    err=np.asarray(err)
    return dict(mae_all=float(err.mean()), mae_ext=float(err[m].mean()),
                rmse_all=float(np.sqrt((err**2).mean())), rmse_ext=float(np.sqrt((err[m]**2).mean())))
def fmt(s): return (f"MAE {s['mae_all']:.3f}/{s['mae_ext']:.3f} (Mean {np.mean([s['mae_all'],s['mae_ext']]):.3f}) | "
                    f"RMSE {s['rmse_all']:.3f}/{s['rmse_ext']:.3f} (Mean {np.mean([s['rmse_all'],s['rmse_ext']]):.3f})")
out=[]
def P(s): print(s); out.append(s)
for season,low in [("HOT",False),("COLD",True)]:
    agg={"anc":[], "soft":[]}
    for reg,S in [("Center","Ce"),("Negev","Ne"),("Northwest","No")]:
        thr=(cthr if low else hthr)(reg)
        if low:
            d,na=ens(f"/home/weizyuv/expreal/{reg}_cold/{S}_csp__*",0); dsoft,ns_=d,na
        else:
            (at,atk),(st_,stk)=PICKS[reg]["anchor"],PICKS[reg]["soft"]
            d,na=ens(f"/home/weizyuv/expreal/{reg}_stable/{at}__*",atk)
            dsoft,ns_=ens(f"/home/weizyuv/expreal/{reg}_stable/{st_}__*",stk)
        m=(d["true_m1_hot"]<=thr) if low else (d["true_m1_hot"]>=thr)
        anc=stats((d["pred_m1_hot"]-d["true_m1_hot"]).abs(), np.asarray(m))
        ms=(dsoft["true_m1_hot"]<=thr) if low else (dsoft["true_m1_hot"]>=thr)
        serr=np.mean([(dsoft[f"pred_m1_{p}"]-dsoft[f"true_m1_{p}"]).abs() for p in ["p3","p7","p15"]],axis=0)
        soft=stats(serr, np.asarray(ms))
        P(f"{season} {reg:10} ANCHOR({na} runs): {fmt(anc)}")
        P(f"{season} {reg:10} SOFT  ({ns_} runs): {fmt(soft)}")
        ranks=[("p1","pred_m1_hot","true_m1_hot"),("p3","pred_m1_p3","true_m1_p3"),
               ("p7","pred_m1_p7","true_m1_p7"),("p15","pred_m1_p15","true_m1_p15")]
        pr=" ".join(f"{r}={float((d[pc]-d[tc]).abs().mean()):.3f}" for r,pc,tc in ranks)
        P(f"{season} {reg:10} per-rank all-case MAE (anchor model): {pr}")
        agg["anc"].append(anc); agg["soft"].append(soft)
    for fam in ["anc","soft"]:
        mm={k: float(np.mean([a[k] for a in agg[fam]])) for k in agg[fam][0]}
        P(f"{season} 3-REGION-MEAN {fam.upper()}: {fmt(mm)}")
# pooled significance vs Prophet/SARIMAX/TabLSTM (hot anchor, pooled 3 regions, paper style)
P("\n== pooled HOT anchor significance vs baselines (paired block bootstrap) ==")
def base_hot(reg,name):
    Bp="/home/weizyuv/Models Evaluations/Predictions VS Actuals/Summer"
    f=glob.glob(f"{Bp}/{reg}/preds_{name}_{reg}.csv") or glob.glob(f"{Bp}/{reg}/preds_{name}_{reg}_test_wide.csv")
    if not f: return None
    d=pd.read_csv(f[0]); pc=[c for c in d.columns if c.startswith("pred_m1_") or (c.startswith("pred") and c!="pred_point")][0]
    tc="true_m1_hot" if "true_m1_hot" in d else "true_m1_main"
    return d.rename(columns={pc:"pred_b",tc:"true_m1_hot"})[["tag","true_m1_hot","pred_b"]]
for bname,label in [("sarimax","SARIMAX"),("prophet","Prophet"),("tab_lstm_hot_israel_wide","Tab-LSTM")]:
    dd_all,dd_ext=[],[]
    for reg,S in [("Center","Ce"),("Negev","Ne"),("Northwest","No")]:
        thr=hthr(reg); (at,atk)=PICKS[reg]["anchor"]
        o,_=ens(f"/home/weizyuv/expreal/{reg}_stable/{at}__*",atk)
        b=base_hot(reg,bname)
        if b is None: continue
        m=o.merge(b.drop(columns=["true_m1_hot"]),on="tag").sort_values("tag")
        eo=(m["pred_m1_hot"]-m["true_m1_hot"]).abs(); eb=(m["pred_b"]-m["true_m1_hot"]).abs()
        d_=np.asarray(eo)-np.asarray(eb); ext=np.asarray(m["true_m1_hot"]>=thr)
        dd_all.append(d_); dd_ext.append(d_[ext])
    if not dd_all: P(f"  {label}: preds not found (check name)"); continue
    da,de=np.concatenate(dd_all),np.concatenate(dd_ext)
    P(f"  vs {label:9} all: d={da.mean():+.3f} p={bootp(da,4):.4f} | ext: d={de.mean():+.3f} p={bootp(de,2):.4f}")
open("/home/weizyuv/expreal/foldin_package.txt","w").write("\n".join(out))
print("\nsaved -> ~/expreal/foldin_package.txt")
