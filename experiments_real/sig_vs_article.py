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
def ac(d,thr):
    m=d["true_m1_hot"]>=thr; ae=(d["pred_m1_hot"]-d["true_m1_hot"]).abs()
    return np.array([ae.mean(), ae[m].mean()])
def sc_(d,thr):
    m=d["true_m1_hot"]>=thr; se,see=[],[]
    for p in ["p3","p7","p15"]:
        e=(d[f"pred_m1_{p}"]-d[f"true_m1_{p}"]).abs(); se.append(e.mean()); see.append(e[m].mean())
    return np.array([np.mean(se), np.mean(see)])
def article_df(reg):
    th=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_hot_israel_wide_{reg}.csv")[0])
    ts=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_soft_israel_wide_{reg}.csv")[0])
    hc=[c for c in th.columns if "pred" in c.lower()][0]
    scm={p:[c for c in ts.columns if "pred" in c.lower() and c.lower().endswith(p)][0] for p in ["p3","p7","p15"]}
    d=th[["tag","true_m1_hot",hc]].merge(ts[["tag","true_m1_p3","true_m1_p7","true_m1_p15"]+list(scm.values())],on="tag")
    return d.rename(columns={hc:"pred_m1_hot",scm["p3"]:"pred_m1_p3",scm["p7"]:"pred_m1_p7",scm["p15"]:"pred_m1_p15"})
POOLS={"Center":("Center_stable",["opt3","optm3","A3_opt","optm","selpb_base","selpb_indom","opa3_opt","opa3_optm","ked"]),
       "Negev":("Negev_stable",["ng3_opt","ng3_optm","ng_opt","ng_optm","ng_selsp","ng_cropN","ng_cropNm","ng_ked"]),
       "Northwest":("Northwest_stable",["nw3_opt","nw3_optm","nw_opt","nw_optm","nw_selsp","nw_ked"])}
def build(rd,tags):
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
rng=np.random.RandomState(0); BOOT=10000
def bb(diff,blk):
    n=len(diff); nb=int(np.ceil(n/blk))
    return np.array([diff[np.concatenate([np.arange(s,min(s+blk,n)) for s in rng.randint(0,n-blk+1,nb)])[:n]].mean() for _ in range(BOOT)])
print(f"{'region':10}{'cell':10}{'ours':>7}{'ARTtab':>8}{'diff':>8}{'p':>7}  verdict")
for reg,(rd,tags) in POOLS.items():
    thr=thrf(reg); pool=build(rd,tags); art=article_df(reg)
    anc=min(pool,key=lambda n: ac(pool[n][0],thr)[1])       # ext-criterion picks
    sft=min(pool,key=lambda n: sc_(pool[n][0],thr)[1])
    da=pool[anc][1].merge(art[["tag","pred_m1_hot"]].rename(columns={"pred_m1_hot":"tb_hot"}),on="tag").sort_values("tag").reset_index(drop=True)
    ds=pool[sft][1].merge(art[["tag","pred_m1_p3","pred_m1_p7","pred_m1_p15"]].rename(columns={f"pred_m1_{p}":f"tb_{p}" for p in ["p3","p7","p15"]}),on="tag").sort_values("tag").reset_index(drop=True)
    ext_a=(da["true_m1_hot"]>=thr).values; ext_s=(ds["true_m1_hot"]>=thr).values
    eo=(da.pred_m1_hot-da.true_m1_hot).abs().values; et=(da.tb_hot-da.true_m1_hot).abs().values
    so=np.mean([(ds[f"pred_m1_{p}"]-ds[f"true_m1_{p}"]).abs().values for p in ["p3","p7","p15"]],axis=0)
    st_=np.mean([(ds[f"tb_{p}"]-ds[f"true_m1_{p}"]).abs().values for p in ["p3","p7","p15"]],axis=0)
    for cell,(o,t,mask,blk) in {"anc_all":(eo,et,None,4),"anc_ext":(eo,et,ext_a,2),
                                "soft_all":(so,st_,None,4),"soft_ext":(so,st_,ext_s,2)}.items():
        d_=(o-t) if mask is None else (o-t)[mask]
        boots=bb(d_,blk); p=2*min((boots<=0).mean(),(boots>=0).mean())
        om=o.mean() if mask is None else o[mask].mean(); tm=t.mean() if mask is None else t[mask].mean()
        verdict=("WIN-sig" if p<0.05 else "win-ns") if om<tm else ("LOSE-sig" if p<0.05 else "lose-ns")
        print(f"{reg:10}{cell:10}{om:7.3f}{tm:8.3f}{om-tm:+8.3f}{p:7.3f}  {verdict}")

print("\n=== POOLED 3-REGION tests (paper's aggregate convention) ===")
agg={"anc_all":[],"anc_ext":[],"soft_all":[],"soft_ext":[]}
for reg,(rd,tags) in POOLS.items():
    thr=thrf(reg); pool=build(rd,tags); art=article_df(reg)
    anc=min(pool,key=lambda n: ac(pool[n][0],thr)[1]); sft=min(pool,key=lambda n: sc_(pool[n][0],thr)[1])
    da=pool[anc][1].merge(art[["tag","pred_m1_hot"]].rename(columns={"pred_m1_hot":"tb_hot"}),on="tag").sort_values("tag").reset_index(drop=True)
    ds=pool[sft][1].merge(art[["tag","pred_m1_p3","pred_m1_p7","pred_m1_p15"]].rename(columns={f"pred_m1_{p}":f"tb_{p}" for p in ["p3","p7","p15"]}),on="tag").sort_values("tag").reset_index(drop=True)
    ea=(da["true_m1_hot"]>=thr).values; es=(ds["true_m1_hot"]>=thr).values
    eo=(da.pred_m1_hot-da.true_m1_hot).abs().values; et=(da.tb_hot-da.true_m1_hot).abs().values
    so=np.mean([(ds[f"pred_m1_{p}"]-ds[f"true_m1_{p}"]).abs().values for p in ["p3","p7","p15"]],axis=0)
    st_=np.mean([(ds[f"tb_{p}"]-ds[f"true_m1_{p}"]).abs().values for p in ["p3","p7","p15"]],axis=0)
    agg["anc_all"].append(eo-et); agg["anc_ext"].append((eo-et)[ea])
    agg["soft_all"].append(so-st_); agg["soft_ext"].append((so-st_)[es])
for cell,parts in agg.items():
    d_=np.concatenate(parts); blk=4 if "all" in cell else 2
    boots=bb(d_,blk); p=2*min((boots<=0).mean(),(boots>=0).mean())
    print(f"  {cell:10}: mean diff {d_.mean():+.3f}  p={p:.4f}  -> {'SIGNIFICANT WIN' if p<0.05 and d_.mean()<0 else ('win-ns' if d_.mean()<0 else ('lose-ns' if p>=0.05 else 'LOSE'))}")
