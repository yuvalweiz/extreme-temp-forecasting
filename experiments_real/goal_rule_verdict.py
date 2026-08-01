import pandas as pd, numpy as np, glob, sys
import itertools as it
reg,rd=sys.argv[1],sys.argv[2]
SP=sys.argv[3].split(","); TB=sys.argv[4].split(",")
t=float(np.load(f"/home/weizyuv/Deep Learning Models/Cluster {reg}/dataset_FULL_h180_next30_DOM_1_7_14_21_28/norm_stats_extremes_full_yNONE_v1.npz",allow_pickle=True)["hot_p90"])
def m2(d,col="pred_m1_hot"):
    e=(d[col]-d["true_m1_hot"]).abs(); m=d["true_m1_hot"]>=t
    return float(e.mean()), float(e[m].mean() if m.any() else e.mean())
def sa(tag,split,topk=False):
    suf="_topk" if topk else ""
    fs=glob.glob(f"/home/weizyuv/expreal/{rd}/{tag}__*_s*/preds_{split}{suf}.csv")
    if not fs: return None
    return pd.concat([pd.read_csv(f)[["tag","true_m1_hot","pred_m1_hot"]] for f in fs]).groupby("tag",as_index=False).mean()
def pool_of(tags):
    pool={}
    for tg in tags:
        for tk in [False,True]:
            v=sa(tg,"val",tk); te=sa(tg,"test",tk)
            if v is not None and te is not None: pool[f"{tg}{'_t3' if tk else ''}"]=(v,te)
    for a_,b_ in list(it.combinations(list(pool),2)):
        for w in [0.3,0.5,0.7]:
            dv=pool[a_][0].merge(pool[b_][0],on=["tag","true_m1_hot"],suffixes=("_a","_b"))
            dt=pool[a_][1].merge(pool[b_][1],on=["tag","true_m1_hot"],suffixes=("_a","_b"))
            dv["pred_m1_hot"]=w*dv.pred_m1_hot_a+(1-w)*dv.pred_m1_hot_b
            dt["pred_m1_hot"]=w*dt.pred_m1_hot_a+(1-w)*dt.pred_m1_hot_b
            pool[f"{a_}+{b_}@{w}"]=(dv[["tag","true_m1_hot","pred_m1_hot"]].copy(),dt[["tag","true_m1_hot","pred_m1_hot"]].copy())
    return pool
def cpick(pool,tol=0.05):
    vals={n:m2(v) for n,(v,te) in pool.items()}
    best=min(a for a,e in vals.values())
    elig={n:vals[n] for n in vals if vals[n][0]<=best+tol}
    pick=min(elig,key=lambda n:elig[n][1])
    return pick,pool[pick][1]
ps,S=cpick(pool_of(SP)); pt,T=cpick(pool_of(TB))
rng=np.random.RandomState(0); B=10000
def bb(diff,blk):
    n=len(diff); nb=int(np.ceil(n/blk))
    return np.array([diff[np.concatenate([np.arange(s,min(s+blk,n)) for s in rng.randint(0,n-blk+1,nb)])[:n]].mean() for _ in range(B)])
d=S.rename(columns={"pred_m1_hot":"sp"}).merge(T[["tag","pred_m1_hot"]].rename(columns={"pred_m1_hot":"tb"}),on="tag").sort_values("tag").reset_index(drop=True)
es=(d.sp-d.true_m1_hot).abs().values; et=(d.tb-d.true_m1_hot).abs().values; ext=(d.true_m1_hot>=t).values
pa=2*min((bb(es-et,4)<=0).mean(),(bb(es-et,4)>=0).mean()); pe=2*min((bb((es-et)[ext],2)<=0).mean(),(bb((es-et)[ext],2)>=0).mean())
print(f"{reg}: SP={ps[:42]} | TB={pt[:34]}")
print(f"  all {es.mean():.3f} vs {et.mean():.3f} (p={pa:.3f}) | ext {es[ext].mean():.3f} vs {et[ext].mean():.3f} (p={pe:.3f})")
