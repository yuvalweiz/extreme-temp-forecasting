"""Featured pipeline (ConvNeXtTiny-TFT-NN) vs the four baselines (Prophet, SARIMAX, Tab-TFT,
Tab-LSTM): per-region moving-block bootstrap on paired per-sample abs-error differences
(B=10,000, block 4, two-sided, seed 0; eval_lib.moving_block_bootstrap), Holm across the four
baselines within each (season, target, region). Soft loss = MAE_3 per sample. Reproduces the
significance_featured_v2.csv protocol on the current released files.
Writes final_tables/significance_featured_v3.csv (+ spatial_vs_tabs subset).
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/weizyuv/article /repo/src"); sys.path.insert(0, "/home/weizyuv/article /repo/src/eval")
from eval_lib import moving_block_bootstrap, holm_bonferroni
RC="/home/weizyuv/article /repo/results/actual_vs_predicted_corrected"; RO="/home/weizyuv/article /repo/results/actual_vs_predicted"
REF="ConvNeXtTiny-TFT-NN"; BASE=["Tab-TFT","Tab-LSTM","Prophet","SARIMAX"]; REGS=["Center","Negev","Northwest"]
def loss(model, season, reg, target):
    base = RC if os.path.exists(f"{RC}/{season}/{reg}/{model}__{target}.csv") else RO
    d=pd.read_csv(f"{base}/{season}/{reg}/{model}__{target}.csv").set_index("sample_id")
    if target=="anchor": return d["abs_error"]
    return pd.Series(np.abs(d[["predicted_p3","predicted_p7","predicted_p15"]].values-d[["actual_p3","actual_p7","actual_p15"]].values).mean(1), index=d.index)
rows=[]
for season in ["summer","winter"]:
    for target in ["anchor","soft"]:
        for reg in REGS:
            ref=loss(REF,season,reg,target); fam=[]
            for b in BASE:
                bl=loss(b,season,reg,target); j=pd.concat([bl.rename("b"),ref.rename("r")],axis=1).dropna()
                dmae,lo,hi,p=moving_block_bootstrap((j.b-j.r).values)
                fam.append(dict(season=season,target=target,region=reg,vs=b,dMAE=round(dmae,3),p=round(p,4)))
            ph=holm_bonferroni([f["p"] for f in fam])
            for f,h in zip(fam,ph): f["p_holm"]=round(float(h),4); rows.append(f)
df=pd.DataFrame(rows); out="/home/weizyuv/expreal/final_tables"
df.to_csv(f"{out}/significance_featured_v3.csv",index=False)
df[df.vs.isin(["Tab-TFT","Tab-LSTM"])].assign(ref=REF)[["ref","season","target","region","vs","dMAE","p_holm"]].to_csv(f"{out}/significance_spatial_vs_tabs_v3.csv",index=False)
v2=pd.read_csv(f"{out}/significance_featured_v2.csv")
m=df.merge(v2,on=["season","target","region","vs"],suffixes=("_v3","_v2"))
w=m[m.season=="winter"]; print("winter rows identical to v2:", bool(((w.dMAE_v3-w.dMAE_v2).abs()<1e-3).all() and ((w.p_holm_v3-w.p_holm_v2).abs()<1e-3).all()), f"({len(w)} rows)")
s=m[m.season=="summer"]; print("summer rows changed:", int(((s.dMAE_v3-s.dMAE_v2).abs()>1e-3).sum()), "of", len(s))
sig=df[df.p_holm<0.05]; print(f"\nsignificant (Holm<0.05) wins: {int((sig.dMAE>0).sum())}, losses: {int((sig.dMAE<0).sum())}, of {len(df)}")
print("\nper-region significant counts (target, season, vs): ")
for (se,tg,vs),g in df.groupby(["season","target","vs"]): print(f"  {se:6s} {tg:6s} {vs:8s} dMAE mean {g.dMAE.mean():+.3f}  sig {int((g.p_holm<0.05).sum())}/3  regions: {list(g[g.p_holm<0.05].region)}")
