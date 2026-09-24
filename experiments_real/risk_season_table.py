"""Risk-season anchor MAE (June-September for the maximum task, December-March for the minimum
task) for every model of the anchor table, plus date-blocked pooled bootstrap p-values for the
featured pipeline vs the kernel variant and vs both tabular ablations within the risk season.
Reads the released row-level files; B=10,000, block 4, two-sided, seed 0.
"""
import os, numpy as np, pandas as pd
RC="/home/weizyuv/article /repo/results/actual_vs_predicted_corrected"; RO="/home/weizyuv/article /repo/results/actual_vs_predicted"; RR="/home/weizyuv/article /repo/results"
REGS=["Center","Northwest","Negev"]; MONTHS={"summer":[6,7,8,9],"winter":[12,1,2,3]}
MODELS=[("ConvNeXtTiny-TFT (learned interp.)","ConvNeXtTiny-TFT-NN",None),("ConvNeXtTiny-TFT (kernel)","ConvNeXtTiny-TFT",None),
        ("ConvNeXtTiny-LSTM (kernel)","ConvNeXtTiny-LSTM",None),("Tab-TFT","Tab-TFT",None),("Tab-LSTM","Tab-LSTM",None),
        ("Prophet","Prophet",None),("SARIMAX","SARIMAX",None),("TimesFM-2.5","timesfm","fm"),("Moirai-1.1","moirai","fm"),
        ("Climatology","Climatology",None),("Seasonal naive","SeasonalNaive",None)]
rng=np.random.default_rng(0)
def mbb(d,L=4,B=10000):
    d=np.asarray(d,float); n=len(d); nb=int(np.ceil(n/L)); obs=d.mean()
    st=rng.integers(0,n-L+1,size=(B,nb)); idx=(st[:,:,None]+np.arange(L)).reshape(B,-1)[:,:n]
    return float((np.abs((d-obs)[idx].mean(1))>=abs(obs)).mean()), obs
def series(stem,kind,season,reg):
    if kind=="fm":
        p=pd.read_csv(f"{RR}/{stem}_{reg}{'' if season=='summer' else '_min'}.csv").set_index("tag")
        t=pd.read_csv(f"{RC}/{season}/{reg}/ConvNeXtTiny-TFT-NN__anchor.csv",parse_dates=["date"]); t["tag"]=t.date.dt.strftime("%Y-%m-%d"); t=t.set_index("tag")
        j=t[["actual","date"]].join(p["pred_m1_hot"],how="inner").dropna()
        return pd.Series((j.pred_m1_hot-j.actual).abs().values,index=j.index), j.date
    base=RC if os.path.exists(f"{RC}/{season}/{reg}/{stem}__anchor.csv") else RO
    d=pd.read_csv(f"{base}/{season}/{reg}/{stem}__anchor.csv",parse_dates=["date"]); d["tag"]=d.date.dt.strftime("%Y-%m-%d"); d=d.set_index("tag")
    return d.abs_error, d.date
rows={}; P={}
for season in ["summer","winter"]:
    per={}
    for label,stem,kind in MODELS:
        vals=[]
        for reg in REGS:
            s,dt=series(stem,kind,season,reg); m=dt.dt.month.isin(MONTHS[season]); per[(label,reg)]=s[m]
            vals.append(s[m].mean())
        rows[(season,label)]=vals+[np.mean(vals)]
    for other in ["ConvNeXtTiny-TFT (kernel)","Tab-TFT","Tab-LSTM"]:
        D=[(per[(other,r)]-per[("ConvNeXtTiny-TFT (learned interp.)",r)]).dropna().rename(r) for r in REGS]
        J=pd.concat(D,axis=1).dropna(); pp,dd=mbb(J.mean(1).values)
        per_reg={r:mbb(D[i].values)[0] for i,r in enumerate(REGS)}
        P[(season,other)]=(dd,pp,per_reg)
df=pd.DataFrame(rows).T; df.columns=REGS+["Mean"]; pd.set_option("display.width",160)
print(df.round(3).to_string()); df.to_csv(os.path.expanduser("~/expreal/risk_season_anchor.csv"))
print("\nfeatured vs other, risk season (delta = other - featured, positive favors featured):")
for k,(dd,pp,pr) in P.items(): print(f"  {k[0]:6s} vs {k[1]:28s} pooled d={dd:+.3f} p={pp:.3f} | per-region p:", {r:round(v,3) for r,v in pr.items()})
