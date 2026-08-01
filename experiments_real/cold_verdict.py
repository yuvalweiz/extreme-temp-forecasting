import pandas as pd, numpy as np, glob
def cthr(reg):
    d=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/Winter Models/dataset_FULL_MIN_h180_next30_DOM_1_7_14_21_28"
    z=np.load(f"{d}/norm_stats_extremes_full_MIN.npz",allow_pickle=True)
    return float(z["cold_p10"])
COLS=["tag","true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15","pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]
for reg,short in [("Center","Ce"),("Negev","Ne"),("Northwest","No")]:
    thr=cthr(reg)
    print(f"\n=== {reg} COLD (ext = coldest <= train_p10 = {thr:.2f}) ===")
    for tag,lab in [(f"{short}_csp","SPATIAL (corrected frames)"),(f"{short}_cft","FAIR TABULAR")]:
        fs=glob.glob(f"/home/weizyuv/expreal/{reg}_cold/{tag}__*/preds_test.csv")
        if not fs: print(f"  {lab}: (none yet)"); continue
        d=pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag",as_index=False).mean()
        m=d["true_m1_hot"]<=thr   # LOW tail
        ae=(d.pred_m1_hot-d.true_m1_hot).abs()
        se=np.mean([(d[f"pred_m1_{p}"]-d[f"true_m1_{p}"]).abs() for p in ["p3","p7","p15"]],axis=0)
        se=pd.Series(se)
        print(f"  {lab:28} anc {ae.mean():.3f}/{ae[m].mean():.3f}  soft {se.mean():.3f}/{se[m.values].mean():.3f}  (n_runs={len(fs)}, n_ext={int(m.sum())})")
