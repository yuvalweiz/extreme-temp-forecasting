import pandas as pd, numpy as np, glob
B="/home/weizyuv/Models Evaluations/Predictions VS Actuals/Summer"
def thrf(reg):
    d=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/dataset_FULL_h180_next30_DOM_1_7_14_21_28"
    return float(np.load(f"{d}/norm_stats_extremes_full_yNONE_v1.npz",allow_pickle=True)["hot_p90"])
def sa(rd,tag,split="test",topk=False):
    suf="_topk" if topk else ""
    fs=glob.glob(f"/home/weizyuv/expreal/{rd}/{tag}__*_s*/preds_{split}{suf}.csv")
    if not fs: return None
    cols=["tag","true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15","pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]
    return pd.concat([pd.read_csv(f)[cols] for f in fs]).groupby("tag",as_index=False).mean()
def metrics4(d,thr,pmap):
    m=d["true_m1_hot"]>=thr
    ae=(d[pmap["hot"]]-d["true_m1_hot"]).abs()
    se,see=[],[]
    for p in ["p3","p7","p15"]:
        e=(d[pmap[p]]-d[f"true_m1_{p}"]).abs(); se.append(e.mean()); see.append(e[m].mean())
    return ae.mean(), ae[m].mean(), np.mean(se), np.mean(see)
def blend(rd,picks):
    parts=[(sa(rd,t),w) for t,w in picks]
    d=parts[0][0].copy()
    for c in ["pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]:
        d[c]=sum(w*p[c].values for p,w in parts)
    return d
OURS={"Center":[("selpb_indom",0.7),("opa3_optm",0.3)],
      "Negev":[("ng_optm",0.7),("ng_ked",0.3)],
      "Northwest":[("nw_opt",0.7),("nw_selsp",0.3)]}
FTAB={"Center":[("ftab3",0.7),("ftg_a3b15",0.3)],
      "Negev":[("ng3_ftab",0.5),("ng_selft",0.5)],
      "Northwest":[("nw3_ftab",0.7),("nw_selft",0.3)]}
print(f"{'region':10}{'model':22}{'anc_all':>8}{'anc_ext':>8}{'soft_all':>9}{'soft_ext':>9}")
for reg,rd in [("Center","Center_stable"),("Negev","Negev_stable"),("Northwest","Northwest_stable")]:
    thr=thrf(reg)
    rows=[]
    d=blend(rd,OURS[reg])
    rows.append(("OURS (spatial frames)",)+metrics4(d,thr,{"hot":"pred_m1_hot","p3":"pred_m1_p3","p7":"pred_m1_p7","p15":"pred_m1_p15"}))
    d=blend(rd,FTAB[reg])
    rows.append(("fair tab (68-station)",)+metrics4(d,thr,{"hot":"pred_m1_hot","p3":"pred_m1_p3","p7":"pred_m1_p7","p15":"pred_m1_p15"}))
    # ARTICLE's tabular ablation (published Tab-LSTM israel_wide)
    th=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_hot_israel_wide_{reg}.csv")[0])
    ts=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_soft_israel_wide_{reg}.csv")[0])
    hotcol=[c for c in th.columns if "pred" in c.lower() and "hot" in c.lower()][0]
    softcols={p:[c for c in ts.columns if "pred" in c.lower() and c.lower().endswith(p)][0] for p in ["p3","p7","p15"]}
    pub=th[["tag","true_m1_hot",hotcol]].merge(ts[["tag","true_m1_p3","true_m1_p7","true_m1_p15"]+list(softcols.values())],on="tag")
    rows.append(("ARTICLE tab (Tab-LSTM)",)+metrics4(pub,thr,{"hot":hotcol,**softcols}))
    print(f"--- {reg} (ext = anchor >= train_p90 = {thr:.1f}) ---")
    for lab,a,e,sA,sE in rows:
        print(f"{'':10}{lab:22}{a:8.3f}{e:8.3f}{sA:9.3f}{sE:9.3f}")
    o=rows[0]; ft=rows[1]; at=rows[2]
    def v(i,ref): return "WIN" if o[i]<ref[i] else "lose"
    print(f"{'':10}vs fair-tab : anc_all {v(1,ft)} | anc_ext {v(2,ft)} | soft_all {v(3,ft)} | soft_ext {v(4,ft)}")
    print(f"{'':10}vs ARTICLE  : anc_all {v(1,at)} | anc_ext {v(2,at)} | soft_all {v(3,at)} | soft_ext {v(4,at)}")
