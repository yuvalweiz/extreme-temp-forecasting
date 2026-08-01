"""GRAND VERDICT v3 — hot + cold vs the ARTICLE tabular, with the squeeze members.
Protocol: small PRE-DECLARED pools per family, member picked on VAL Mean-MAE of that
family only, test evaluated once. Reports old pick vs new pick vs article.
"""
import pandas as pd, numpy as np, glob

COLS = ["tag","true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15",
        "pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]

def hthr(reg):
    z=np.load(f"/home/weizyuv/Deep Learning Models/Cluster {reg}/dataset_FULL_h180_next30_DOM_1_7_14_21_28/norm_stats_extremes_full_yNONE_v1.npz",allow_pickle=True)
    for k in ["hot_p90","p90","hot_thr"]:
        if k in z: return float(z[k])
def cthr(reg):
    d=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/Winter Models/dataset_FULL_MIN_h180_next30_DOM_1_7_14_21_28"
    return float(np.load(f"{d}/norm_stats_extremes_full_MIN.npz",allow_pickle=True)["cold_p10"])

def ens(pat, split, tk):
    fs = glob.glob(f"{pat}/preds_{split}{'_topk' if tk else ''}.csv")
    if not fs: return None, 0
    return pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag",as_index=False).mean(), len(fs)

def fam_cells(d, thr, low):
    m = (d["true_m1_hot"]<=thr) if low else (d["true_m1_hot"]>=thr)
    ae = (d["pred_m1_hot"]-d["true_m1_hot"]).abs()
    se = [(d[f"pred_m1_{p}"]-d[f"true_m1_{p}"]).abs() for p in ["p3","p7","p15"]]
    return dict(anc_all=ae.mean(), anc_ext=ae[m].mean(),
                soft_all=np.mean([e.mean() for e in se]), soft_ext=np.mean([e[m].mean() for e in se]))

def art_hot(reg):
    B="/home/weizyuv/Models Evaluations/Predictions VS Actuals/Summer"
    tm=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_hot_israel_wide_{reg}.csv")[0])
    hc=[c for c in tm.columns if c.startswith("pred")][0]
    tm=tm.rename(columns={hc:"pred_m1_hot","true_m1_main":"true_m1_hot"})
    ts=pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_soft_israel_wide_{reg}.csv")[0])
    sc={p:[c for c in ts.columns if "pred" in c.lower() and c.lower().endswith(p)][0] for p in ["p3","p7","p15"]}
    ts=ts.rename(columns={sc[p]:f"pred_m1_{p}" for p in sc})
    keep=[c for c in ["tag","true_m1_p3","true_m1_p7","true_m1_p15","pred_m1_p3","pred_m1_p7","pred_m1_p15"] if c in ts]
    return tm[["tag","true_m1_hot","pred_m1_hot"]].merge(ts[keep],on="tag")
def art_cold(reg):
    B=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/Final Results Cluster {reg} Winter/TAB_LSTM_ISRAEL_WIDE_WINTER"
    am=pd.read_csv(glob.glob(f"{B}/preds_tab_lstm_main_israel_wide_WINTER_{reg}.csv")[0]).rename(
        columns={"true_m1_main":"true_m1_hot","pred_m1_main":"pred_m1_hot"})
    asf=pd.read_csv(glob.glob(f"{B}/preds_tab_lstm_soft_israel_wide_WINTER_{reg}.csv")[0]).drop(columns=["true_m1_main"])
    return am.merge(asf,on="tag")

# pre-declared pools (tag, topk?) per family; hot old picks included as members
HOT = {
 "Center":   dict(anc=[("opt3",1),("Ce_pb07",1),("Ce_pbo",1),("Ce_os3",1),("Ce_kt",1)],
              soft=[("opa3_opt",1),("Ce_kt",1)]),
 "Negev":    dict(anc=[("ng_keda3",1),("Ne_pb07",1),("Ne_pbo",1),("Ne_os3",1),("Ne_kt",1)],
              soft=[("ng3_opt",0),("Ne_kt",1)]),
 "Northwest":dict(anc=[("nw_opt",0),("No_pb07",1),("No_pbo",1),("No_os3",1),("No_kt",1)],
              soft=[("nw_ked",1),("No_kt",1)]),
}
COLD = {
 "Center":   dict(anc=[("Ce_csp",0),("Ce_cked",1),("Ce_ckeda3",1),("Ce_ca3",1),("Ce_cpb03",1),("Ce_ckt",1),("Ce_cos3",1)],
              soft=[("Ce_csp",0),("Ce_ckt",1),("Ce_cpb03",1)]),
 "Negev":    dict(anc=[("Ne_csp",0),("Ne_cked",1),("Ne_ckeda3",1),("Ne_ca3",1),("Ne_cpb03",1),("Ne_ckt",1),("Ne_cos3",1)],
              soft=[("Ne_csp",0),("Ne_ckt",1),("Ne_cpb03",1)]),
 "Northwest":dict(anc=[("No_csp",0),("No_cked",1),("No_ckeda3",1),("No_ca3",1),("No_cpb03",1),("No_ckt",1),("No_cos3",1)],
              soft=[("No_csp",0),("No_ckt",1),("No_cpb03",1)]),
}

def run(season, POOLS, rootfmt, art_fn, low):
    print(f"\n================ {season} ================")
    tot_art = 0
    for reg in ["Center","Negev","Northwest"]:
        thr = (cthr if low else hthr)(reg); root = rootfmt.format(reg=reg)
        art = fam_cells(art_fn(reg), thr, low)
        row = {}
        for fam in ["anc","soft"]:
            V, T = {}, {}
            for tag, tk in POOLS[reg][fam]:
                v,nv = ens(f"{root}/{tag}__*", "val", tk); t,nt = ens(f"{root}/{tag}__*", "test", tk)
                if v is not None and t is not None: V[tag],T[tag]=v,t
            def vs(d):
                c = fam_cells(d, thr, low)
                return (c[f"{fam}_all"]+c[f"{fam}_ext"])/2
            pick = min(V, key=lambda n: vs(V[n]))
            row[fam] = (pick, fam_cells(T[pick], thr, low))
        cells = dict(anc_all=row["anc"][1]["anc_all"], anc_ext=row["anc"][1]["anc_ext"],
                     soft_all=row["soft"][1]["soft_all"], soft_ext=row["soft"][1]["soft_ext"])
        w = sum(cells[c] < art[c] for c in cells); tot_art += w
        print(f"{reg:10} picks: anc={row['anc'][0]} soft={row['soft'][0]}  -> {w}/4 vs ARTICLE")
        for c in ["anc_all","anc_ext","soft_all","soft_ext"]:
            print(f"   {c:9} OURS {cells[c]:.3f}  ART {art[c]:.3f}  {'WIN ' if cells[c]<art[c] else 'LOSS'} (d={art[c]-cells[c]:+.3f})")
    print(f"TOTAL {season}: {tot_art}/12 vs ARTICLE tabular")

run("HOT", HOT, "/home/weizyuv/expreal/{reg}_stable", art_hot, low=False)
run("COLD", COLD, "/home/weizyuv/expreal/{reg}_cold", art_cold, low=True)
