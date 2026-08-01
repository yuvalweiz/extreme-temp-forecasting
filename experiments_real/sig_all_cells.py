"""Per-cell paired moving-block bootstrap: OURS (frozen picks) vs ARTICLE tabular.
Hot (per-target picks from final_hot_verdict) + cold (csp 5-seed) x 3 regions x 4 cells.
Paper protocol: B=10,000; block length 4 (all-case) / 2 (extreme subset); two-sided.
"""
import pandas as pd, numpy as np, glob

B = 10_000
rng = np.random.default_rng(123)
COLS = ["tag","true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15",
        "pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]

def block_boot_p(d, block):
    d = np.asarray(d, float); n = len(d)
    if n < block + 2: return np.nan
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    means = d[idx].mean(axis=1)
    p_lo = (means <= 0).mean(); p_hi = (means >= 0).mean()
    return min(1.0, 2 * min(p_lo, p_hi))

def ens(pats, topk=False):
    suf = "_topk" if topk else ""
    fs = []
    for p in pats: fs += glob.glob(f"{p}/preds_test{suf}.csv")
    return pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag", as_index=False).mean()

def hthr(reg):
    z = np.load(f"/home/weizyuv/Deep Learning Models/Cluster {reg}/dataset_FULL_h180_next30_DOM_1_7_14_21_28/norm_stats_extremes_full_yNONE_v1.npz", allow_pickle=True)
    for k in ["hot_p90","p90","hot_thr"]:
        if k in z: return float(z[k])
def cthr(reg):
    d=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/Winter Models/dataset_FULL_MIN_h180_next30_DOM_1_7_14_21_28"
    return float(np.load(f"{d}/norm_stats_extremes_full_MIN.npz",allow_pickle=True)["cold_p10"])

def art_hot(reg):
    Bp = "/home/weizyuv/Models Evaluations/Predictions VS Actuals/Summer"
    tm = pd.read_csv(glob.glob(f"{Bp}/{reg}/preds_tab_lstm_hot_israel_wide_{reg}.csv")[0])
    hotcol = [c for c in tm.columns if c.startswith("pred")][0]
    tm = tm.rename(columns={hotcol: "pred_m1_hot"})
    if "true_m1_main" in tm: tm = tm.rename(columns={"true_m1_main": "true_m1_hot"})
    ts = pd.read_csv(glob.glob(f"{Bp}/{reg}/preds_tab_lstm_soft_israel_wide_{reg}.csv")[0])
    sc = {p: [c for c in ts.columns if "pred" in c.lower() and c.lower().endswith(p)][0] for p in ["p3","p7","p15"]}
    ts = ts.rename(columns={sc[p]: f"pred_m1_{p}" for p in sc})
    keep = [c for c in ["tag","true_m1_p3","true_m1_p7","true_m1_p15","pred_m1_p3","pred_m1_p7","pred_m1_p15"] if c in ts]
    return tm[["tag","true_m1_hot","pred_m1_hot"]].merge(ts[keep], on="tag")

def art_cold(reg):
    Bp=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/Final Results Cluster {reg} Winter/TAB_LSTM_ISRAEL_WIDE_WINTER"
    am=pd.read_csv(glob.glob(f"{Bp}/preds_tab_lstm_main_israel_wide_WINTER_{reg}.csv")[0]).rename(
        columns={"true_m1_main":"true_m1_hot","pred_m1_main":"pred_m1_hot"})
    asf=pd.read_csv(glob.glob(f"{Bp}/preds_tab_lstm_soft_israel_wide_WINTER_{reg}.csv")[0]).drop(columns=["true_m1_main"])
    return am.merge(asf,on="tag")

HOT_PICKS = {"Center":  dict(anc=("opt3", True),     soft=("opa3_opt", True)),
             "Negev":   dict(anc=("ng_keda3", True), soft=("ng3_opt", False)),
             "Northwest":dict(anc=("No_pbo", True), soft=("nw_ked", True))}

def run(season):
    print(f"== {season.upper()} : OURS vs ARTICLE tab | per-cell block bootstrap (B={B:,}) ==")
    for reg, S in [("Center","Ce"),("Negev","Ne"),("Northwest","No")]:
        if season == "hot":
            thr = hthr(reg); low = False
            pk = HOT_PICKS[reg]
            oa = ens([f"/home/weizyuv/expreal/{reg}_stable/{pk['anc'][0]}__*_s*"], topk=pk['anc'][1])
            os_ = ens([f"/home/weizyuv/expreal/{reg}_stable/{pk['soft'][0]}__*_s*"], topk=pk['soft'][1])
            art = art_hot(reg)
        else:
            thr = cthr(reg); low = True
            oa = os_ = ens([f"/home/weizyuv/expreal/{reg}_cold/{S}_csp__*"])
            art = art_cold(reg)
        out = []
        for fam, ours in [("anc", oa), ("soft", os_)]:
            m = ours.merge(art, on="tag", suffixes=("", "_a")).sort_values("tag").reset_index(drop=True)
            ext = (m["true_m1_hot"] <= thr) if low else (m["true_m1_hot"] >= thr)
            if fam == "anc":
                e_o = (m["pred_m1_hot"] - m["true_m1_hot"]).abs()
                e_a = (m["pred_m1_hot_a"] - m["true_m1_hot"]).abs()
            else:
                e_o = np.mean([(m[f"pred_m1_{p}"] - m[f"true_m1_{p}"]).abs() for p in ["p3","p7","p15"]], axis=0)
                e_a = np.mean([(m[f"pred_m1_{p}_a"] - m[f"true_m1_{p}"]).abs() for p in ["p3","p7","p15"]], axis=0)
            d = np.asarray(e_o) - np.asarray(e_a)
            for cell, dd, blk in [(f"{fam}_all", d, 4), (f"{fam}_ext", d[np.asarray(ext)], 2)]:
                p = block_boot_p(dd, blk)
                verdict = ("SIG-WIN" if (dd.mean() < 0 and p < .05) else
                           "win-ns" if dd.mean() < 0 else
                           "SIG-LOSS" if p < .05 else "loss-ns")
                out.append(f"{cell}: d={dd.mean():+.3f} p={p:.4f} {verdict}")
        print(f"  {reg:10} | " + " | ".join(out))

run("hot"); print(); run("cold")
