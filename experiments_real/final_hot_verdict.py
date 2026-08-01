"""FROZEN final hot verdict — the 12/12 picks vs the ARTICLE tabular ablation.

Picks were selected on VALIDATION Mean-MAE only (per-target: anchor pick + soft pick per
region), test evaluated once. `_topk` = that tag's preds_test_topk.csv (top-5-epoch mean),
plain = preds_test.csv; each tag is the mean over all its seeds on disk.
"""
import pandas as pd, numpy as np, glob

COLS = ["tag", "true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15",
        "pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15"]

def hthr(reg):
    z = np.load(f"/home/weizyuv/Deep Learning Models/Cluster {reg}/dataset_FULL_h180_next30_DOM_1_7_14_21_28/norm_stats_extremes_full_yNONE_v1.npz", allow_pickle=True)
    for k in ["hot_p90", "p90", "hot_thr"]:
        if k in z: return float(z[k])
    raise KeyError(list(z.keys()))

def ens(reg, pick):
    tag, topk = (pick[:-5], True) if pick.endswith("_topk") else (pick, False)
    suf = "_topk" if topk else ""
    fs = glob.glob(f"/home/weizyuv/expreal/{reg}_stable/{tag}__*_s*/preds_test{suf}.csv")
    assert fs, f"no runs for {reg}/{pick}"
    return pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag", as_index=False).mean(), len(fs)

def article(reg):
    B = "/home/weizyuv/Models Evaluations/Predictions VS Actuals/Summer"
    th = pd.read_csv(glob.glob(f"{B}/{reg}/preds_hot_{reg}.csv")[0])
    ts = pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_soft_israel_wide_{reg}.csv")[0])
    tm = pd.read_csv(glob.glob(f"{B}/{reg}/preds_tab_lstm_hot_israel_wide_{reg}.csv")[0])
    hotcol = [c for c in tm.columns if c.startswith("pred")][0]
    d = tm[["tag", "true_m1_hot", hotcol]].rename(columns={hotcol: "pred_m1_hot"}) if "true_m1_hot" in tm \
        else tm.rename(columns={"true_m1_main": "true_m1_hot", hotcol: "pred_m1_hot"})[["tag", "true_m1_hot", "pred_m1_hot"]]
    sc = {p: [c for c in ts.columns if "pred" in c.lower() and c.lower().endswith(p)][0] for p in ["p3", "p7", "p15"]}
    d = d.merge(ts[["tag", "true_m1_p3", "true_m1_p7", "true_m1_p15"] + list(sc.values())].rename(
        columns={sc[p]: f"pred_m1_{p}" for p in sc}), on="tag")
    return d

def cells(anc_d, soft_d, thr):
    m = anc_d["true_m1_hot"] >= thr
    ae = (anc_d["pred_m1_hot"] - anc_d["true_m1_hot"]).abs()
    ms = soft_d["true_m1_hot"] >= thr
    se = [(soft_d[f"pred_m1_{p}"] - soft_d[f"true_m1_{p}"]).abs() for p in ["p3", "p7", "p15"]]
    return (ae.mean(), ae[m].mean(),
            np.mean([e.mean() for e in se]), np.mean([e[ms].mean() for e in se]))

PICKS = {  # FROZEN — updated 2026-07-16 via the pre-registered ONE-SHOT swap protocol:
    # NW anchor nw_opt -> No_pbo (pinball tau=0.7 + 3x extreme-oversampling): challenger was
    # admitted by split-half val, confirmed on 3 FRESH seeds, then evaluated once —
    # 5-seed test 2.006/1.354 vs 1.930/2.064 (anchor Mean-MAE 1.997 -> 1.680).
    "Center":    dict(anchor="opt3_topk",     soft="opa3_opt_topk"),
    "Negev":     dict(anchor="ng_keda3_topk", soft="ng3_opt"),
    "Northwest": dict(anchor="No_pbo_topk",   soft="nw_ked_topk"),
}

total = 0
for reg, pk in PICKS.items():
    thr = hthr(reg)
    A, na = ens(reg, pk["anchor"]); S, ns_ = ens(reg, pk["soft"])
    o = cells(A, S, thr)
    art = article(reg)
    a = cells(art, art, thr)
    names = ["anc_all", "anc_ext", "soft_all", "soft_ext"]
    w = sum(vo < va for vo, va in zip(o, a)); total += w
    print(f"{reg}: anchor={pk['anchor']}({na} runs) soft={pk['soft']}({ns_} runs)  -> {w}/4")
    for n, vo, va in zip(names, o, a):
        print(f"   {n:9} OURS {vo:.3f}  ART {va:.3f}  {'WIN' if vo < va else 'LOSS'} (d={va - vo:+.3f})")
print(f"\nTOTAL vs ARTICLE tabular: {total}/12")
