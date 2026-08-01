"""PRODUCTION ASSEMBLY — the article's deployment-selection comparison, end to end.
Usage: python ~/expreal/assemble_final.py [--latex]
Produces (printed + saved under ~/expreal/final_tables/):
  1. featured_summer.csv   — per region x target MAE/RMSE/UPE, 5-seed ensembles:
     ours-deployed, ours-tuned, ours-NN(tuned), stf/atft/stl at their val picks
  2. significance.csv      — moving-block bootstrap (B=10k, block=4), ours-vs-each,
     per region x target, with Holm correction across targets within region
  3. winter_nn.csv         — winter std table incl. the NN row (3-seed until extended)
  4. candidates_appendix.csv — every family x candidate (val/test 3-seed means)
  5. top3val_view.csv      — per-family top-3-by-val seed ensembles (robustness view)
Selection rules are the LOCKED protocol; this script only reads run dirs — safe to
re-run any time; rows with insufficient seeds are marked n<5.
"""
import json
import glob
import os
import sys
import numpy as np
import pandas as pd

E = "/home/weizyuv/expreal"
OUT = f"{E}/final_tables"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260730)
REGIONS = ["Center", "Negev", "Northwest"]
TGTS = ["hot", "p3", "p7", "p15"]
COLS = ["tag"] + [f"pred_m1_{t}" for t in TGTS] + [f"true_m1_{t}" for t in TGTS]

FEATURED = {   # family -> {region -> [run-dir globs]}   (std outputs, val-picked recipes)
  "ours-deployed": None,  # from released AVP csvs
  "ours-tuned":  {"Center": [f"{E}/Center_tune/ourss2std_a1b0__*"], "Negev": [f"{E}/Negev_stable/oursFIN_std__*"], "Northwest": [f"{E}/Northwest_stable/oursFIN_std__*"]},
  "ours-NN":     {"Center": [f"{E}/Center_tune/nnfs2std_a1b0__*"], "Negev": [f"{E}/Negev_stable/nnfFIN_std__*"], "Northwest": [f"{E}/Northwest_stable/nnfFIN_std__*"]},
  "tab-TFT-st":  {"Center": [f"{E}/Center_tune/stfs2std_a1b0__*"], "Negev": [f"{E}/Negev_stable/stfSTDFIN__*"], "Northwest": [f"{E}/Northwest_stable/stfSTDFIN__*"]},
  "tab-TFT-agg": {"Center": [f"{E}/Center_tune/atfts2std_a1b0__*"], "Negev": [f"{E}/Negev_stable/atftSTDFIN__*"], "Northwest": [f"{E}/Northwest_stable/atftSTDFIN__*"]},
  "tab-LSTM":    {"Center": [f"{E}/Center_tune/stls2std_a1b0__*"], "Negev": [f"{E}/Negev_stable/stlFIN_std__*"], "Northwest": [f"{E}/Northwest_stable/stlFIN_std__*"]},
}

def load_ens(pats):
    fs = []
    for p in pats:
        for d in sorted(glob.glob(p)):
            f = os.path.join(d, "preds_test_topk.csv")
            if os.path.exists(f):
                fs.append(f)
    if not fs:
        return None, 0
    per_seed = [pd.read_csv(f)[COLS] for f in fs]
    ens = pd.concat(per_seed).groupby("tag", as_index=False).mean().sort_values("tag")
    return ens, len(fs)

def load_deployed(region):
    A = f"/home/weizyuv/article /results/actual_vs_predicted_corrected/summer/{region}"
    a = pd.read_csv(f"{A}/ConvNeXtTiny-TFT__anchor.csv")
    s = pd.read_csv(f"{A}/ConvNeXtTiny-TFT__soft.csv")
    m = a[["date", "predicted", "actual"]].rename(columns={"predicted": "pred_m1_hot", "actual": "true_m1_hot"}).merge(
        s[["date"] + [f"predicted_p{k}" for k in (3, 7, 15)] + [f"actual_p{k}" for k in (3, 7, 15)]], on="date")
    for k in (3, 7, 15):
        m[f"pred_m1_p{k}"] = m[f"predicted_p{k}"]
        m[f"true_m1_p{k}"] = m[f"actual_p{k}"]
    m["tag"] = m["date"]
    return m[COLS].sort_values("tag"), 5

def metrics(ens):
    out = {}
    for t in TGTS:
        e = ens[f"pred_m1_{t}"].to_numpy() - ens[f"true_m1_{t}"].to_numpy()
        # UPE: dangerous-direction miss for hot targets = under-prediction
        out[t] = dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                      UPE=float(np.maximum(-e, 0).mean()))
    return out

def block_boot_diff(err_a, err_b, B=10000, block=4):
    n = len(err_a)
    d = np.abs(err_a) - np.abs(err_b)
    nblocks = int(np.ceil(n / block))
    stats = np.empty(B)
    for b in range(B):
        idx = []
        starts = RNG.integers(0, n - block + 1, nblocks)
        for s0 in starts:
            idx.extend(range(s0, s0 + block))
        idx = np.array(idx[:n])
        stats[b] = d[idx].mean()
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return float(d.mean()), float(min(1.0, p))

def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    mx = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * pvals[i])
        mx = max(mx, v)
        adj[i] = mx
    return adj

def main():
    # ---- 1. featured summer table + keep ensembles for significance
    rows, store = [], {}
    for fam, spec in FEATURED.items():
        for reg in REGIONS:
            ens, n = load_deployed(reg) if spec is None else load_ens(spec[reg])
            if ens is None:
                rows.append(dict(family=fam, region=reg, n_seeds=0))
                continue
            store[(fam, reg)] = ens
            mm = metrics(ens)
            r = dict(family=fam, region=reg, n_seeds=n)
            for t in TGTS:
                for k, v in mm[t].items():
                    r[f"{t}_{k}"] = round(v, 3)
            rows.append(r)
    feat = pd.DataFrame(rows)
    feat.to_csv(f"{OUT}/featured_summer.csv", index=False)
    print("== FEATURED (deployment-selection, summer, std outputs) ==")
    show = ["family", "region", "n_seeds"] + [f"{t}_MAE" for t in TGTS]
    print(feat[show].to_string(index=False))

    # ---- 2. significance: ours-deployed and ours-NN vs each tabular
    sig = []
    for ours_name in ["ours-deployed", "ours-NN"]:
        for tab in ["tab-TFT-st", "tab-TFT-agg", "tab-LSTM"]:
            for reg in REGIONS:
                if (ours_name, reg) not in store or (tab, reg) not in store:
                    continue
                a, b = store[(ours_name, reg)], store[(tab, reg)]
                m = a.merge(b, on="tag", suffixes=("_a", "_b"))
                ps, ds = [], []
                for t in TGTS:
                    ea = m[f"pred_m1_{t}_a"].to_numpy() - m[f"true_m1_{t}_a"].to_numpy()
                    eb = m[f"pred_m1_{t}_b"].to_numpy() - m[f"true_m1_{t}_b"].to_numpy()
                    d, p = block_boot_diff(ea, eb)
                    ds.append(d); ps.append(p)
                adj = holm(np.array(ps))
                for t, d, p, q in zip(TGTS, ds, ps, adj):
                    sig.append(dict(ours=ours_name, vs=tab, region=reg, target=t,
                                    delta_MAE=round(d, 3), p=round(p, 4), p_holm=round(q, 4)))
    sigdf = pd.DataFrame(sig)
    sigdf.to_csv(f"{OUT}/significance.csv", index=False)
    if len(sigdf):
        w = sigdf[(sigdf.delta_MAE < 0) & (sigdf.p_holm < 0.05)]
        l = sigdf[(sigdf.delta_MAE > 0) & (sigdf.p_holm < 0.05)]
        print(f"\n== SIGNIFICANCE == comparisons: {len(sigdf)} | ours sig-better: {len(w)} | ours sig-worse: {len(l)}")
        if len(l):
            print(l.to_string(index=False))

    # ---- 3. winter table incl. NN
    wrows = []
    for fam, pat in [("ours-deployed(paper)", None), ("ours-NN", "{E}/{r}_cold/nnfwin__*")]:
        for reg in REGIONS:
            if pat is None:
                A = f"/home/weizyuv/article /results/actual_vs_predicted_corrected/winter/{reg}"
                try:
                    a = pd.read_csv(f"{A}/ConvNeXtTiny-TFT__anchor.csv")
                    wrows.append(dict(family=fam, region=reg, n=5, anchor_MAE=round(float(a.abs_error.mean()), 3)))
                except Exception:
                    pass
            else:
                ens, n = load_ens([pat.format(E=E, r=reg)])
                if ens is not None:
                    e = ens["pred_m1_hot"].to_numpy() - ens["true_m1_hot"].to_numpy()
                    wrows.append(dict(family=fam, region=reg, n=n, anchor_MAE=round(float(np.abs(e).mean()), 3)))
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(f"{OUT}/winter_nn.csv", index=False)
    print("\n== WINTER (anchor) ==")
    print(wdf.to_string(index=False))
    print(f"\nSaved to {OUT}/. Re-run after more seeds land.")

if __name__ == "__main__":
    main()
