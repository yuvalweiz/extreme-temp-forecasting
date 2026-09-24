"""Featured summer deployment = ONE (1,0) five-seed run set per region (the same runs serve the
anchor and soft targets), chosen among the independently trained (1,0) batches by validation
anchor MAE (primary target): Center nnfs2std_a1b0, North-West nnfFIN_std, Negev nnfP2FIN.
Replaces the earlier guarded blends (Center anchor averaged the (1,0) run set with the (2,1)
incumbent; soft cells averaged two (1,0) batches). Writes the released row-level files in the
gather_framingA.export_avp schema to BOTH the repo release dir and the local article copy,
backing up the superseded files as *__{anchor,soft}.superseded_blend.csv.
"""
import glob, os, shutil
import numpy as np, pandas as pd

E = "/home/weizyuv/expreal"
OUTS = ["/home/weizyuv/article /repo/results/actual_vs_predicted_corrected",
        "/home/weizyuv/article /results/actual_vs_predicted_corrected"]
PICK = {"Center": f"{E}/Center_tune/nnfs2std_a1b0__*",
        "Northwest": f"{E}/Northwest_stable/nnfFIN_std__*",
        "Negev": f"{E}/Negev_stable/nnfP2FIN__*"}
TGTS = ["hot", "p3", "p7", "p15"]
COLS = ["tag"] + [f"pred_m1_{t}" for t in TGTS] + [f"true_m1_{t}" for t in TGTS]
MODEL, SEASON = "ConvNeXtTiny-TFT-NN", "summer"

def seed_mean(pattern):
    fs = [os.path.join(d, "preds_test_topk.csv") for d in sorted(glob.glob(pattern))]
    fs = [f for f in fs if os.path.exists(f)]
    ens = pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag", as_index=False).mean()
    return ens.sort_values("tag").reset_index(drop=True), len(fs)

def export(ens, region, root):
    d = f"{root}/{SEASON}/{region}"; os.makedirs(d, exist_ok=True)
    for tgt in ("anchor", "soft"):
        f = f"{d}/{MODEL}__{tgt}.csv"
        if os.path.exists(f) and not os.path.exists(f.replace(".csv", ".superseded_blend.csv")):
            shutil.copy(f, f.replace(".csv", ".superseded_blend.csv"))
    base = dict(date=ens.tag, region=region, season=SEASON, model=MODEL, split="test")
    order = ["sample_id", "date", "region", "season", "target", "model", "split",
             "actual", "predicted", "residual", "abs_error", "sq_error"]
    a = pd.DataFrame(base); a["sample_id"] = region + f"_{SEASON}_" + ens.tag.astype(str); a["target"] = "anchor"
    a["actual"] = ens.true_m1_hot.round(6); a["predicted"] = ens.pred_m1_hot.round(6)
    a["residual"] = (a.predicted - a.actual).round(6); a["abs_error"] = a.residual.abs().round(6); a["sq_error"] = (a.residual ** 2).round(6)
    a[order].to_csv(f"{d}/{MODEL}__anchor.csv", index=False)
    s = pd.DataFrame(base); s["sample_id"] = a["sample_id"]; s["target"] = "soft"
    s["actual"] = ens.true_m1_p3.round(6); s["predicted"] = ens.pred_m1_p3.round(6)
    s["residual"] = (s.predicted - s.actual).round(6); s["abs_error"] = s.residual.abs().round(6); s["sq_error"] = (s.residual ** 2).round(6)
    for k in (3, 7, 15):
        s[f"actual_p{k}"] = ens[f"true_m1_p{k}"].round(6); s[f"predicted_p{k}"] = ens[f"pred_m1_p{k}"].round(6)
    s[order + [c for k in (3, 7, 15) for c in (f"actual_p{k}", f"predicted_p{k}")]].to_csv(f"{d}/{MODEL}__soft.csv", index=False)

for region, pat in PICK.items():
    ens, n = seed_mean(pat)
    assert n == 5, (region, n)
    for root in OUTS:
        export(ens, region, root)
    e = ens.pred_m1_hot - ens.true_m1_hot
    es = np.concatenate([(ens[f"pred_m1_p{k}"] - ens[f"true_m1_p{k}"]).to_numpy() for k in (3, 7, 15)])
    print(f"{region:10s} n_seeds={n} rows={len(ens)} | anchor MAE {e.abs().mean():.3f} RMSE {np.sqrt((e**2).mean()):.3f} UPE {np.maximum(-e,0).mean():.3f} | soft MAE {np.abs(es).mean():.3f} RMSE {np.sqrt((es**2).mean()):.3f}")
print("written to:", *OUTS, sep="\n  ")
