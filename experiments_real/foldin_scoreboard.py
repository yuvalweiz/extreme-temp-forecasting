"""FULL fold-in scoreboard: every model x region x season x target, full-set MAE/RMSE.
Sources:
  - corrected AVP (ours 5-seed picks, ConvNeXtTiny-LSTM hlst/clst, Tab-LSTM atl/catl)
    -> metrics_summary_corrected.csv (run foldin_avp.py first)
  - published AVP -> metrics_summary.csv (classical + foundation + published deep, for reference)
  - Tab-TFT = atft/catft ensembles computed directly from run dirs (5 seeds if extension landed)
Writes: article /results/FOLDIN_SCOREBOARD.txt
"""
import glob
import os

import numpy as np
import pandas as pd

AVP_C = "/home/weizyuv/article /results/actual_vs_predicted_corrected"
AVP_O = "/home/weizyuv/article /results/actual_vs_predicted"
OUT = "/home/weizyuv/article /results/FOLDIN_SCOREBOARD.txt"
COLS = ["tag", "true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15",
        "pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15"]
REGIONS = [("Center", "Ce"), ("Negev", "Ne"), ("Northwest", "No")]


def ens(pat, tk):
    fs = sorted(glob.glob(os.path.join(pat, f"preds_test{'_topk' if tk else ''}.csv")))
    if not fs:
        return None, 0
    return (pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag", as_index=False).mean(), len(fs))


def mets(d):
    e = (d["pred_m1_hot"] - d["true_m1_hot"]).to_numpy(float)
    err = (d[[f"pred_m1_{s}" for s in ["p3", "p7", "p15"]]].to_numpy(float)
           - d[[f"true_m1_{s}" for s in ["p3", "p7", "p15"]]].to_numpy(float))
    return dict(anchor_mae=np.abs(e).mean(), anchor_rmse=np.sqrt((e ** 2).mean()),
                soft_mae=np.abs(err).mean(), soft_rmse=np.sqrt((err ** 2).mean()))


def main():
    lines = []
    P = lines.append
    smc = pd.read_csv(os.path.join(AVP_C, "metrics_summary_corrected.csv"))
    smo = pd.read_csv(os.path.join(AVP_O, "metrics_summary.csv"))

    tft_tab = {}
    for reg, S in REGIONS:
        d, n = ens(f"/home/weizyuv/expreal/{reg}_stable/atft__*", 1)
        if d is not None:
            tft_tab[("summer", reg)] = (mets(d), n)
        d, n = ens(f"/home/weizyuv/expreal/{reg}_cold/{S}_catft__*", 0)
        if d is not None:
            tft_tab[("winter", reg)] = (mets(d), n)

    def cell(df, season, region, target, model, col):
        r = df[(df.season == season) & (df.region == region) & (df.target == target) & (df.model == model)]
        return float(r.iloc[0][col]) if len(r) else np.nan

    P("=" * 100)
    P("FOLD-IN SCOREBOARD — full-test-set MAE / RMSE (per region + 3-region mean)")
    P("OURS + LSTM-head + Tab-LSTM = corrected 5-seed protocol; Tab-TFT = atft/catft ensembles;")
    P("classical/foundation = deterministic published preds. (pub) rows = published single-checkpoint, for reference.")
    P("=" * 100)
    for season in ["summer", "winter"]:
        for target in ["anchor", "soft"]:
            P(f"\n### {season.upper()} {target.upper()}  (MAE | RMSE)")
            hdr = f"{'model':22s}" + "".join(f"{r:>16s}" for r, _ in REGIONS) + f"{'MEAN':>16s}"
            P(hdr)
            rows = [("OURS ConvNeXtTiny-TFT", smc, "ConvNeXtTiny-TFT"),
                    ("ConvNeXtTiny-LSTM 5s", smc, "ConvNeXtTiny-LSTM"),
                    ("Tab-LSTM 5s", smc, "Tab-LSTM"),
                    ("Prophet", smo, "Prophet"),
                    ("SARIMAX", smo, "SARIMAX"),
                    ("Tab-LSTM (pub 1-run)", smo, "Tab-LSTM"),
                    ("TFT (pub 1-ckpt)", smo, "ConvNeXtTiny-TFT")]
            for label, df, model in rows:
                vals = []
                for r, _ in REGIONS:
                    m = cell(df, season, r, target, model, "mae"); q = cell(df, season, r, target, model, "rmse")
                    vals.append(f"{m:6.3f}|{q:6.3f}" if np.isfinite(m) else "     --      ")
                mm = cell(df, season, "MEAN-3REGIONS", target, model, "mae")
                qq = cell(df, season, "MEAN-3REGIONS", target, model, "rmse")
                vals.append(f"{mm:6.3f}|{qq:6.3f}" if np.isfinite(mm) else "     --      ")
                P(f"{label:22s}" + "".join(f"{v:>16s}" for v in vals))
            vals, ms, qs = [], [], []
            ncounts = []
            for r, _ in REGIONS:
                got = tft_tab.get((season, r))
                if got:
                    mm = got[0][f"{target}_mae"]; qq = got[0][f"{target}_rmse"]
                    ms.append(mm); qs.append(qq); ncounts.append(got[1])
                    vals.append(f"{mm:6.3f}|{qq:6.3f}")
                else:
                    vals.append("     --      ")
            vals.append(f"{np.mean(ms):6.3f}|{np.mean(qs):6.3f}" if len(ms) == 3 else "     --      ")
            P(f"{'Tab-TFT (atft/catft)':22s}" + "".join(f"{v:>16s}" for v in vals) +
              f"   [seeds: {ncounts}]")
    txt = "\n".join(lines)
    open(OUT, "w").write(txt)
    print(txt)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
