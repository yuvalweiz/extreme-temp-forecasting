"""Reference (naive) forecasts recommended by Hewamalage et al. (DMKD 2023):
- Climatology: training-period day-of-year mean of the regional daily series.
- Seasonal naive: the observed regional series of the same calendar window one year earlier.
Both produce a 30-day daily trajectory per test sample, reduced to the paper's four
order-statistic targets by the same rank reduction as all trajectory baselines, and are
exported as row-level AVP files into results/actual_vs_predicted_corrected/.

Run:  python src/baselines/run_reference_forecasts.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import repo_paths as RP
import series as S

AC = os.path.join(RP.results_dir(), "actual_vs_predicted_corrected")
ORDER = ["sample_id", "date", "region", "season", "target", "model", "split",
         "actual", "predicted", "residual", "abs_error", "sq_error"]

def export(model, season, reg, R, pfx):
    d = os.path.join(AC, season, reg)
    R = R.copy()
    R["date"] = pd.to_datetime(R.tag).dt.strftime("%Y-%m-%d")
    for target, acol, pcol in [("anchor", "a1", f"{pfx}1"), ("soft", "a3", f"{pfx}3")]:
        t = pd.DataFrame({"date": R.date})
        t["sample_id"] = reg + f"_{season}_" + R.date
        t["region"], t["season"], t["target"], t["model"], t["split"] = reg, season, target, model, "test"
        t["actual"] = R[acol].round(6)
        t["predicted"] = R[pcol].round(6)
        t["residual"] = (t.predicted - t.actual).round(6)
        t["abs_error"] = t.residual.abs().round(6)
        t["sq_error"] = (t.residual ** 2).round(6)
        cols = list(ORDER)
        if target == "soft":
            for k in (3, 7, 15):
                t[f"actual_p{k}"] = R[f"a{k}"].round(6)
                t[f"predicted_p{k}"] = R[f"{pfx}{k}"].round(6)
                cols += [f"actual_p{k}", f"predicted_p{k}"]
        t.dropna(subset=["predicted"])[cols].to_csv(os.path.join(d, f"{model}__{target}.csv"), index=False)

for col, hot, season in [("max_dry_temp", True, "summer"), ("min_dry_temp", False, "winter")]:
    for reg in ["Center", "Negev", "Northwest"]:
        y = S.build_cluster_daily(reg, cols=[col])[col].dropna().sort_index()
        df = S.build_targets(y, hottest=hot)
        df["tag"] = pd.to_datetime(df["tag"])
        train_end = pd.Timestamp(df[df.split == "train"].pred_point.max())
        ytr = y.loc[:train_end]
        doy = ytr.index.dayofyear.where(ytr.index.dayofyear <= 365, 365)
        clim = ytr.groupby(doy).mean()
        avp = pd.read_csv(os.path.join(AC, season, reg, "ConvNeXtTiny-TFT-NN__anchor.csv"))
        avs = pd.read_csv(os.path.join(AC, season, reg, "ConvNeXtTiny-TFT-NN__soft.csv"))
        rows = []
        for tag, a1, a3, a7, a15 in zip(pd.to_datetime(avp.date), avp.actual,
                                        avs.actual_p3, avs.actual_p7, avs.actual_p15):
            win = pd.date_range(tag, periods=30, freq="D")
            d2 = pd.Series(win.dayofyear).where(pd.Series(win.dayofyear) <= 365, 365)
            cstats = S.pick_order_stats(pd.Series(clim.reindex(d2).to_numpy(), index=win), hottest=hot)
            nvals = y.reindex(win - pd.Timedelta(days=365))
            nstats = (S.pick_order_stats(pd.Series(nvals.to_numpy(), index=win), hottest=hot)
                      if nvals.notna().sum() >= 15 else (np.nan,) * 4)
            rows.append((tag, a1, a3, a7, a15) + cstats + nstats)
        R = pd.DataFrame(rows, columns=["tag", "a1", "a3", "a7", "a15",
                                        "c1", "c3", "c7", "c15", "n1", "n3", "n7", "n15"])
        export("Climatology", season, reg, R, "c")
        export("SeasonalNaive", season, reg, R, "n")
        print(f"{season}/{reg}: {len(R)} samples exported")
print("done")
