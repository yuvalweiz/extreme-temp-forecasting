"""Per-seed anchor-MAE spread of the deployed 5-seed ensembles (featured learned
pipeline and the per-station Tab-TFT ablation, both seasons, all regions).
Per-seed predictions come from the run dirs the released ensembles were averaged
from (gather_framingA SOURCES + winter Tab-TFT stfwin runs); truth joined from
the released AVP actuals so every number reconciles with the paper tables.
"""
import glob, os
import numpy as np
import pandas as pd

E = os.path.expanduser("~/expreal")
R = os.path.expanduser("~/article /repo/results/actual_vs_predicted_corrected").replace("~", os.path.expanduser("~"))
R = "/home/weizyuv/article /repo/results/actual_vs_predicted_corrected"

SOURCES = {
  ("ConvNeXtTiny-TFT-NN", "summer", "Center"):    [f"{E}/Center_tune/nnfs2std_a1b0__*"],
  ("ConvNeXtTiny-TFT-NN", "summer", "Negev"):     [f"{E}/Negev_stable/nnfFIN_std__*"],
  ("ConvNeXtTiny-TFT-NN", "summer", "Northwest"): [f"{E}/Northwest_stable/nnfFIN_std__*"],
  ("ConvNeXtTiny-TFT-NN", "winter", "Center"):    [f"{E}/Center_cold/nnfwin__*"],
  ("ConvNeXtTiny-TFT-NN", "winter", "Negev"):     [f"{E}/Negev_cold/nnfwin__*"],
  ("ConvNeXtTiny-TFT-NN", "winter", "Northwest"): [f"{E}/Northwest_cold/nnfwin__*"],
  ("Tab-TFT", "summer", "Center"):    [f"{E}/Center_tune/stfs2std_a1b0__*"],
  ("Tab-TFT", "summer", "Negev"):     [f"{E}/Negev_stable/stfSTDFIN__*"],
  ("Tab-TFT", "summer", "Northwest"): [f"{E}/Northwest_stable/stfSTDFIN__*"],
  ("Tab-TFT", "winter", "Center"):    [f"{E}/Center_cold/stfwin__*"],
  ("Tab-TFT", "winter", "Negev"):     [f"{E}/Negev_cold/stfwin__*"],
  ("Tab-TFT", "winter", "Northwest"): [f"{E}/Northwest_cold/stfwin__*"],
}

rows = []
for (model, season, reg), globs in SOURCES.items():
    avp_name = "ConvNeXtTiny-TFT-NN" if model.endswith("NN") else model
    avp = pd.read_csv(f"{R}/{season}/{reg}/{avp_name}__anchor.csv", parse_dates=["date"])
    truth = avp.set_index(avp["date"].dt.strftime("%Y-%m-%d"))["actual"]
    ens_mae = avp["abs_error"].mean()
    seed_maes = []
    for g in globs:
        for d in sorted(glob.glob(g)):
            f = os.path.join(d, "preds_test_topk.csv")
            if not os.path.exists(f):
                f = os.path.join(d, "preds_test.csv")
            if not os.path.exists(f):
                continue
            p = pd.read_csv(f)
            p["tag"] = p["tag"].astype(str).str[:10]
            p = p.set_index("tag")
            v = p["pred_m1_hot"]
            j = truth.to_frame("y").join(v.rename("p"), how="inner").dropna()
            seed_maes.append((j["p"] - j["y"]).abs().mean())
    sm = np.array(seed_maes)
    rows.append(dict(model=model, season=season, region=reg, n_seeds=len(sm),
                     seed_mean=sm.mean(), seed_std=sm.std(ddof=1), seed_min=sm.min(),
                     seed_max=sm.max(), ensemble_MAE=ens_mae,
                     ens_gain=sm.mean() - ens_mae))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))
print("\nseed_std range:", round(df.seed_std.min(),3), "-", round(df.seed_std.max(),3))
print("ensembling gain (mean seed MAE - ensemble MAE) range:",
      round(df.ens_gain.min(),3), "-", round(df.ens_gain.max(),3))
df.to_csv(f"{E}/seed_variance_report.csv", index=False)
