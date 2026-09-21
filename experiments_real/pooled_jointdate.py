"""Region-pooled significance with JOINT-DATE moving-block bootstrap: dates are the
resampling unit and each date carries all three regions' paired differences, so
cross-region correlation on the same date (0.4-0.8) is preserved instead of being
treated as independent information. Compares against the paper-style pooling
(concatenate regions) for every pooled claim made in the manuscript: classical
(Prophet, SARIMAX), seasonal naive, foundation (TimesFM, Moirai), and the tabular
ablations, on anchor and soft targets, both seasons. B=10,000, block length 4,
two-sided, seed 0 (paper protocol).
"""
import numpy as np
import pandas as pd
import os

RC = "/home/weizyuv/article /repo/results/actual_vs_predicted_corrected"
RO = "/home/weizyuv/article /repo/results/actual_vs_predicted"
RR = "/home/weizyuv/article /repo/results"
REGS = ["Center", "Northwest", "Negev"]
rng = np.random.default_rng(0)

def mbb_p(d, L=4, B=10000):
    d = np.asarray(d, float)
    n = len(d); nb = int(np.ceil(n / L)); obs = d.mean()
    st = rng.integers(0, n - L + 1, size=(B, nb))
    idx = (st[:, :, None] + np.arange(L)).reshape(B, -1)[:, :n]
    return float((np.abs((d - obs)[idx].mean(1)) >= abs(obs)).mean()), obs

def series(model, season, reg, target):
    """abs-error series indexed by date for a model."""
    if model in ("TimesFM", "Moirai"):
        stem = {"TimesFM": "timesfm", "Moirai": "moirai"}[model]
        suf = "" if season == "summer" else "_min"
        p = pd.read_csv(f"{RR}/{stem}_{reg}{suf}.csv").set_index("tag")
        t = pd.read_csv(f"{RC}/{season}/{reg}/ConvNeXtTiny-TFT-NN__{target}.csv", parse_dates=["date"])
        t["tag"] = t["date"].dt.strftime("%Y-%m-%d")
        t = t.set_index("tag")
        if target == "anchor":
            j = t[["actual"]].join(p["pred_m1_hot"], how="inner").dropna()
            return (j["pred_m1_hot"] - j["actual"]).abs()
        acts = t[["actual_p3", "actual_p7", "actual_p15"]]
        j = acts.join(p[["pred_m1_p3", "pred_m1_p7", "pred_m1_p15"]], how="inner").dropna()
        e = np.abs(j[["pred_m1_p3", "pred_m1_p7", "pred_m1_p15"]].values
                   - j[["actual_p3", "actual_p7", "actual_p15"]].values).mean(1)
        return pd.Series(e, index=j.index)
    base = RC if os.path.exists(f"{RC}/{season}/{reg}/{model}__{target}.csv") else RO
    d = pd.read_csv(f"{base}/{season}/{reg}/{model}__{target}.csv", parse_dates=["date"])
    d["tag"] = d["date"].dt.strftime("%Y-%m-%d")
    d = d.set_index("tag")
    if target == "anchor":
        return d["abs_error"]
    e = np.abs(d[["predicted_p3", "predicted_p7", "predicted_p15"]].values
               - d[["actual_p3", "actual_p7", "actual_p15"]].values).mean(1)
    return pd.Series(e, index=d.index)

OUR = "ConvNeXtTiny-TFT-NN"
BASELINES = ["Prophet", "SARIMAX", "SeasonalNaive", "TimesFM", "Moirai", "Tab-TFT", "Tab-LSTM"]

rows = []
for season in ["summer", "winter"]:
    for target in ["anchor", "soft"]:
        for base in BASELINES:
            D = []
            for reg in REGS:
                o = series(OUR, season, reg, target)
                b = series(base, season, reg, target)
                D.append((b - o).dropna().rename(reg))
            concat = np.concatenate([d.values for d in D])
            J = pd.concat(D, axis=1).dropna()
            joint = J.mean(1).values
            p_c, d_c = mbb_p(concat)
            p_j, d_j = mbb_p(joint)
            rows.append(dict(season=season, target=target, baseline=base,
                             dMAE=round(d_j, 3), p_concat=round(p_c, 4), p_jointdate=round(p_j, 4),
                             n_dates=len(J)))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
df.to_csv(os.path.expanduser("~/expreal/pooled_jointdate.csv"), index=False)
