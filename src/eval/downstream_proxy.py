"""
Downstream impact proxy (answers R1.2 / R3.3 / R3.7 / meta: "quantify how the
MAE gain translates to demand/reserve planning").

Idea (deliberately simple + clearly caveated): peak cooling load rises ~linearly
with temperature above a comfort base via cooling degree-days (CDD). A
temperature-forecast error of ΔT°C on the month's hottest day therefore maps to a
peak-load-forecast error of ~ s·ΔT (MW), where s is the grid's load-temperature
sensitivity. Reserve margins are sized to peak-load forecast uncertainty, so a
lower temperature MAE directly lowers the required reserve buffer.

We report the PROPORTIONAL reduction (slope-independent) plus an illustrative MW
figure. This is a planning proxy, not a calibrated demand model.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP

PREDS = RP.published_preds("Summer")
TFM = RP.results_dir()          # timesfm_<Region>.csv live in repo/results
REGIONS = ["Center", "Negev", "Northwest"]
T_BASE = 24.0          # cooling base temperature (°C), illustrative for Israel
SENS_MW_PER_C = 300.0  # illustrative grid load-temperature sensitivity (MW/°C)
OUT = RP.results_dir()


def load_region(region):
    d = pd.read_csv(os.path.join(PREDS, region, f"preds_hot_{region}.csv"))
    d["tag"] = pd.to_datetime(d["tag"])
    out = d[["tag", "true_m1_hot"]].rename(columns={"true_m1_hot": "true"})
    out["ours_tft"] = d["pred_temporalfusion_hot"].values
    for fn, col, name in [(f"preds_tab_lstm_hot_israel_wide_{region}.csv", "pred_m1_hot", "tab_lstm"),
                          (f"preds_sarimax_{region}_test_wide.csv", "pred_m1_hot", "sarimax"),
                          (f"preds_prophet_{region}.csv", "pred_m1_hot", "prophet")]:
        fp = os.path.join(PREDS, region, fn)
        if os.path.exists(fp):
            t = pd.read_csv(fp)
            t["tag"] = pd.to_datetime(t["tag"])
            out = out.merge(t[["tag", col]].rename(columns={col: name}), on="tag", how="left")
    tf = os.path.join(TFM, f"timesfm_{region}.csv")
    if os.path.exists(tf):
        t = pd.read_csv(tf); t["tag"] = pd.to_datetime(t["tag"])
        out = out.merge(t[["tag", "pred_m1_hot"]].rename(columns={"pred_m1_hot": "timesfm"}), on="tag", how="left")
    return out


def cdd(temp):
    return np.maximum(0.0, np.asarray(temp, float) - T_BASE)


def main():
    rows = []
    for r in REGIONS:
        d = load_region(r)
        true_load = cdd(d["true"]) * SENS_MW_PER_C
        for m in ["ours_tft", "tab_lstm", "sarimax", "prophet", "timesfm"]:
            if m not in d.columns:
                continue
            mask = d[m].notna().values
            temp_mae = float(np.abs(d.loc[mask, m].values - d.loc[mask, "true"].values).mean())
            pred_load = cdd(d.loc[mask, m].values) * SENS_MW_PER_C
            load_mae_mw = float(np.abs(pred_load - true_load[mask]).mean())
            rows.append(dict(region=r, model=m, temp_mae=temp_mae, peak_load_mae_MW=load_mae_mw))
    df = pd.DataFrame(rows)
    avg = df.groupby("model").agg(temp_mae=("temp_mae", "mean"),
                                  peak_load_mae_MW=("peak_load_mae_MW", "mean")).reset_index()
    order = ["ours_tft", "tab_lstm", "sarimax", "prophet", "timesfm"]
    avg = avg.set_index("model").reindex([m for m in order if m in avg.model.values]).reset_index()
    ours = avg.loc[avg.model == "ours_tft", "peak_load_mae_MW"].iloc[0]
    avg["MW_saved_vs_ours"] = avg["peak_load_mae_MW"] - ours
    avg["pct_reduction_vs_ours"] = 100 * avg["MW_saved_vs_ours"] / avg["peak_load_mae_MW"]
    os.makedirs(OUT, exist_ok=True)
    avg.round(1).to_csv(os.path.join(OUT, "downstream_proxy.csv"), index=False)
    print(f"=== Downstream peak-load proxy (CDD base={T_BASE}°C, s={SENS_MW_PER_C} MW/°C, avg 3 regions) ===")
    print("  proportional reduction is slope-independent; MW figures are illustrative\n")
    print(avg.round(1).to_string(index=False))
    print(f"\n  -> our hottest-day temp MAE translates to ~{ours:.0f} MW peak-load-forecast error;")
    for _, r in avg.iterrows():
        if r.model != "ours_tft":
            print(f"     vs {r.model}: we cut peak-load-forecast error by ~{r.MW_saved_vs_ours:.0f} MW "
                  f"({r.pct_reduction_vs_ours:.0f}% less) -> proportionally smaller reserve buffer.")


if __name__ == "__main__":
    main()
