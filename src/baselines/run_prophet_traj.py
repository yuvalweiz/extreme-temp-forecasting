"""Prophet trajectory baseline (refit harness): per test prediction point, refit on the
strictly-past history and forecast the next 30 daily values; save the full trajectory
plus the order-stat reduction. Protocol per the manuscript: yearly+weekly seasonality
(daily off), training-only day-of-year climatology regressor; seasonality mode and
changepoint prior scale selected on the VALIDATION split.

Run:  REGION=Center [TARGET_COL=min_dry_temp] python run_prophet_traj.py
Out:  repo/results/prophet_traj_<REGION>[_min].csv
"""
import os
import sys
import logging
import numpy as np
import pandas as pd

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import repo_paths as RP
import series as S
from prophet import Prophet

REGION = os.environ.get("REGION", "Center")
TARGET = os.environ.get("TARGET_COL", "max_dry_temp")
HOTTEST = TARGET == "max_dry_temp"
OUT = RP.results_dir()
GRID = [(m, c) for m in ("additive", "multiplicative") for c in (0.01, 0.05, 0.25)]
VAL_STRIDE = 2   # selection on every 2nd validation point (fit cost); test uses all points


def order_stats(vals, hottest):
    s = np.sort(np.asarray(vals, float))
    return (s[::-1] if hottest else s)[[0, 2, 6, 14]]


def main():
    cd = S.build_cluster_daily(REGION, cols=[TARGET])
    y = cd[TARGET].interpolate(limit_direction="both")
    df = S.build_targets(cd[TARGET], hottest=HOTTEST)
    val, test = (df[df.split == s].reset_index(drop=True) for s in ("val", "test"))
    train_end = pd.Timestamp(df[df.split == "train"]["pred_point"].max())
    ytr = y.loc[:train_end]
    clim = ytr.groupby(ytr.index.dayofyear.where(ytr.index.dayofyear <= 365, 365)).mean()

    def clim_col(index):
        doy = index.dayofyear.where(index.dayofyear <= 365, 365)
        return clim.reindex(doy).to_numpy()

    def forecast_point(pp, mode, cps):
        hist = y.loc[:pp]
        dtr = pd.DataFrame({"ds": hist.index, "y": hist.to_numpy(), "clim": clim_col(hist.index)})
        m = Prophet(seasonality_mode=mode, changepoint_prior_scale=cps,
                    yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.add_regressor("clim")
        m.fit(dtr)
        fut_idx = pd.date_range(pp + pd.Timedelta(days=1), periods=30, freq="D")
        fut = pd.DataFrame({"ds": fut_idx, "clim": clim_col(fut_idx)})
        return m.predict(fut)["yhat"].to_numpy()

    best = None
    for mode, cps in GRID:
        errs = []
        for _, r in val.iloc[::VAL_STRIDE].iterrows():
            fc = forecast_point(pd.Timestamp(r["pred_point"]), mode, cps)
            pr = order_stats(fc, HOTTEST)
            tr = [r.true_m1_hot, r.true_m1_p3, r.true_m1_p7, r.true_m1_p15]
            errs.append(np.abs(np.array(pr) - np.array(tr)).mean())
        score = float(np.mean(errs))
        print(f"[grid] mode={mode} cps={cps}: val avg-stat MAE {score:.3f}", flush=True)
        if best is None or score < best[0]:
            best = (score, mode, cps)
    score, mode, cps = best
    print(f"[selected] mode={mode} cps={cps} (val {score:.3f})", flush=True)

    rows = []
    for _, r in test.iterrows():
        fc = forecast_point(pd.Timestamp(r["pred_point"]), mode, cps)
        p1, p3, p7, p15 = order_stats(fc, HOTTEST)
        row = {"tag": r["tag"], "pred_m1_hot": p1, "pred_m1_p3": p3, "pred_m1_p7": p7, "pred_m1_p15": p15}
        row.update({f"fc_d{i+1}": float(v) for i, v in enumerate(fc)})
        rows.append(row)
    pred = pd.DataFrame(rows).merge(test, on="tag")
    suffix = "" if HOTTEST else "_min"
    pred.to_csv(os.path.join(OUT, f"prophet_traj_{REGION}{suffix}.csv"), index=False)
    for c, t in [("pred_m1_hot", "true_m1_hot"), ("pred_m1_p3", "true_m1_p3"),
                 ("pred_m1_p7", "true_m1_p7"), ("pred_m1_p15", "true_m1_p15")]:
        print(f"  test MAE {c}: {float((pred[c]-pred[t]).abs().mean()):.3f}")


if __name__ == "__main__":
    main()
