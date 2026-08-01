"""SARIMAX trajectory baseline (refit harness): per test prediction point, forecast the
next 30 daily values and save the full trajectory (fc_d1..fc_d30) plus the standard
order-stat reduction. Protocol per the manuscript: exogenous = training-only day-of-year
climatology (raw + first sine/cosine harmonics); (seasonal) orders and trend selected on
the VALIDATION split from a candidate grid; parameters fitted on the training period and
re-applied (Kalman filter re-run, no refit) to the strictly-past history of each
prediction point.

Run:  REGION=Center [TARGET_COL=min_dry_temp] python run_sarimax_traj.py
Out:  repo/results/sarimax_traj_<REGION>[_min].csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import repo_paths as RP
import series as S
from statsmodels.tsa.statespace.sarimax import SARIMAX

REGION = os.environ.get("REGION", "Center")
TARGET = os.environ.get("TARGET_COL", "max_dry_temp")
HOTTEST = TARGET == "max_dry_temp"
OUT = RP.results_dir()

GRID = [  # (order, seasonal_order, trend)
    ((1, 0, 1), (0, 0, 0, 0), "c"),
    ((2, 0, 1), (0, 0, 0, 0), "c"),
    ((1, 1, 1), (0, 0, 0, 0), "n"),
    ((2, 1, 2), (0, 0, 0, 0), "n"),
    ((1, 0, 1), (1, 0, 1, 7), "c"),
    ((2, 0, 1), (1, 0, 1, 7), "c"),
    ((1, 1, 1), (1, 0, 1, 7), "n"),
]


def exog_frame(index, clim):
    doy = index.dayofyear.where(index.dayofyear <= 365, 365)
    x = pd.DataFrame(index=index)
    x["clim"] = clim.reindex(doy).to_numpy()
    ang = 2 * np.pi * doy.to_numpy() / 365.0
    x["sin1"], x["cos1"] = np.sin(ang), np.cos(ang)
    return x


def order_stats(vals, hottest):
    s = np.sort(np.asarray(vals, float))
    return (s[::-1] if hottest else s)[[0, 2, 6, 14]]


def main():
    cd = S.build_cluster_daily(REGION, cols=[TARGET])
    y = cd[TARGET].interpolate(limit_direction="both")
    df = S.build_targets(cd[TARGET], hottest=HOTTEST)
    train, val, test = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))

    train_end = pd.Timestamp(train["pred_point"].max())
    ytr = y.loc[:train_end]
    clim = ytr.groupby(ytr.index.dayofyear.where(ytr.index.dayofyear <= 365, 365)).mean()

    def forecasts_for(points, params_fit):
        rows = []
        for _, r in points.iterrows():
            pp = pd.Timestamp(r["pred_point"])
            hist = y.loc[:pp]
            fut_idx = pd.date_range(pp + pd.Timedelta(days=1), periods=30, freq="D")
            res_pp = params_fit.apply(hist, exog=exog_frame(hist.index, clim))
            fc = res_pp.forecast(steps=30, exog=exog_frame(fut_idx, clim))
            rows.append((r["tag"], np.asarray(fc, float)))
        return rows

    # --- validation selection ---
    best = None
    for order, sorder, trend in GRID:
        try:
            fit = SARIMAX(ytr, exog=exog_frame(ytr.index, clim), order=order,
                          seasonal_order=sorder, trend=trend,
                          enforce_stationarity=False, enforce_invertibility=False
                          ).fit(disp=False, maxiter=200)
            errs = []
            for tag, fc in forecasts_for(val, fit):
                row = val[val.tag == tag].iloc[0]
                pr = order_stats(fc, HOTTEST)
                tr = [row.true_m1_hot, row.true_m1_p3, row.true_m1_p7, row.true_m1_p15]
                errs.append(np.abs(np.array(pr) - np.array(tr)).mean())
            score = float(np.mean(errs))
            print(f"[grid] {order}x{sorder} trend={trend}: val avg-stat MAE {score:.3f}", flush=True)
            if best is None or score < best[0]:
                best = (score, order, sorder, trend, fit)
        except Exception as e:
            print(f"[grid] {order}x{sorder} trend={trend}: FAILED ({e})", flush=True)
    score, order, sorder, trend, fit = best
    print(f"[selected] {order}x{sorder} trend={trend} (val {score:.3f})", flush=True)

    # --- test forecasts ---
    rows = []
    for tag, fc in forecasts_for(test, fit):
        p1, p3, p7, p15 = order_stats(fc, HOTTEST)
        row = {"tag": tag, "pred_m1_hot": p1, "pred_m1_p3": p3, "pred_m1_p7": p7, "pred_m1_p15": p15}
        row.update({f"fc_d{i+1}": float(v) for i, v in enumerate(fc)})
        rows.append(row)
    pred = pd.DataFrame(rows).merge(test, on="tag")
    suffix = "" if HOTTEST else "_min"
    pred.to_csv(os.path.join(OUT, f"sarimax_traj_{REGION}{suffix}.csv"), index=False)
    for c, t in [("pred_m1_hot", "true_m1_hot"), ("pred_m1_p3", "true_m1_p3"),
                 ("pred_m1_p7", "true_m1_p7"), ("pred_m1_p15", "true_m1_p15")]:
        print(f"  test MAE {c}: {float((pred[c]-pred[t]).abs().mean()):.3f}")


if __name__ == "__main__":
    main()
