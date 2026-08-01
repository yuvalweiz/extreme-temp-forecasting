"""
Export the complete row-level actual-vs-predicted data behind every reported
table cell, and recompute the paper's full-evaluation-set MAE / RMSE for all
region x model x target x season combinations.

This is the single script that regenerates:
  * one row-level CSV per (season, region, target, model) with the actual and
    predicted values, residuals, absolute and squared errors of every test
    sample used in the paper's evaluation;
  * metrics_summary.csv -- the final MAE / RMSE of every combination (plus the
    3-region means reported in the paper's aggregated tables);
  * a verification against the frozen published reference files
    (_outputs/{summer,winter}_metrics_*.csv), printed as max|diff| PASS/FAIL.

Evaluation-set definition (identical to the published protocol):
  * all chronologically held-out test samples of each region (no subsetting);
  * anchor target: absolute error on the monthly extreme day (p1);
  * soft target:   per-rank errors on (p3,p7,p15), aggregated per the paper's
                   MAE_3 / RMSE_3 definitions (mean over ranks of |err|, and
                   sqrt of the mean over ranks of err^2 -- "mean-of-abs").
                   NOTE: the originally published SUMMER soft cells used
                   abs-of-mean instead; the resubmission unifies both seasons
                   to the manuscript's printed MAE_3/RMSE_3 formulas, so the
                   summer soft cells are recomputed here (winter published
                   soft cells already used this convention and are unchanged).

Run:  PYTHONPATH=src python src/eval/export_actual_vs_predicted.py
Env:  AVP_OUT   output directory (default <repo>/results/actual_vs_predicted)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP
import eval_lib as E

REGIONS = ["Center", "Negev", "Northwest"]
SOFT = ["p3", "p7", "p15"]
PRETTY = {"temporalfusion": "ConvNeXtTiny-TFT", "lstm": "ConvNeXtTiny-LSTM",
          "sarimax": "SARIMAX", "prophet": "Prophet", "tab_lstm": "Tab-LSTM"}

OUT = os.environ.get("AVP_OUT") or os.path.join(RP.results_dir(), "actual_vs_predicted")


def _read(fp):
    df = pd.read_csv(fp)
    if "tag" not in df.columns and "tag_dt" in df.columns:
        df = df.rename(columns={"tag_dt": "tag"})
    df["tag"] = pd.to_datetime(df["tag"], errors="coerce")
    return df.dropna(subset=["tag"]).sort_values("tag").reset_index(drop=True)


def _find(base, *names):
    """Case-tolerant file lookup (winter SARIMAX files mix test/TEST)."""
    listing = {f.lower(): f for f in os.listdir(base)}
    for n in names:
        if n.lower() in listing:
            return os.path.join(base, listing[n.lower()])
    return None


# ---------------------------------------------------------------------------
# loaders: dict model -> (anchor_df[tag,true,pred], soft_df[tag,true_p*,pred_p*])
# ---------------------------------------------------------------------------
def load_season_region(season, region):
    """Returns (anchor: dict model->df[tag,true,pred],
                soft:   dict model->df[tag,true_p3..,pred_p3..])."""
    base = os.path.join(RP.published_preds(season), region)
    m1 = "hot" if season == "Summer" else "cold"
    anchor, soft = {}, {}

    # deep spatiotemporal models (ours)
    f = _find(base, f"preds_{m1}_{region}.csv")
    d = _read(f)
    for name in ["temporalfusion", "lstm"]:
        anchor[name] = d[["tag", f"true_m1_{m1}", f"pred_{name}_{m1}"]].rename(
            columns={f"true_m1_{m1}": "true", f"pred_{name}_{m1}": "pred"})
    fs = _find(base, f"preds_soft_{region}.csv")
    ds = _read(fs)
    for name in ["temporalfusion", "lstm"]:
        cols = {f"pred_{name}_soft_{s}": f"pred_{s}" for s in SOFT}
        keep = ["tag"] + [f"true_m1_{s}" for s in SOFT] + list(cols)
        soft[name] = ds[keep].rename(columns=cols)

    # classical + tabular baselines (anchor and soft ranks in one file each)
    if season == "Summer":
        cands = {"prophet": [f"preds_prophet_{region}.csv"],
                 "sarimax": [f"preds_sarimax_{region}_test_wide.csv"]}
        tab_anchor = [f"preds_tab_lstm_hot_israel_wide_{region}.csv"]
        tab_soft = [f"preds_tab_lstm_soft_israel_wide_{region}.csv"]
        tcol = "true_m1_hot"; pcol = "pred_m1_hot"
    else:
        cands = {"prophet": [f"preds_prophet_MIN_{region}.csv"],
                 "sarimax": [f"preds_sarimax_min_test_{region}.csv",
                             f"preds_sarimax_min_TEST_{region}.csv"]}
        tab_anchor = [f"preds_tab_lstm_main_israel_wide_WINTER_{region}.csv"]
        tab_soft = [f"preds_tab_lstm_soft_israel_wide_WINTER_{region}.csv"]
        tcol = "true_m1_cold"; pcol = "pred_m1_cold"

    for name, names in cands.items():
        fp = _find(base, *names)
        d = _read(fp)
        anchor[name] = d[["tag", tcol, pcol]].rename(columns={tcol: "true", pcol: "pred"})
        cols = {f"pred_m1_{s}": f"pred_{s}" for s in SOFT}
        keep = ["tag"] + [f"true_m1_{s}" for s in SOFT] + list(cols)
        soft[name] = d[keep].rename(columns=cols)

    fa = _find(base, *tab_anchor)
    d = _read(fa)
    tc = tcol if tcol in d.columns else "true_m1_main"
    pc = pcol if pcol in d.columns else "pred_m1_main"
    anchor["tab_lstm"] = d[["tag", tc, pc]].rename(columns={tc: "true", pc: "pred"})
    fso = _find(base, *tab_soft)
    d = _read(fso)
    cols = {f"pred_m1_{s}": f"pred_{s}" for s in SOFT}
    keep = ["tag"] + [f"true_m1_{s}" for s in SOFT] + list(cols)
    soft["tab_lstm"] = d[keep].rename(columns=cols)
    return anchor, soft


# ---------------------------------------------------------------------------
# row-level export + metrics
# ---------------------------------------------------------------------------
def rows_anchor(df, season, region, model):
    t = df["true"].astype(float); p = df["pred"].astype(float)
    res = p - t
    return pd.DataFrame({
        "sample_id": [f"{region}_{season.lower()}_{d:%Y-%m-%d}" for d in df["tag"]],
        "date": df["tag"].dt.strftime("%Y-%m-%d"),
        "region": region, "season": season.lower(), "target": "anchor",
        "model": PRETTY[model], "split": "test",
        "actual": t, "predicted": p, "residual": res,
        "abs_error": res.abs(), "sq_error": res ** 2})


def rows_soft(df, season, region, model):
    t = df[[f"true_m1_{s}" for s in SOFT]].astype(float).to_numpy()
    p = df[[f"pred_{s}" for s in SOFT]].astype(float).to_numpy()
    err = p - t
    # per-sample soft losses per the paper's MAE_3 / RMSE_3 definitions:
    # abs_error  = (1/3) sum_k |err_k|   (mean over samples  -> MAE_3)
    # sq_error   = (1/3) sum_k err_k^2   (sqrt of mean       -> RMSE_3)
    out = pd.DataFrame({
        "sample_id": [f"{region}_{season.lower()}_{d:%Y-%m-%d}" for d in df["tag"]],
        "date": df["tag"].dt.strftime("%Y-%m-%d"),
        "region": region, "season": season.lower(), "target": "soft",
        "model": PRETTY[model], "split": "test",
        "actual": t.mean(axis=1), "predicted": p.mean(axis=1),
        "residual": err.mean(axis=1),
        "abs_error": np.abs(err).mean(axis=1), "sq_error": (err ** 2).mean(axis=1)})
    for s, k in zip(SOFT, range(len(SOFT))):
        out[f"actual_{s}"] = t[:, k]
        out[f"predicted_{s}"] = p[:, k]
    return out


def metrics(rows):
    a = rows["abs_error"].to_numpy(float)
    m = np.isfinite(a)
    return dict(n=int(m.sum()), mae=float(np.mean(a[m])),
                rmse=float(np.sqrt(np.mean(a[m] ** 2))))


def main():
    os.makedirs(OUT, exist_ok=True)
    summary, problems = [], []
    for season in ["Summer", "Winter"]:
        for region in REGIONS:
            anchor, soft = load_season_region(season, region)
            odir = os.path.join(OUT, season.lower(), region)
            os.makedirs(odir, exist_ok=True)
            for target, data, mk in [("anchor", anchor, rows_anchor),
                                     ("soft", soft, rows_soft)]:
                for model, df in data.items():
                    if df["tag"].duplicated().any():
                        problems.append(f"DUPLICATE tags: {season}/{region}/{target}/{model}")
                    rows = mk(df, season, region, model)
                    fp = os.path.join(odir, f"{PRETTY[model]}__{target}.csv")
                    rows.to_csv(fp, index=False, float_format="%.6f")
                    rec = dict(season=season.lower(), region=region,
                               target=target, model=PRETTY[model],
                               source_csv=os.path.relpath(fp, OUT),
                               **metrics(rows))
                    if target == "soft":
                        # legacy summer convention (abs-of-mean), kept only to
                        # verify data integrity vs the published reference
                        r = rows["residual"].to_numpy(float)
                        rec["mae_legacy_absofmean"] = float(np.mean(np.abs(r)))
                        rec["rmse_legacy_absofmean"] = float(np.sqrt(np.mean(r ** 2)))
                    summary.append(rec)
    sm = pd.DataFrame(summary)

    # 3-region means (the aggregation used by the paper's region-averaged tables)
    agg = (sm.groupby(["season", "target", "model"], dropna=False)
             .agg(n_regions=("region", "nunique"), n=("n", "sum"),
                  mae=("mae", "mean"), rmse=("rmse", "mean"),
                  mae_legacy_absofmean=("mae_legacy_absofmean", "mean"),
                  rmse_legacy_absofmean=("rmse_legacy_absofmean", "mean")).reset_index())
    agg.insert(1, "region", "MEAN-3REGIONS")
    agg["source_csv"] = ""
    sm = pd.concat([sm, agg], ignore_index=True)
    sm.to_csv(os.path.join(OUT, "metrics_summary.csv"), index=False, float_format="%.6f")

    # ---- verification vs frozen published references ----
    print("=== verification vs published reference files ===")
    maxerr = 0.0
    for season, ref_name, m1 in [("summer", "summer_metrics_avg_across_clusters.csv", "hot"),
                                 ("winter", "winter_metrics_avg_across_clusters.csv", "cold")]:
        ref_fp = os.path.join(RP.published_preds(season.capitalize()), "_outputs", ref_name)
        if not os.path.exists(ref_fp):
            print(f"  [skip] no reference file {ref_fp}")
            continue
        ref = pd.read_csv(ref_fp)
        ref["model"] = (ref["model"].str.replace("_hot_israel_wide", "", regex=False)
                        .str.replace("_soft_israel_wide", "", regex=False)
                        .str.replace("_main_israel_wide", "", regex=False))
        for _, rr in ref.iterrows():
            target = "anchor" if rr["target"] == m1 else "soft"
            model = PRETTY.get(rr["model"], rr["model"])
            a = sm[(sm.season == season) & (sm.region == "MEAN-3REGIONS")
                   & (sm.target == target) & (sm.model == model)]
            if a.empty:
                problems.append(f"MISSING combo for reference row {season}/{target}/{model}")
                continue
            a = a.iloc[0]
            if season == "summer" and target == "soft":
                # published summer soft cells used the legacy abs-of-mean
                # convention -- verify data integrity against that, and print
                # the unified (mean-of-abs) value reported in the resubmission
                d = max(abs(a["mae_legacy_absofmean"] - rr["mae_all"]),
                        abs(a["rmse_legacy_absofmean"] - rr["rmse_all"]))
                maxerr = max(maxerr, d)
                print(f"  {season} {target:6s} {model:18s} "
                      f"legacy {a['mae_legacy_absofmean']:.4f}/{a['rmse_legacy_absofmean']:.4f} "
                      f"vs ref {rr['mae_all']:.4f}/{rr['rmse_all']:.4f} | max|d|={d:.6f} "
                      f"|| unified MAE3/RMSE3 = {a['mae']:.4f}/{a['rmse']:.4f}")
            else:
                d = max(abs(a["mae"] - rr["mae_all"]), abs(a["rmse"] - rr["rmse_all"]))
                maxerr = max(maxerr, d)
                print(f"  {season} {target:6s} {model:18s} "
                      f"mae {a['mae']:.4f} vs {rr['mae_all']:.4f} | "
                      f"rmse {a['rmse']:.4f} vs {rr['rmse_all']:.4f} | max|d|={d:.6f}")
    print(f"\n[VALIDATION] max abs diff vs published references = {maxerr:.6f} "
          f"({'PASS' if maxerr < 5e-4 else 'FAIL'})")
    for p in problems:
        print("[PROBLEM]", p)
    print(f"\nwrote {len(sm)} summary rows -> {os.path.join(OUT, 'metrics_summary.csv')}")


if __name__ == "__main__":
    main()
