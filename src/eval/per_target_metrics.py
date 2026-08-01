"""
A6 (answers R3/R4 "averaging is controversial"): report MAE/RMSE for EACH order
statistic p1,p3,p7,p15 SEPARATELY (not just the hot anchor and the soft mean),
per model, all-case and extreme-case, averaged across the 3 clusters.

Run: python per_target_metrics.py  ->  repo/results/summer_per_target.csv
Uses the same preds CSVs + train-p90 extreme mask as eval_lib (no leakage).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP
import eval_lib as E

PREDS = RP.published_preds("Summer")
DATASETS = {r: RP.eval_dataset(r) for r in RP.REGIONS}
REGIONS = list(DATASETS)
TARGETS = ["hot", "p3", "p7", "p15"]   # p1==hot
OUT = RP.results_dir()


def _read(fp):
    d = pd.read_csv(fp)
    if "tag" not in d.columns and "tag_dt" in d.columns:
        d = d.rename(columns={"tag_dt": "tag"})
    d["tag"] = pd.to_datetime(d["tag"], errors="coerce")
    return d.dropna(subset=["tag"])


def region_per_target(region, thr):
    """Return rows: model x target -> mae_all, mae_ext (one region)."""
    base = os.path.join(PREDS, region)
    out = []

    def add(model, df, true_cols, pred_cols):
        true_hot = df["true_m1_hot"].to_numpy(float)
        ext = true_hot >= thr
        for tgt, tc, pc in zip(TARGETS, true_cols, pred_cols):
            if tc not in df.columns or pc not in df.columns:
                continue
            e = np.abs(df[pc].to_numpy(float) - df[tc].to_numpy(float))
            out.append(dict(region=region, model=model, target=tgt,
                            mae_all=float(np.nanmean(e)),
                            mae_ext=float(np.nanmean(e[ext])) if ext.any() else np.nan))

    # deep model hot (p1) + soft (p3,p7,p15) come from two files
    fh = os.path.join(base, f"preds_hot_{region}.csv")
    fs = os.path.join(base, f"preds_soft_{region}.csv")
    if os.path.exists(fh) and os.path.exists(fs):
        dh, ds = _read(fh), _read(fs)
        for name in ["temporalfusion", "lstm"]:
            m = dh[["tag", "true_m1_hot", f"pred_{name}_hot"]].merge(
                ds[["tag", "true_m1_p3", "true_m1_p7", "true_m1_p15",
                    f"pred_{name}_soft_p3", f"pred_{name}_soft_p7", f"pred_{name}_soft_p15"]], on="tag")
            add(name, m, ["true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15"],
                [f"pred_{name}_hot", f"pred_{name}_soft_p3", f"pred_{name}_soft_p7", f"pred_{name}_soft_p15"])
    # classical + tabular: one wide file each with pred_m1_p1..p15
    for fn, name in [(f"preds_prophet_{region}.csv", "prophet"),
                     (f"preds_sarimax_{region}_test_wide.csv", "sarimax")]:
        fp = os.path.join(base, fn)
        if os.path.exists(fp):
            add(name, _read(fp), ["true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15"],
                ["pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15"])
    fp = os.path.join(base, f"preds_tab_lstm_hot_israel_wide_{region}.csv")
    fps = os.path.join(base, f"preds_tab_lstm_soft_israel_wide_{region}.csv")
    if os.path.exists(fp) and os.path.exists(fps):
        m = _read(fp)[["tag", "true_m1_hot", "pred_m1_hot"]].merge(
            _read(fps)[["tag", "true_m1_p3", "true_m1_p7", "true_m1_p15",
                        "pred_m1_p3", "pred_m1_p7", "pred_m1_p15"]], on="tag")
        add("tab_lstm", m, ["true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15"],
            ["pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15"])
    return out


def main():
    thr = {r: E.train_extreme_threshold(DATASETS[r], q=E.EXTREME_Q) for r in REGIONS}
    rows = []
    for r in REGIONS:
        rows += region_per_target(r, thr[r])
    df = pd.DataFrame(rows)
    avg = (df.groupby(["model", "target"]).agg(mae_all=("mae_all", "mean"),
                                               mae_ext=("mae_ext", "mean")).reset_index())
    os.makedirs(OUT, exist_ok=True)
    avg.to_csv(os.path.join(OUT, "summer_per_target.csv"), index=False)
    piv = avg.pivot(index="model", columns="target", values="mae_all")[TARGETS]
    order = ["temporalfusion", "lstm", "tab_lstm", "sarimax", "prophet"]
    piv = piv.reindex([m for m in order if m in piv.index])
    print("=== Summer per-target MAE (all-case), avg of 3 clusters ===")
    print("    columns p1(=hot)/p3/p7/p15 -> shows the aggregate hides per-target spread\n")
    print(piv.round(3).to_string())
    print(f"\n[OK] wrote {OUT}/summer_per_target.csv (all + extreme, per model/target)")


if __name__ == "__main__":
    main()
