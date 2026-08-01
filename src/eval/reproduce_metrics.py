"""
Reproduce the paper's summer metric table from the prediction CSVs and verify
it matches Models Evaluations/.../summer_metrics_avg_across_clusters.csv.

Run:  python reproduce_metrics.py
Writes: repo/results/summer_metrics_by_region.csv
        repo/results/summer_metrics_avg.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP
import eval_lib as E

PREDS = RP.published_preds("Summer")
REF = os.path.join(PREDS, "_outputs", "summer_metrics_avg_across_clusters.csv")
DATASETS = {r: RP.eval_dataset(r) for r in RP.REGIONS}
REGIONS = list(DATASETS)
OUT = RP.results_dir()


def main():
    thr = {r: E.train_extreme_threshold(DATASETS[r], q=E.EXTREME_Q) for r in REGIONS}
    print("[train p90 thresholds]", {r: round(v, 3) for r, v in thr.items()})

    rows = []
    for r in REGIONS:
        for target, loader, lossfn in [
            ("hot", E.load_region_hot, E.hot_losses),
            ("soft", E.load_region_soft, E.soft_losses),
        ]:
            for model, df in loader(PREDS, r).items():
                s = E.summarize(lossfn(df, thr[r]))
                rows.append(dict(region=r, target=target, model=model, **s))
    by_region = pd.DataFrame(rows)
    by_region.to_csv(os.path.join(OUT, "summer_metrics_by_region.csv"), index=False)

    # average across the 3 clusters (mean of per-region metrics)
    avg = (by_region.groupby(["target", "model"])
           .agg(n_regions=("region", "nunique"),
                mae_all=("mae_all", "mean"), rmse_all=("rmse_all", "mean"),
                mae_ext=("mae_ext", "mean"), rmse_ext=("rmse_ext", "mean"),
                avg_mae=("avg_mae", "mean"), avg_rmse=("avg_rmse", "mean"),
                n_all=("n_all", "sum"), n_ext=("n_ext", "sum"))
           .reset_index())
    avg.to_csv(os.path.join(OUT, "summer_metrics_avg.csv"), index=False)

    # ---- compare to reference ----
    ref = pd.read_csv(REF)
    ref["model"] = ref["model"].replace({
        "tab_lstm_hot_israel_wide": "tab_lstm",
        "tab_lstm_soft_israel_wide": "tab_lstm"})
    print("\n=== reproduced vs reference (mae_all / mae_ext / avg_mae) ===")
    maxerr = 0.0
    for _, a in avg.iterrows():
        rr = ref[(ref.target == a.target) & (ref.model == a.model)]
        if rr.empty:
            continue
        rr = rr.iloc[0]
        for col in ["mae_all", "mae_ext", "avg_mae"]:
            maxerr = max(maxerr, abs(a[col] - rr[col]))
        print(f"  {a.target:4s} {a.model:16s} "
              f"repro=({a.mae_all:.4f},{a.mae_ext:.4f},{a.avg_mae:.4f}) "
              f"ref=({rr.mae_all:.4f},{rr.mae_ext:.4f},{rr.avg_mae:.4f})")
    print(f"\n[VALIDATION] max abs diff vs reference = {maxerr:.4f} "
          f"({'PASS' if maxerr < 1e-2 else 'CHECK'})")


if __name__ == "__main__":
    main()
