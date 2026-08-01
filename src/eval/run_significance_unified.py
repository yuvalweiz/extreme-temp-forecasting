"""
Unified-protocol significance for the resubmission: per-region moving-block
bootstrap (B=10,000, block length 4, two-sided) on the FULL-evaluation-set
per-sample loss differentials of ConvNeXtTiny-TFT vs each baseline, with
Holm correction across the four baseline comparisons within each
(season, target, region) family. Both seasons; no extreme-subset metrics.

Soft losses follow the manuscript's MAE_3 definition (mean over the p3/p7/p15
absolute errors per sample) in BOTH seasons -- this matches the published
winter evaluation; the published summer soft significance used the legacy
abs-of-mean convention, so summer soft rows are recomputed here.

Consumes the row-level CSVs written by export_actual_vs_predicted.py
(per-sample abs_error), so metrics and significance share one code path.

Run:  PYTHONPATH=src python src/eval/run_significance_unified.py
Env:  AVP_OUT  actual_vs_predicted dir (default <repo>/results/actual_vs_predicted)
      N_BOOT   bootstrap resamples (default 10000)
Writes: <AVP_OUT>/significance_unified.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP
import eval_lib as E

AVP = os.environ.get("AVP_OUT") or os.path.join(RP.results_dir(), "actual_vs_predicted")
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
REGIONS = ["Center", "Negev", "Northwest"]
REF = "ConvNeXtTiny-TFT"
BASELINES = ["Prophet", "SARIMAX", "Tab-LSTM", "ConvNeXtTiny-LSTM"]


def losses(season, region, model, target):
    fp = os.path.join(AVP, season, region, f"{model}__{target}.csv")
    d = pd.read_csv(fp)
    return d[["sample_id", "abs_error"]]


def main():
    rows = []
    for season in ["summer", "winter"]:
        for target in ["anchor", "soft"]:
            pooled = {}
            for region in REGIONS:
                ref = losses(season, region, REF, target)
                for b in BASELINES:
                    m = ref.merge(losses(season, region, b, target),
                                  on="sample_id", suffixes=("_ref", "_base"))
                    diff = (m["abs_error_base"] - m["abs_error_ref"]).to_numpy(float)
                    md, lo, hi, p = E.moving_block_bootstrap(diff, 4, N_BOOT, seed=0)
                    rows.append(dict(season=season, target=target, region=region,
                                     baseline=b, n=len(diff), dMAE=md,
                                     ci=f"[{lo:.3f},{hi:.3f}]", p=p))
                    pooled.setdefault(b, []).append(diff)
            for b, ds in pooled.items():
                diff = np.concatenate(ds)
                md, lo, hi, p = E.moving_block_bootstrap(diff, 4, N_BOOT, seed=0)
                rows.append(dict(season=season, target=target, region="ALL",
                                 baseline=b, n=len(diff), dMAE=md,
                                 ci=f"[{lo:.3f},{hi:.3f}]", p=p))

    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for (s, t, r), g in df.groupby(["season", "target", "region"]):
        df.loc[g.index, "p_holm"] = E.holm_bonferroni(g["p"]).values
    df.to_csv(os.path.join(AVP, "significance_unified.csv"), index=False)

    with pd.option_context("display.width", 200):
        show = df[df.region != "ALL"].copy()
        show["sig"] = show["p_holm"] < 0.05
        print(show.to_string(index=False))
        print("\n=== paper-table summary (mean dMAE across regions; #sig regions of 3) ===")
        for (s, t, b), g in show.groupby(["season", "target", "baseline"]):
            print(f"  {s:6s} {t:6s} vs {b:18s} meanDMAE={g['dMAE'].mean():+.3f} "
                  f"sig={int(g['sig'].sum())}/3")
    print(f"\n[OK] wrote {os.path.join(AVP, 'significance_unified.csv')}")


if __name__ == "__main__":
    main()
