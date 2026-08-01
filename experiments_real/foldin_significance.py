"""FOLD-IN significance: corrected ConvNeXtTiny-TFT (5-seed picks) vs the four baselines
(Prophet, SARIMAX from the published AVP exports; the NEW 5-seed Tab-LSTM and
ConvNeXtTiny-LSTM from the corrected AVP exports), per region + pooled, both seasons,
full test set, moving-block bootstrap (B=10,000, block 4, two-sided), Holm across the
four baselines within each (season, target, region).

Run AFTER foldin_avp.py.  Writes: <AVP_C>/significance_corrected.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/weizyuv/article /repo/src")
sys.path.insert(0, "/home/weizyuv/article /repo/src/eval")
import eval_lib as E

AVP_OLD = "/home/weizyuv/article /results/actual_vs_predicted"      # classical baselines
AVP_C = "/home/weizyuv/article /results/actual_vs_predicted_corrected"
REGIONS = ["Center", "Negev", "Northwest"]
REF = "ConvNeXtTiny-TFT"
BASE = [("Prophet", AVP_OLD), ("SARIMAX", AVP_OLD),
        ("Tab-LSTM", AVP_C), ("ConvNeXtTiny-LSTM", AVP_C)]
N_BOOT = int(os.environ.get("N_BOOT", "10000"))


def losses(root, season, region, model, target):
    fp = os.path.join(root, season, region, f"{model}__{target}.csv")
    d = pd.read_csv(fp)
    key = d["date"].astype(str)
    return pd.DataFrame({"key": key, "abs": d["abs_error"].astype(float)})


def main():
    rows = []
    for season in ["summer", "winter"]:
        for target in ["anchor", "soft"]:
            pooled = {}
            for region in REGIONS:
                ref = losses(AVP_C, season, region, REF, target)
                for bname, root in BASE:
                    b = losses(root, season, region, bname, target)
                    m = ref.merge(b, on="key", suffixes=("_r", "_b"))
                    diff = (m["abs_b"] - m["abs_r"]).to_numpy(float)
                    md, lo, hi, p = E.moving_block_bootstrap(diff, 4, N_BOOT, seed=0)
                    rows.append(dict(season=season, target=target, region=region,
                                     baseline=bname, n=len(diff), dMAE=md,
                                     ci=f"[{lo:.3f},{hi:.3f}]", p=p))
                    pooled.setdefault(bname, []).append(diff)
            for bname, ds in pooled.items():
                diff = np.concatenate(ds)
                md, lo, hi, p = E.moving_block_bootstrap(diff, 4, N_BOOT, seed=0)
                rows.append(dict(season=season, target=target, region="ALL",
                                 baseline=bname, n=len(diff), dMAE=md,
                                 ci=f"[{lo:.3f},{hi:.3f}]", p=p))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _, g in df.groupby(["season", "target", "region"]):
        df.loc[g.index, "p_holm"] = E.holm_bonferroni(g["p"]).values
    df.to_csv(os.path.join(AVP_C, "significance_corrected.csv"), index=False)
    with pd.option_context("display.width", 200):
        show = df[df.region != "ALL"].copy(); show["sig"] = show["p_holm"] < 0.05
        print(show.to_string(index=False))
        print("\n=== paper-table summary (mean per-region dMAE; #sig regions) ===")
        for (s, t, b), g in show.groupby(["season", "target", "baseline"]):
            print(f"  {s:6s} {t:6s} vs {b:18s} meanDMAE={g['dMAE'].mean():+.3f} sig={int(g['sig'].sum())}/3")
        print("\n=== pooled ===")
        print(df[df.region == "ALL"][["season", "target", "baseline", "n", "dMAE", "p", "p_holm"]].round(4).to_string(index=False))
    print(f"\n[OK] wrote {os.path.join(AVP_C, 'significance_corrected.csv')}")


if __name__ == "__main__":
    main()
