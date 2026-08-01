"""
P1 reproducibility: the dataset summary table the reviewers asked for
(R2/R4: dataset size, splits, #samples, #features). All pulled from the
on-disk artifacts (norm_stats + k_values + split CSVs), no leakage.

Run: python dataset_stats.py  ->  repo/results/dataset_stats.md
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                                        # <repo>/src
sys.path.insert(0, _HERE)
import repo_paths as RP
import eval_lib as E

KVAL = RP.k_values_csv()
# split CSVs / norm stats / features_order: canonical dataset if present, else the
# small bundled copies in data/dataset_meta/<Region>/ (identical files)
DATASETS = {r: RP.dataset_meta(r) for r in RP.REGIONS}
# per-sample y lookup (extreme-test count): canonical dataset if present, else the
# bundled station-vector dataset (identical y values)
SAMPLES = {r: RP.eval_dataset(r) for r in RP.REGIONS}
TAG = "yNONE_v1"
OUT = RP.results_dir()
KEEP_DOM = {1, 7, 14, 21, 28}


def split_n(d, split):
    df = pd.read_csv(os.path.join(d, f"split_{split}_{TAG}.csv"))
    df["tag"] = pd.to_datetime(df["tag"])
    return int(df[df["tag"].dt.day.isin(KEEP_DOM)].shape[0])


def main():
    os.makedirs(OUT, exist_ok=True)
    L = []
    L.append("# Dataset summary (for the reproducibility appendix)\n")
    L.append("**Source:** Israel Meteorological Service daily station observations, "
             "68 stations (union of 7 climate clusters), 2005-2025.\n")
    L.append("**Spatial frame:** study-wide 44x137 elevation-aware grid (SRTM DEM), "
             "shared across regions; only the target series is region-specific.\n")
    L.append("**Input window:** 180 days. **Forecast window:** 30 days. "
             "**Prediction days:** day-of-month in {1,7,14,21,28}.\n")
    L.append("**Targets:** order statistics p1/p3/p7/p15 = 1st/3rd/7th/15th hottest "
             "(coldest) day of the 30-day window.\n")

    # channels + R2 selection
    k = pd.read_csv(KVAL)
    L.append("\n## Feature channels (R^2 selection)\n")
    L.append("| feature | R^2 | k (elev/dist ratio) | used |")
    L.append("|---|---|---|---|")
    feats9 = open(os.path.join(DATASETS["Center"], "features_order.txt")).read().split()
    for _, r in k.iterrows():
        used = "yes" if r["Feature"] in feats9 else "no (dropped)"
        L.append(f"| {r['Feature']} | {r['R² Score']:.3f} | {r['k']:.2f} | {used} |")
    L.append(f"\n-> {len(feats9)} channels used: {', '.join(feats9)} (prs_stn dropped).\n")

    # splits per region
    L.append("\n## Chronological split (no leakage), #samples after DOM filter\n")
    L.append("| region | train | val | test | n_ext(test, p90) | train period | test period |")
    L.append("|---|---|---|---|---|---|---|")
    for reg, d in DATASETS.items():
        ns = np.load(os.path.join(d, f"norm_stats_extremes_full_{TAG}.npz"), allow_pickle=True)
        ntr, nva, nte = split_n(d, "train"), split_n(d, "val"), split_n(d, "test")
        thr = float(ns["hot_p90"]) if "hot_p90" in ns.files else float("nan")
        # extreme test count
        te = pd.read_csv(os.path.join(d, f"split_test_{TAG}.csv")); te["tag"] = pd.to_datetime(te["tag"])
        te = te[te["tag"].dt.day.isin(KEEP_DOM)]
        next_ = "n/a"
        try:
            sd = SAMPLES[reg]
            ys = [float(np.load(os.path.join(sd, os.path.basename(p)), allow_pickle=True)["y"][0])
                  if os.path.exists(os.path.join(sd, os.path.basename(p))) else np.nan
                  for p in te["path"]]
            if not np.isfinite(ys).any():          # bare git clone: bundled y table instead
                yv = E.load_y_values(sd, TAG)
                ys = yv.loc[yv["split"] == "test", "y_p1"].astype(float).tolist() if yv is not None else ys
            next_ = int(np.nansum(np.array(ys) >= thr))
        except Exception:
            pass
        tp = f"{str(ns['train_start'])[:10]}..{str(ns['train_end'])[:10]}" if "train_start" in ns.files else "?"
        sp = f"{str(ns['test_start'])[:10]}..{str(ns['test_end'])[:10]}" if "test_start" in ns.files else "?"
        L.append(f"| {reg} | {ntr} | {nva} | {nte} | {next_} | {tp} | {sp} |")

    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "dataset_stats.md"), "w").write(txt)
    print(txt)
    print(f"[OK] wrote {OUT}/dataset_stats.md")


if __name__ == "__main__":
    main()
