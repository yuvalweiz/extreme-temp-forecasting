"""Output-format program scoreboard: seed-ENSEMBLED (mean prediction across seeds, then
metric — the paper's ensembling protocol) avg-over-outputs MAE per family/format/region,
for any set of run tags. Usage: edit FAMILIES or pass tags via CLI later."""
import json
import glob
import sys
import numpy as np
import pandas as pd

COLS = ["tag", "pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15",
        "true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15"]
PRED = ["pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15"]
TRUE = ["true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15"]


def ens_metric(pattern, root="stable"):
    """Seed-ensemble the topk preds matching pattern; return (avg-over-outputs MAE, n_seeds, per-output)."""
    fs = []
    for d in sorted(glob.glob(pattern)):
        f = f"{d}/preds_test_topk.csv"
        import os
        if not os.path.exists(f):
            f = f"{d}/preds_test.csv"
        if os.path.exists(f):
            fs.append(f)
    if not fs:
        return None, 0, None
    ens = pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag", as_index=False).mean()
    E = np.abs(ens[PRED].to_numpy() - ens[TRUE].to_numpy())
    return float(E.mean()), len(fs), E.mean(0).round(3).tolist()


def table(fams, fmts=("mk4", "mkt"), regions=("Center", "Negev", "Northwest")):
    for fmt in fmts:
        print(f"\n=== {fmt} — SEED-ENSEMBLED avg-over-outputs MAE ===")
        print(f"{'family':12s} " + " ".join(f"{r[:6]:>8s}" for r in regions) + f" {'3-reg':>8s}")
        for fam in fams:
            row = []
            for reg in regions:
                v, n, _ = ens_metric(f"/home/weizyuv/expreal/{reg}_stable/{fam}_{fmt}__*")
                row.append(v if v is not None else np.nan)
            if all(np.isnan(x) for x in row):
                continue
            print(f"{fam:12s} " + " ".join(f"{x:8.3f}" if not np.isnan(x) else f"{'—':>8s}" for x in row)
                  + f" {np.nanmean(row):8.3f}")


if __name__ == "__main__":
    fams = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["ours", "olstm", "stl", "stf", "atft", "ourspm", "stlpm", "stfpm", "atftpm"]
    table(fams)
