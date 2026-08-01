"""Stack the canonical EXP daily frame bank into one memmap for long-window ablations.

Output: ~/expreal/framebank_exp/bank.npy  (D, 9, 44, 137) float32  (~1.6 GB)
        ~/expreal/framebank_exp/dates.txt (one YYYY-MM-DD per line, aligned to axis 0)

Feature order = features_order.txt of the canonical datasets (identical for all regions).
"""
import os
import numpy as np
import pandas as pd

BANK = ("/home/weizyuv/Deep Learning Preprocess/Cluster All_ISRAEL/"
        "Daily Aggregation EXP 05_01_2026_ALL_ISRAEL_HIGHRES")
FEATS = [l.strip() for l in open("/home/weizyuv/Deep Learning Models/Cluster Center/"
         "dataset_FULL_h180_next30_DOM_1_7_14_21_28/features_order.txt") if l.strip()]
OUT = "/home/weizyuv/expreal/framebank_exp"

os.makedirs(OUT, exist_ok=True)
days = pd.date_range("2005-01-01", "2025-06-30", freq="D")
h0 = np.load(os.path.join(BANK, f"exponential_{FEATS[0]}_2005-01-01.npy"))
bank = np.lib.format.open_memmap(os.path.join(OUT, "bank.npy"), mode="w+",
                                 dtype=np.float32, shape=(len(days), len(FEATS), *h0.shape))
missing = 0
for di, d in enumerate(days):
    ds = d.strftime("%Y-%m-%d")
    for ci, f in enumerate(FEATS):
        p = os.path.join(BANK, f"exponential_{f}_{ds}.npy")
        if os.path.exists(p):
            bank[di, ci] = np.load(p)
        else:
            bank[di, ci] = np.nan
            missing += 1
    if di % 1000 == 0:
        print(f"{di}/{len(days)}", flush=True)
bank.flush()
with open(os.path.join(OUT, "dates.txt"), "w") as fh:
    fh.write("\n".join(d.strftime("%Y-%m-%d") for d in days))
print(f"done: {bank.shape}, missing feature-days: {missing}")
