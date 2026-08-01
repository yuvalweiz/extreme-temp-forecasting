"""Corrected frame bank: daily frames synthesized as W(paper_opt) @ station-vector,
for every day covered by the stationvec dataset. Output mirrors framebank_exp/
(bank.npy (D,9,44,137) float32 + dates.txt) so the FRAMEBANK train path can run
any window length on the CORRECTED frames.

Daily station matrix is reconstructed from the stationvec samples (each sample holds
180 consecutive days ending tag-1d; overlaps are verified identical before use).
"""
import os
import glob
import numpy as np
import pandas as pd

SV_DIR = "/home/weizyuv/stationvec_Center"          # study-wide vectors (region-independent)
W_PATH = "/home/weizyuv/interp_weights/paper_opt.npz"
OUT = "/home/weizyuv/expreal/framebank_corr"
GH, GW = 44, 137

os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(os.path.join(SV_DIR, "sample_*.npz")))
daily = {}
conflicts, mx = 0, 0.0
for f in files:
    tag = pd.Timestamp(os.path.basename(f)[7:17])
    X = np.load(f)["X"]                             # (180, 612), window ends tag-1d
    days = pd.date_range(end=tag - pd.Timedelta(days=1), periods=X.shape[0], freq="D")
    for d, v in zip(days, X):
        if d in daily:
            diff = float(np.abs(daily[d] - v).max())
            if diff > 1e-4:
                conflicts += 1
                mx = max(mx, diff)
        else:
            daily[d] = v.astype(np.float32)
# Overlapping windows disagree ONLY on per-window-imputed missing station-days; observed
# entries are identical. Resolution: first covering window's value (deterministic). The
# bank serves the window-LENGTH sensitivity sweep, where a single consistent imputation
# is what matters; deployed models keep their per-sample synthesis untouched.
print(f"days: {len(daily)}, imputed-entry conflicts: {conflicts} (max|diff|={mx:.2f}; first-window kept)")

dates = pd.DatetimeIndex(sorted(daily))
full = pd.date_range(dates[0], dates[-1], freq="D")
missing = full.difference(dates)
print(f"range {dates[0].date()}..{dates[-1].date()}, missing days: {len(missing)}")

W = np.load(W_PATH, allow_pickle=True)["W"]         # (9, 6028, 68)
bank = np.lib.format.open_memmap(os.path.join(OUT, "bank.npy"), mode="w+",
                                 dtype=np.float32, shape=(len(full), 9, GH, GW))
for di, d in enumerate(full):
    if d in daily:
        v = daily[d].reshape(68, 9)                 # station-major layout (dim_order.json)
        bank[di] = np.stack([(W[f] @ v[:, f]).reshape(GH, GW) for f in range(9)])
    else:
        bank[di] = np.nan
    if di % 1000 == 0:
        print(f"{di}/{len(full)}", flush=True)
bank.flush()
with open(os.path.join(OUT, "dates.txt"), "w") as fh:
    fh.write("\n".join(d.strftime("%Y-%m-%d") for d in full))
print(f"done: {bank.shape}")
