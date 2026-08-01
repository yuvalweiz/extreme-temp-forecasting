"""Persist the daily station-value matrix (D, 68, 9) reconstructed from the stationvec
samples (first-covering-window resolution for per-window-imputed entries — same rule as
framebank_corr). ~18 MB. Also stores the date index and train-period feature stats."""
import glob
import os
import numpy as np
import pandas as pd

SV = "/home/weizyuv/stationvec_Center"
OUT = "/home/weizyuv/expreal/nninterp"
os.makedirs(OUT, exist_ok=True)

daily = {}
for f in sorted(glob.glob(os.path.join(SV, "sample_*.npz"))):
    tag = pd.Timestamp(os.path.basename(f)[7:17])
    X = np.load(f)["X"]
    days = pd.date_range(end=tag - pd.Timedelta(days=1), periods=X.shape[0], freq="D")
    for d, v in zip(days, X):
        if d not in daily:
            daily[d] = v.astype(np.float32)
dates = pd.DatetimeIndex(sorted(daily))
full = pd.date_range(dates[0], dates[-1], freq="D")
M = np.full((len(full), 68, 9), np.nan, np.float32)
for i, d in enumerate(full):
    if d in daily:
        M[i] = daily[d].reshape(68, 9)
train_mask = full <= pd.Timestamp("2019-01-28")
mu = np.nanmean(M[train_mask].reshape(-1, 9), axis=0)
sd = np.nanstd(M[train_mask].reshape(-1, 9), axis=0)
np.savez(os.path.join(OUT, "daily_sv.npz"), M=M,
         dates=np.array([d.strftime("%Y-%m-%d") for d in full]),
         feat_mu=mu.astype(np.float32), feat_sd=sd.astype(np.float32),
         train_end="2019-01-28")
print(f"saved {M.shape}, nan days: {int(np.isnan(M[:,0,0]).sum())}, mu[:3]={mu[:3].round(2)}")
