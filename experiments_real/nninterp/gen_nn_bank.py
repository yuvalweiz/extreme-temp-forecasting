"""Generate the learned-interpolation frame bank: for every day and feature, query the
trained NNInterp at all 6028 grid pixels. Output mirrors framebank_corr/ layout
(bank.npy (D,9,44,137) float32 + dates.txt + norm_stats_x.npz) so the forecaster's
FRAMEBANK path consumes it directly. ~1.6 GB; minutes on the 4090."""
import os
import numpy as np
import pandas as pd
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/home/weizyuv/expreal/framebank_nn"
os.makedirs(OUT, exist_ok=True)
D = np.load("/home/weizyuv/expreal/nninterp/daily_sv.npz", allow_pickle=True)
M, dates = D["M"], D["dates"].astype(str)
MU, SD = D["feat_mu"], D["feat_sd"]
G = np.load("/home/weizyuv/article /repo/data/grid_metadata.npz", allow_pickle=True)
st_lat, st_lon, st_h = G["station_lat"], G["station_lon"], G["station_elev"]
lat_g, lon_g, elev_g = G["lat_grid"], G["lon_grid"], G["elev_grid"]

import sys
sys.path.insert(0, "/home/weizyuv/expreal/nninterp")
from train_nninterp import NNInterp, coords_feats  # reuses LAT/LON normalization

model = NNInterp().to(DEV)
model.load_state_dict(torch.load("/home/weizyuv/expreal/nninterp/nninterp.pt")["model"])
model.eval()

ST = coords_feats(st_lat, st_lon, st_h)                       # (68,3)
PIX = coords_feats(lat_g.ravel(), lon_g.ravel(), elev_g.ravel())  # (6028,3)
pix_t = torch.tensor(PIX, device=DEV)[None]                   # (1,6028,3)

bank = np.lib.format.open_memmap(os.path.join(OUT, "bank.npy"), mode="w+",
                                 dtype=np.float32, shape=(len(dates), 9, 44, 137))
with torch.no_grad():
    for di in range(len(dates)):
        for f in range(9):
            vals = (M[di, :, f] - MU[f]) / SD[f]
            ctx = torch.tensor(np.concatenate([ST, vals[:, None]], -1)[None], device=DEV)
            p = model(ctx, pix_t, torch.tensor([f], device=DEV)).squeeze(0).cpu().numpy()
            bank[di, f] = (p * SD[f] + MU[f]).reshape(44, 137)
        if di % 500 == 0:
            print(f"{di}/{len(dates)}", flush=True)
bank.flush()
with open(os.path.join(OUT, "dates.txt"), "w") as fh:
    fh.write("\n".join(dates))
# train-period x stats (same convention as framebank_corr)
cut = dates <= "2019-01-28"
idx = np.where(cut)[0]
s1 = np.zeros(9); s2 = np.zeros(9); n = 0
for i in idx:
    x = np.asarray(bank[i], dtype=np.float64)
    s1 += x.mean(axis=(1, 2)); s2 += (x ** 2).mean(axis=(1, 2)); n += 1
mean = s1 / n
std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8))
np.savez(os.path.join(OUT, "norm_stats_x.npz"),
         x_mean=mean.astype(np.float32), x_std=std.astype(np.float32))
print(f"done: {bank.shape}; stats mean[:3]={mean[:3].round(2)}")
