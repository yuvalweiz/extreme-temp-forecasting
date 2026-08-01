"""
Lapse-rate frame fix, done memory-wise (no frame regeneration, no big arrays).

Because horizontal interpolation is linear, the physically-correct lapse-rate frame is:
    lapse_frame = EXP_frame + Γ·(interp_elev − pixel_elev)
where interp_elev = horizontal-distance interpolation of STATION elevations (same kernel
as the temperature frames), pixel_elev = grid elevation, Γ = lapse rate (°C per m).
So we only need ONE static (44,137) correction map + new normalization stats. Applied at
load time -> the model sees true lapse-corrected frames; nothing large is written.

Saves: repo/src/pipeline/lapse_correction.npy (44,137) and lapse_norm.npz (x_mean/x_std).
env: REGION (norm-stats are region-specific via that region's TRAIN samples), GAMMA, KEXP
"""
import os, sys, glob
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                                         # <repo>/src
import repo_paths as RP
from pipeline.config import REGION_DATASET
from pipeline.data import load_split, _abspath, load_norm_stats

GRID = RP.grid_metadata_npz()               # bundled: data/grid_metadata.npz
OUT = _HERE                                 # writes next to this file (small static assets)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def main():
    region = os.environ.get("REGION", "Center")
    gamma = float(os.environ.get("GAMMA", "0.0065"))   # °C per meter (6.5 °C/km)
    kexp = float(os.environ.get("KEXP", "0.2"))         # matches the EXP horizontal kernel
    g = np.load(GRID, allow_pickle=True)
    elev_grid = g["elev_grid"].astype(np.float64)       # (44,137) meters
    glat, glon = g["lat_grid"].astype(np.float64), g["lon_grid"].astype(np.float64)
    slat, slon, selev = (g["station_lat"].astype(np.float64), g["station_lon"].astype(np.float64),
                         g["station_elev"].astype(np.float64))
    H, W = elev_grid.shape; S = len(selev)

    # interp_elev: horizontal-distance interpolation of station elevations onto the grid
    interp_elev = np.zeros((H, W), np.float64)
    for i in range(H):
        d = haversine_km(glat[i][:, None], glon[i][:, None], slat[None, :], slon[None, :])  # (W,S)
        w = np.exp(-kexp * d); w /= (w.sum(1, keepdims=True) + 1e-12)
        interp_elev[i] = (w * selev[None, :]).sum(1)
    correction = (gamma * (interp_elev - elev_grid)).astype(np.float32)   # (44,137), °C
    np.save(f"{OUT}/lapse_correction.npy", correction)
    print(f"[lapse] correction map: mean={correction.mean():.2f}°C range=[{correction.min():.2f},{correction.max():.2f}] "
          f"(Γ={gamma*1000:.1f}°C/km, kexp={kexp})", flush=True)
    print(f"[lapse] high-elev pixels get cooled, low warmed -> real elevation signal (vs inert kernel)", flush=True)

    # new norm stats: stream TRAIN samples (one at a time -> memory-light), add correction
    ds = REGION_DATASET[region]
    ns = load_norm_stats(ds)
    df = load_split(ds, "train").iloc[::4]; C = ns["C"]                 # subsample (~200) -> fast, accurate mean/std
    xsum = np.zeros(C, np.float64); xsq = np.zeros(C, np.float64); n = 0
    corr = correction[None, None]                                       # (1,1,H,W) broadcast over T,C
    for i, (_, row) in enumerate(df.iterrows()):
        p = _abspath(ds, row["path"])
        if not os.path.exists(p):
            continue
        X = np.load(p)["X"].astype(np.float32) + corr                   # (T,C,H,W) lapse-corrected
        xf = X.astype(np.float64)
        xsum += xf.sum((0, 2, 3)); xsq += (xf**2).sum((0, 2, 3)); n += xf[:, 0].size
        if i % 40 == 0: print(f"  norm-stats {i}/{len(df)}", flush=True)
    xmean = (xsum/n).astype(np.float32)
    xstd = np.sqrt(np.maximum(xsq/n - (xsum/n)**2, 1e-8)).astype(np.float32)
    np.savez(f"{OUT}/lapse_norm.npz", x_mean=xmean, x_std=xstd)
    print(f"[lapse] new norm stats (region={region}): x_mean[maxdry]={xmean[7]:.2f} (orig {ns['x_mean'][7]:.2f}); saved.", flush=True)
    print(f"[DONE]", flush=True)


if __name__ == "__main__":
    main()
