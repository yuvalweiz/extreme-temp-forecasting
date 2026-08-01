"""Build paper_opt_ktuned.npz: Eq.-d3d kernel with LOSO-tuned per-feature (k_scale, gamma)
from ~/expreal/ktune_results.csv (train-period-only tuning). Mirrors compute_interp_weights."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/weizyuv/artsrc")
sys.path.insert(0, "/home/weizyuv/article /repo/src/data")   # symlink breaks CW's _REPO logic
import pipeline.compute_interp_weights as CW
_R = "/home/weizyuv/article /repo"                            # fix symlink-derived paths
CW.DIM_ORDER = f"{_R}/data/stationvec_Center/dim_order.json"
CW.FEATS = "/home/weizyuv/Deep Learning Models/Cluster Center/dataset_FULL_h180_next30_DOM_1_7_14_21_28/features_order.txt"
CW.OUT_DIR = f"{_R}/data/interp_weights"

tune = pd.read_csv("/home/weizyuv/expreal/ktune_results.csv").set_index("feature")
g = np.load(CW.GRID, allow_pickle=True)
lat_g, lon_g, elev_g = g["lat_grid"], g["lon_grid"], g["elev_grid"]
grid_names = [str(s) for s in g["station_names"]]
order = json.load(open(CW.DIM_ORDER))["station_order"]
ix = np.array([grid_names.index(s) for s in order])
st_lat, st_lon, st_h = (g[k].astype(np.float64)[ix] for k in ("station_lat","station_lon","station_elev"))
feats = open(CW.FEATS).read().split()
kf = dict(zip(pd.read_csv(CW.KVAL)["Feature"], pd.read_csv(CW.KVAL)["k"]))
kf_scaled = [float(kf[f]) * float(tune.loc[f, "best_scale"]) for f in feats]
gam = {f: float(tune.loc[f, "gamma"]) for f in feats}
pix = np.stack([lon_g.ravel(), lat_g.ravel(), elev_g.ravel()], axis=1).astype(np.float64)
print("k_tuned: " + " ".join(f"{f}={k:.2f}(g{gam[f]:g})" for f, k in zip(feats, kf_scaled)))
W = CW.build_variant(dict(kernel="paper", gamma=gam, elev=True),
                     pix, st_lat, st_lon, st_h, kf_scaled, feats=feats)
assert np.isfinite(W).all() and np.abs(W.sum(2) - 1).max() < 1e-5
out = os.path.join(CW.OUT_DIR, "paper_opt_ktuned.npz")
ref = np.load(os.path.join(CW.OUT_DIR, "paper_opt.npz"), allow_pickle=True)  # match loader schema
np.savez_compressed(out, W=W, kernel="paper", elev=True, variant="paper_opt_ktuned",
                    gamma=np.array([gam[f] for f in feats], np.float32),
                    k_f=np.array(kf_scaled, np.float32), features=feats,
                    station_names=ref["station_names"], grid_h=ref["grid_h"], grid_w=ref["grid_w"],
                    note="LOSO-train-tuned per-feature (k_scale,gamma); ktune_results.csv 2026-07-12")
print("saved:", out, f"{os.path.getsize(out)/1e6:.1f} MB")
