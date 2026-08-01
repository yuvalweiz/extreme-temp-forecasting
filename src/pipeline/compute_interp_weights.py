"""
Precompute station->grid interpolation WEIGHT matrices (memory-wise synthesized
frames, ablation A3). A daily frame for feature c is linear in the station values:

    frame[c] = (W_c @ s_c).reshape(44, 137),   W_c (6028, 68) row-stochastic.

So instead of materializing frame banks per interpolation variant, we store one
small W = (9, 6028, 68) float32 (~15 MB) per VARIANT and synthesize frames on the
fly from the existing station-vector datasets (data.SynthFrameDataset, PAPERW=...).

Kernels come 1:1 from src/data/interpolation.py:
  * paper_g{1,2,4}   -- weights_paper (paper.tex eq. d3d): max-normalized horizontal
                        + k_f-scaled max-normalized elevation, w=exp(-gamma*d3d),
                        norm="global" (per-frame maxima over ALL pixel-station pairs).
  * paper_g2_noelev  -- same, elevation term dropped (isolates elevation contribution).
  * idw_p2           -- classic IDW baseline: w = 1/d_km^2, horizontal only
                        (weights_as_coded kernel="idw", k_f=0 -> exact-hit safeguard kept).

Station axis follows data/stationvec_*/dim_order.json station_order (== grid_metadata
order, asserted); feature axis follows features_order.txt (== stationvec feature order).
Writes to  repo/data/interp_weights/<variant>.npz  keys: W, kernel, gamma, elev,
variant, station_names, features, k_f, grid_h, grid_w.

Run:  python src/pipeline/compute_interp_weights.py   (CPU, ~1 min, ~74 MB total)
"""
import os
import sys
import json
import numpy as np
import pandas as pd

# realpath -> the _REPO derivation survives symlinked checkouts
_SRC = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(_SRC, "data"))
sys.path.insert(0, _SRC)
import interpolation as INT  # noqa: E402  (weights_paper / weights_as_coded, 1:1)
import repo_paths as RP      # noqa: E402

GRID = RP.grid_metadata_npz()                       # bundled: data/grid_metadata.npz
KVAL = RP.k_values_csv()                            # bundled: data/k_values.csv
FEATS = RP.features_order_txt()                     # bundled: data/features_order.txt
DIM_ORDER = os.path.join(RP.stationvec_dir("Center"), "dim_order.json")
SV_SAMPLE = os.path.join(RP.stationvec_dir("Center"), "sample_2005-07-01.npz")
OUT_DIR = RP.interp_weights_dir()

# variant name -> kernel spec
# per-feature gamma* selected by TRAIN-ONLY leave-one-station-out fidelity
# (src/eval/loso_interpolation.py, LOSO_END=2019-01-28): sharp gamma for low-k_f
# dry/heat channels, soft gamma for high-k_f wet channels (k_f already sharpens
# the elevation axis inside d3d, gamma compensates on the horizontal).
GAMMA_OPT = {"max_wet_temp": 2.0, "tmp_air_wet": 1.0, "min_wet_temp": 1.0,
             "tmp_air_dry": 12.0, "tmp_dew_pnt": 12.0, "max_heat_stress": 12.0,
             "min_heat_stress": 1.0, "max_dry_temp": 12.0, "min_dry_temp": 2.0}

VARIANTS = {
    "paper_g1":        dict(kernel="paper", gamma=1.0, elev=True),
    "paper_g2":        dict(kernel="paper", gamma=2.0, elev=True),
    "paper_g4":        dict(kernel="paper", gamma=4.0, elev=True),
    "paper_g12":       dict(kernel="paper", gamma=12.0, elev=True),
    "paper_opt":       dict(kernel="paper", gamma=GAMMA_OPT, elev=True),  # per-feature gamma*
    "paper_g2_noelev": dict(kernel="paper", gamma=2.0, elev=False),  # k_f irrelevant
    "idw_p2":          dict(kernel="idw",   gamma=None, elev=False), # w=1/d_km^2, horizontal only
    # the ORIGINAL as-coded kernel through the SAME synth machinery: isolates
    # synthesis/imputation effects from kernel effects (compare vs disk frames)
    "as_coded_synth":  dict(kernel="as_coded", gamma=None, elev=True),
}


def build_variant(spec, pix, st_lat, st_lon, st_h, kf_per_feat, feats=None):
    """W (C, P, S) float32, one row-stochastic W_c per feature.
    spec['gamma'] may be a scalar or a {feature_name: gamma} dict (per-feature gamma*)."""
    C = len(kf_per_feat)
    W = np.empty((C, pix.shape[0], st_lat.shape[0]), np.float32)
    for c, k_f in enumerate(kf_per_feat):
        if spec["kernel"] == "paper":
            gam = spec["gamma"][feats[c]] if isinstance(spec["gamma"], dict) else spec["gamma"]
            w = INT.weights_paper(pix, st_lat, st_lon, st_h, k_f=k_f,
                                  gamma=gam, elev=spec["elev"], norm="global")
        elif spec["kernel"] == "idw":
            # k_f=0 kills the elevation term exactly -> pure horizontal 1/d^2 (+ exact-hit safeguard)
            w = INT.weights_as_coded(pix, st_lat, st_lon, st_h, k_f=0.0, kernel="idw", p=2)
        elif spec["kernel"] == "as_coded":
            # the published frames' kernel (K_EXP=0.2, near-inert elevation term)
            w = INT.weights_as_coded(pix, st_lat, st_lon, st_h, k_f=k_f,
                                     kernel="exponential", k_exp=0.2)
        else:
            raise ValueError(spec["kernel"])
        W[c] = w.astype(np.float32)
    return W


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    g = np.load(GRID, allow_pickle=True)
    lat_g, lon_g, elev_g = g["lat_grid"], g["lon_grid"], g["elev_grid"]  # (44,137)
    gh, gw = elev_g.shape
    grid_names = [str(s) for s in g["station_names"]]

    # station axis MUST follow the station-vector layout (dim_order.json)
    order = json.load(open(DIM_ORDER))["station_order"]
    assert sorted(order) == sorted(grid_names), "station set mismatch grid vs stationvec"
    ix = np.array([grid_names.index(s) for s in order])
    st_lat = g["station_lat"].astype(np.float64)[ix]
    st_lon = g["station_lon"].astype(np.float64)[ix]
    st_h = g["station_elev"].astype(np.float64)[ix]

    # feature axis follows features_order.txt (== stationvec feature order, asserted)
    feats = open(FEATS).read().split()
    assert feats == json.load(open(DIM_ORDER))["feature_order"], "feature order mismatch"
    kf = dict(zip(pd.read_csv(KVAL)["Feature"], pd.read_csv(KVAL)["k"]))
    kf_per_feat = [float(kf[f]) for f in feats]

    # pixels flattened in C order of the (44,137) grids == stored frame layout
    pix = np.stack([lon_g.ravel(), lat_g.ravel(), elev_g.ravel()], axis=1).astype(np.float64)
    print(f"[grid] {gh}x{gw} -> P={pix.shape[0]} | S={len(order)} stations | C={len(feats)} features")
    print(f"[k_f ] " + " ".join(f"{f}={k:.2f}" for f, k in zip(feats, kf_per_feat)))

    for name, spec in VARIANTS.items():
        W = build_variant(spec, pix, st_lat, st_lon, st_h, kf_per_feat, feats=feats)
        rs = W.sum(axis=2)
        assert np.isfinite(W).all(), f"{name}: NaN/inf in W"
        assert np.abs(rs - 1.0).max() < 1e-5, f"{name}: rows not stochastic"
        out = os.path.join(OUT_DIR, f"{name}.npz")
        if spec["gamma"] is None:
            gam_arr = np.full(len(feats), np.nan, np.float32)
        elif isinstance(spec["gamma"], dict):
            gam_arr = np.array([spec["gamma"][f] for f in feats], np.float32)
        else:
            gam_arr = np.full(len(feats), float(spec["gamma"]), np.float32)
        np.savez(out, W=W, kernel=spec["kernel"],
                 gamma=gam_arr,
                 elev=bool(spec["elev"]), variant=name,
                 station_names=np.array(order, object), features=np.array(feats, object),
                 k_f=np.array(kf_per_feat, np.float32), grid_h=gh, grid_w=gw)
        print(f"[{name:16s}] W{W.shape} rowsum dev max={np.abs(rs-1).max():.2e} "
              f"| {os.path.getsize(out)/1e6:.1f} MB -> {out}")

    # ---------------- spot check (docs/INTERPOLATION_FINDING.md lines 26-33) ------------
    # high pixel (600 m) near an 800 m hill station: paper puts the weight ON the hill,
    # as-coded/idw put it on nearby LOWLAND stations.
    t_lat = np.array([32.0, 32.1, 31.9, 32.05])
    t_lon = np.array([34.8, 34.9, 34.7, 34.85])
    t_h = np.array([10.0, 800.0, 20.0, 50.0])      # station 1 = hill
    vals = np.array([30.0, 22.0, 31.0, 29.0])       # cooler on the hill
    tpix = np.array([[34.82, 32.02, 600.0]])        # high pixel near the hill
    k_f = kf["max_dry_temp"]
    wc = INT.weights_as_coded(tpix, t_lat, t_lon, t_h, k_f)[0]
    wi = INT.weights_as_coded(tpix, t_lat, t_lon, t_h, 0.0, kernel="idw", p=2)[0]
    wp = INT.weights_paper(tpix, t_lat, t_lon, t_h, k_f, gamma=2.0)[0]
    print(f"\n[spot check] high pixel (600 m) near hill station, k_f={k_f:.2f} (max_dry_temp):")
    print(f"  as-coded  w={np.round(wc,3)} -> T={wc@vals:.2f} degC  (lowland)")
    print(f"  idw_p2    w={np.round(wi,3)} -> T={wi@vals:.2f} degC  (lowland)")
    print(f"  paper g2  w={np.round(wp,3)} -> T={wp@vals:.2f} degC  (hill station dominates)")
    assert wp.argmax() == 1 and wc.argmax() != 1 and wi.argmax() != 1, "spot-check direction failed"

    # ------------- elevation structure of a real synthesized frame -------------------
    # max_dry_temp on a summer day: paper frames should anti-correlate with elevation
    # (cooler up high) much more strongly than IDW.
    sv = np.load(SV_SAMPLE)["X"][-1].reshape(len(order), len(feats))  # last input day 2005-06-30
    c = feats.index("max_dry_temp")
    s_c = sv[:, c].astype(np.float64)
    corr = {}
    for name in ("paper_g2", "idw_p2"):
        Wc = np.load(os.path.join(OUT_DIR, f"{name}.npz"))["W"][c].astype(np.float64)
        frame = Wc @ s_c
        corr[name] = np.corrcoef(frame, elev_g.ravel())[0, 1]
        print(f"[elev corr] {name:9s} corr(synth max_dry_temp frame, elev_grid) = {corr[name]:+.3f} "
              f"(frame range [{frame.min():.1f},{frame.max():.1f}] degC, day 2005-06-30)")
    assert corr["paper_g2"] < corr["idw_p2"], "paper frame should be more elevation-structured"
    print("[OK] all validations passed")


if __name__ == "__main__":
    main()
