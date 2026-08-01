"""
LOSO (leave-one-station-out) interpolation benchmark — the DIRECT measure of
"meaningful pixels": hold out each station, interpolate its daily value from the
other 67 at its exact coordinate, compare against what it actually measured.
Ground truth exists exactly at station coordinates, so LOSO MAE ranks kernels by
pixel-level fidelity (the goal: each pixel as close as possible to the real value).

Kernels compared:
  as_coded    exp(-0.2*d_eff), d_eff=sqrt(d_km^2+(0.001*k_f*dh)^2)  [the current frames]
  idw_p2      plain inverse-distance (1/d^2, horizontal only)
  idw_lapse   IDW on sea-level-reduced temps (T - G*h), re-apply G*h_pixel (G=6.5C/km)
              [the physical gold standard for temperature interpolation]
  paper_g{1,2,4}   paper eq. d3d (max-normalized, k_f on elevation), gamma bandwidth
  paper_g2_noelev  same, elevation term dropped (isolates elevation's contribution)

Output: per-kernel MAE (mean over stations), split by station-elevation tercile
(elevation-aware kernels should win where elevation matters). Per-day weights are
renormalized over available stations (same convention as the frame generator).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP
from data.interpolation import weights_as_coded, weights_paper

GRID = RP.grid_metadata_npz()               # bundled: data/grid_metadata.npz
KVALS = RP.k_values_csv()                   # bundled: data/k_values.csv
CSV_DIR = RP.stations_daily_dir()           # NOT bundled: per-station daily CSVs (DATA_ROOT)
GAMMA_LAPSE = 0.0065          # degC per meter
MIN_STATIONS = 30             # skip days with fewer available donors
# gamma/kernel SELECTION must be leakage-free: restrict to the TRAIN period via env
# (train cutoff 2019-01-28 per the dataset splits). Default = full range (reporting).
END_DATE = os.environ.get("LOSO_END", "2025-06-30")


def load_station_series(names, feature, agg):
    """Daily series per station (2005..2025). agg: 'max'|'min' like the generator."""
    series = {}
    for nm in names:
        p = os.path.join(CSV_DIR, nm.replace(" ", "_") + ".csv")
        df = pd.read_csv(p, usecols=lambda c: c in ("Date", feature))
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        s = pd.to_numeric(df[feature], errors="coerce")
        s.index = df["Date"].dt.normalize()
        g = s.groupby(level=0)
        s = g.max() if agg == "max" else (g.min() if agg == "min" else g.mean())
        series[nm] = s[(s.index >= "2005-01-01") & (s.index <= END_DATE)]
    dates = pd.date_range("2005-01-01", END_DATE, freq="D")
    V = np.full((len(dates), len(names)), np.nan, np.float32)
    for j, nm in enumerate(names):
        V[:, j] = series[nm].reindex(dates).to_numpy(np.float32)
    return dates, V


def loso_pred(w, V_donors, h_donors=None, h_target=None, lapse=False):
    """Per-day renormalized weighted average; optional lapse-rate reduction."""
    Vd = V_donors.copy()
    if lapse:
        Vd = Vd - GAMMA_LAPSE * h_donors[None, :]        # reduce to sea level
    mask = np.isfinite(Vd)
    wm = w[None, :] * mask
    denom = wm.sum(1)
    ok = (mask.sum(1) >= MIN_STATIONS) & (denom > 1e-9)
    pred = np.where(ok, np.nansum(wm * np.where(mask, Vd, 0.0), 1) / np.maximum(denom, 1e-9), np.nan)
    if lapse:
        pred = pred + GAMMA_LAPSE * h_target             # re-apply target elevation
    return pred


def main():
    g = np.load(GRID, allow_pickle=True)
    names = [str(x) for x in g["station_names"]]
    slat, slon, sh = (g[k].astype(np.float64) for k in ("station_lat", "station_lon", "station_elev"))
    kdf = pd.read_csv(KVALS)

    features_order = [l.strip() for l in open(RP.features_order_txt()) if l.strip()]
    only = os.environ.get("LOSO_FEATURES")   # e.g. LOSO_FEATURES=max_dry_temp for a quick run
    if only:
        keep = {f.strip() for f in only.split(",")}
        features_order = [f for f in features_order if f in keep]
    def _agg(f):   # generator convention: min_*->min, max_*->max, else mean
        return "min" if f.startswith("min_") else ("max" if f.startswith("max_") else "mean")
    for feature, agg in [(f, _agg(f)) for f in features_order]:
        k_f = float(kdf.loc[kdf["Feature"] == feature].iloc[0].get("k",
                    kdf.loc[kdf["Feature"] == feature].iloc[0, 1]))
        dates, V = load_station_series(names, feature, agg)
        print(f"\n=== LOSO — {feature} (agg={agg}, k_f={k_f:.2f}) | days={len(dates)}, stations={len(names)} ===")

        kernels = {
            "as_coded(K=0.2)": dict(kind="as_coded", kw=dict(kernel="exponential", k_exp=0.2)),
            "idw_p2":          dict(kind="as_coded", kw=dict(kernel="idw", p=2, alpha_km_per_m=0.0)),
            "idw_lapse(6.5)":  dict(kind="as_coded", kw=dict(kernel="idw", p=2, alpha_km_per_m=0.0), lapse=True),
            "paper_g1":        dict(kind="paper", kw=dict(gamma=1.0, elev=True)),
            "paper_g2":        dict(kind="paper", kw=dict(gamma=2.0, elev=True)),
            "paper_g4":        dict(kind="paper", kw=dict(gamma=4.0, elev=True)),
            "paper_g6":        dict(kind="paper", kw=dict(gamma=6.0, elev=True)),
            "paper_g8":        dict(kind="paper", kw=dict(gamma=8.0, elev=True)),
            "paper_g12":       dict(kind="paper", kw=dict(gamma=12.0, elev=True)),
            "paper_g4_noelev": dict(kind="paper", kw=dict(gamma=4.0, elev=False)),
            "paper_g2_noelev": dict(kind="paper", kw=dict(gamma=2.0, elev=False)),
        }
        errs = {k: [] for k in kernels}          # per-station MAE
        elev_terc = np.digitize(sh, np.quantile(sh, [1/3, 2/3]))

        for j in range(len(names)):
            donors = np.arange(len(names)) != j
            pix = np.array([[slon[j], slat[j], sh[j]]])
            has_truth = np.isfinite(V[:, j])
            if has_truth.sum() < 365:
                continue
            for kname, spec in kernels.items():
                if spec["kind"] == "as_coded":
                    w = weights_as_coded(pix, slat[donors], slon[donors], sh[donors], k_f, **spec["kw"])[0]
                else:
                    w = weights_paper(pix, slat[donors], slon[donors], sh[donors], k_f, **spec["kw"])[0]
                pred = loso_pred(w, V[:, donors], sh[donors], sh[j], lapse=spec.get("lapse", False))
                m = has_truth & np.isfinite(pred)
                if m.sum() >= 365:
                    errs[kname].append((j, float(np.abs(pred[m] - V[m, j]).mean())))

        print(f"{'kernel':18}{'MAE_all':>8}{'lo-elev':>9}{'mid':>7}{'hi-elev':>9}   (pixel fidelity, degC; lower=better)")
        rows = []
        for kname, ev in errs.items():
            if not ev:
                continue
            js = np.array([e[0] for e in ev]); ms = np.array([e[1] for e in ev])
            terc = elev_terc[js]
            row = (ms.mean(),
                   ms[terc == 0].mean() if (terc == 0).any() else np.nan,
                   ms[terc == 1].mean() if (terc == 1).any() else np.nan,
                   ms[terc == 2].mean() if (terc == 2).any() else np.nan)
            rows.append((kname, *row))
        for kname, a, lo, mid, hi in sorted(rows, key=lambda r: r[1]):
            print(f"{kname:18}{a:8.3f}{lo:9.3f}{mid:7.3f}{hi:9.3f}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
