"""
UNIVERSAL KRIGING with ELEVATION DRIFT (KED) station->grid interpolation weights —
the geostatistical alternative to the paper's exponential-d3d kernel (ablation A3+).

Why: the paper kernel is a heuristic distance decay; KED is the BLUP (best linear
unbiased predictor) under a fitted covariance model, with elevation entering as an
external drift. Elevation-awareness is kept — but principled: the unbiasedness
constraints force  sum(w)=1  and  sum(w*elev_station)=elev_pixel,  i.e. every pixel
implicitly gets the day's own GLS-estimated lapse rate (the drift coefficients are
never applied explicitly; they are integrated out by the constraints).

Method (per feature, TRAIN period 2005-01-01..2019-01-28 only — leakage-free):
  1) pooled OLS of station values on elevation (all train day-station pairs)
     -> slope b; residual r = v - (a + b*elev). Only b matters for the variogram
     (within-day pair differences cancel a and any common daily shift).
  2) empirical residual semivariance per station pair, gamma_ij =
     0.5*mean_t[(r_ti - r_tj)^2] over subsampled train days (stride 5), binned by
     horizontal (haversine) distance and fit by weighted least squares to an
     EXPONENTIAL model  gamma(h) = c0 + c1*(1 - exp(-h/range_km))
     (fit up to ~2/3 of the max pair distance; bin weight = day-pair count).
  3) KED weights per pixel: solve  [[C, F],[F^T, 0]] [w; mu] = [c; f0]  with
     C_ij = c1*exp(-h_ij/range) (diag c0+c1), drift F=[1, elev_km], f0=[1, elev_pixel_km].
     Weights sum to 1 by construction; NEGATIVE weights are normal for kriging and
     are kept pure (no clipping) — %negative is recorded.

Validation (the decisive judge): leave-one-station-out on the TRAIN period, same
conventions as src/eval/loso_interpolation.py (MIN_STATIONS=30 donors, >=365 truth
days, per-station MAE averaged over stations, elevation terciles). Kriging is
LOSO-honest: beta AND the variogram are re-fit per fold from the 67 donors, and the
KED system is re-solved exactly per daily availability pattern (the kriging analog
of the harness' per-day weight renormalization). Champions compared: the paper
kernel at its train-selected per-feature gamma* (compute_interp_weights.GAMMA_OPT)
and the original as-coded kernel (exp(-0.2*d_eff)), both via src/data/interpolation.

Decision rule: kriging_ked.npz (same key schema as paper_opt.npz; W (9,6028,68)
float32, station/feature order per dim_order.json) is written ONLY if kriging beats
the paper-gamma* LOSO MAE by > 0.02 degC on at least one feature. Note for use:
data.py _synth_frames applies W by plain matmul (negatives flow through fine), but
_synth_frames_masked renormalizes by the sum of available weights and assumes
NONNEGATIVE weights -> use the UNMASKED synth path with this file (no PAPERW_MASKS).

Run:  python src/pipeline/compute_kriging_weights.py       (CPU, ~10-20 min all 9)
      KRIG_FEATURES=max_dry_temp,min_dry_temp ...          (smoke: subset)
Writes AT MOST data/interp_weights/kriging_ked.npz — everything else read-only.
"""
import os
import sys
sys.dont_write_bytecode = True   # repo is read-only for this task: no __pycache__ drops
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares

# realpath -> survives symlinked checkouts
_SRC = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(_SRC, "data"))
sys.path.insert(0, _SRC)
import interpolation as INT  # noqa: E402  (haversine + champion kernels, 1:1)
import repo_paths as RP      # noqa: E402

GRID = RP.grid_metadata_npz()                       # bundled
KVAL = RP.k_values_csv()                            # bundled
FEATS = RP.features_order_txt()                     # bundled
DIM_ORDER = os.path.join(RP.stationvec_dir("Center"), "dim_order.json")
CSV_DIR = RP.stations_daily_dir(check=False)        # NOT bundled (raw IMS derivative)
OUT_NPZ = os.path.join(RP.interp_weights_dir(), "kriging_ked.npz")

TRAIN_START, TRAIN_END = "2005-01-01", "2019-01-28"   # train split cutoff (leakage-free)
VARIO_STRIDE = 5          # every 5th train day for the empirical variogram
N_BINS = 18               # distance bins for the variogram fit
FIT_MAX_FRAC = 0.66       # fit up to this fraction of the max pair distance
MIN_STATIONS = 30         # LOSO: skip days with fewer available donors (harness convention)
MIN_DAYS = 365            # LOSO: min truth/pred days per station (harness convention)
WIN_MARGIN = 0.02         # degC: kriging must beat paper-gamma* by more than this
JITTER_REL = 1e-8         # relative diagonal jitter for the kriging system

# train-selected per-feature gamma* of the paper kernel (compute_interp_weights.py)
GAMMA_OPT = {"max_wet_temp": 2.0, "tmp_air_wet": 1.0, "min_wet_temp": 1.0,
             "tmp_air_dry": 12.0, "tmp_dew_pnt": 12.0, "max_heat_stress": 12.0,
             "min_heat_stress": 1.0, "max_dry_temp": 12.0, "min_dry_temp": 2.0}


def _agg(f):   # generator convention: min_*->min, max_*->max, else mean
    return "min" if f.startswith("min_") else ("max" if f.startswith("max_") else "mean")


def load_all_series(names, feats, dates):
    """V (T, S, C) float32 daily station values, one CSV read per station."""
    V = np.full((len(dates), len(names), len(feats)), np.nan, np.float32)
    for j, nm in enumerate(names):
        p = os.path.join(CSV_DIR, nm.replace(" ", "_") + ".csv")
        df = pd.read_csv(p, usecols=lambda c: c == "Date" or c in feats)
        dt = pd.to_datetime(df["Date"], errors="coerce")
        df = df[dt.notna()]
        idx = pd.DatetimeIndex(dt[dt.notna()]).normalize()
        for c, f in enumerate(feats):
            if f not in df.columns:
                continue
            s = pd.to_numeric(df[f], errors="coerce")
            s.index = idx
            g = s.groupby(level=0)
            a = _agg(f)
            s = g.max() if a == "max" else (g.min() if a == "min" else g.mean())
            V[:, j, c] = s.reindex(dates).to_numpy(np.float32)
    return V


# ------------------------------------------------- drift (pooled OLS on elevation)
def station_stats(Vc):
    """Per-station finite count and value sum (sufficient stats for pooled OLS)."""
    fin = np.isfinite(Vc)
    return fin.sum(0).astype(np.float64), np.where(fin, Vc, 0.0).sum(0).astype(np.float64)


def pooled_beta(n_s, sum_s, elev, keep=None):
    """Pooled OLS slope of v on elev over all (day, station) pairs; keep = station mask."""
    if keep is not None:
        n_s, sum_s, elev = n_s[keep], sum_s[keep], elev[keep]
    N, Sx, Sxx = n_s.sum(), (n_s * elev).sum(), (n_s * elev ** 2).sum()
    Sy, Sxy = sum_s.sum(), (elev * sum_s).sum()
    den = N * Sxx - Sx ** 2
    return float((N * Sxy - Sx * Sy) / den) if den > 0 else 0.0


# ------------------------------------------------- empirical variogram machinery
def pair_stats(Vs):
    """Per station pair, over subsampled days both finite:
    M1 = mean(v_i - v_j), M2 = mean((v_i - v_j)^2), N = count.  (S,S) each.
    Residual semivariance for ANY drift slope b follows analytically:
      0.5*E[((v_i-v_j) - b*(e_i-e_j))^2] = 0.5*(M2 - 2b*dE*M1 + b^2*dE^2)."""
    fin = np.isfinite(Vs)
    D = np.where(fin[:, :, None] & fin[:, None, :],
                 Vs[:, :, None] - Vs[:, None, :], 0.0)
    N = (fin[:, :, None] & fin[:, None, :]).sum(0).astype(np.float64)
    M1 = np.where(N > 0, D.sum(0) / np.maximum(N, 1), np.nan)
    M2 = np.where(N > 0, (D ** 2).sum(0) / np.maximum(N, 1), np.nan)
    return M1, M2, N


def semivar_for_beta(M1, M2, dE, b):
    return 0.5 * (M2 - 2.0 * b * dE * M1 + (b * dE) ** 2)


def bin_variogram(H, G, N, edges):
    """Weighted bin means over upper-triangle pairs. Returns (h_b, g_b, w_b)."""
    iu = np.triu_indices(H.shape[0], 1)
    h, g, n = H[iu], G[iu], N[iu]
    ok = np.isfinite(g) & (n > 0)
    h, g, n = h[ok], g[ok], n[ok]
    hb, gb, wb = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h >= lo) & (h < hi)
        if not m.any():
            continue
        w = n[m]
        hb.append((h[m] * w).sum() / w.sum())
        gb.append((g[m] * w).sum() / w.sum())
        wb.append(w.sum())
    return np.array(hb), np.array(gb), np.array(wb)


def fit_exp_variogram(h_b, g_b, w_b):
    """gamma(h) = c0 + c1*(1-exp(-h/r)), weighted LSQ. Returns (c0, c1, r)."""
    if len(h_b) < 5:
        raise RuntimeError("too few variogram bins")
    k = max(3, len(g_b) // 3)
    g_hi = float(np.average(g_b[-k:], weights=w_b[-k:]))
    g_lo = float(g_b[0])
    x0 = np.array([max(0.5 * g_lo, 1e-6), max(g_hi - 0.5 * g_lo, 1e-6), 60.0])
    wn = np.sqrt(w_b / w_b.sum())

    def resid(x):
        c0, c1, r = x
        return wn * (c0 + c1 * (1.0 - np.exp(-h_b / r)) - g_b)

    sol = least_squares(resid, x0, bounds=([0.0, 1e-9, 2.0], [10 * g_hi, 10 * g_hi, 1500.0]),
                        method="trf", max_nfev=3000)
    return tuple(float(v) for v in sol.x)


# ------------------------------------------------- KED solves
def cov_matrix(H, c0, c1, r):
    C = c1 * np.exp(-H / r)
    np.fill_diagonal(C, c0 + c1 + JITTER_REL * (c0 + c1))
    return C


def ked_system(Css, e_km):
    """Assemble the (n+2, n+2) KED KKT matrix from covariance Css and drift [1, e_km]."""
    n = Css.shape[0]
    A = np.zeros((n + 2, n + 2))
    A[:n, :n] = Css
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    A[:n, n + 1] = e_km
    A[n + 1, :n] = e_km
    return A


def ked_grid_weights(H_ss, H_ps, e_s_km, e_p_km, params):
    """All-pixel KED weights: one LU factorization, P right-hand sides. (P,S)."""
    c0, c1, r = params
    S, P = H_ss.shape[0], H_ps.shape[0]
    A = ked_system(cov_matrix(H_ss, c0, c1, r), e_s_km)
    B = np.zeros((S + 2, P))
    B[:S] = c1 * np.exp(-H_ps.T / r)
    B[S] = 1.0
    B[S + 1] = e_p_km
    sol = lu_solve(lu_factor(A), B)
    return sol[:S].T


def kriging_loso_station(Vd, Hdd, h_td, e_d_km, e_t_km, params):
    """KED-interpolate one held-out station from its donors; exact re-solve per
    unique daily availability pattern (the kriging analog of per-day renorm).
    Returns (pred (T,), mean fraction of negative weights over solves)."""
    c0, c1, r = params
    fin = np.isfinite(Vd)
    ok = fin.sum(1) >= MIN_STATIONS
    pred = np.full(Vd.shape[0], np.nan)
    if not ok.any():
        return pred, np.nan
    Cfull = cov_matrix(Hdd, c0, c1, r)
    ctar = c1 * np.exp(-h_td / r)
    days = np.nonzero(ok)[0]
    uniq, inv = np.unique(fin[ok], axis=0, return_inverse=True)
    negs = []
    for u in range(uniq.shape[0]):
        av = np.nonzero(uniq[u])[0]
        A = ked_system(Cfull[np.ix_(av, av)], e_d_km[av])
        b = np.concatenate([ctar[av], [1.0, e_t_km]])
        try:
            w = np.linalg.solve(A, b)[:av.size]
        except np.linalg.LinAlgError:
            continue
        negs.append(float((w < 0).mean()))
        rows = days[inv == u]
        pred[rows] = Vd[np.ix_(rows, av)] @ w
    return pred, (float(np.mean(negs)) if negs else np.nan)


def renorm_pred(w, Vd):
    """Harness convention for the fixed-kernel champions: per-day renormalized
    weighted average over available donors (loso_interpolation.loso_pred, no lapse)."""
    mask = np.isfinite(Vd)
    wm = w[None, :] * mask
    denom = wm.sum(1)
    ok = (mask.sum(1) >= MIN_STATIONS) & (denom > 1e-9)
    return np.where(ok, np.nansum(wm * np.where(mask, Vd, 0.0), 1) / np.maximum(denom, 1e-9),
                    np.nan)


# ------------------------------------------------- main
def main():
    t00 = time.time()
    g = np.load(GRID, allow_pickle=True)
    grid_names = [str(s) for s in g["station_names"]]
    order = json.load(open(DIM_ORDER))["station_order"]
    assert sorted(order) == sorted(grid_names), "station set mismatch grid vs stationvec"
    ix = np.array([grid_names.index(s) for s in order])
    slat = g["station_lat"].astype(np.float64)[ix]
    slon = g["station_lon"].astype(np.float64)[ix]
    sh = g["station_elev"].astype(np.float64)[ix]
    lat_g, lon_g, elev_g = g["lat_grid"], g["lon_grid"], g["elev_grid"]   # (44,137)
    gh, gw = elev_g.shape

    feats = open(FEATS).read().split()
    assert feats == json.load(open(DIM_ORDER))["feature_order"], "feature order mismatch"
    kdf = pd.read_csv(KVAL)
    kf_per_feat = {f: float(kdf.loc[kdf["Feature"] == f, "k"].iloc[0]) for f in feats}
    run_feats = [f.strip() for f in os.environ.get("KRIG_FEATURES", ",".join(feats)).split(",")]

    # geometry (feature-independent)
    S = len(order)
    H_ss = INT.haversine_km(slat[:, None], slon[:, None], slat[None, :], slon[None, :])
    pixlat, pixlon = lat_g.ravel().astype(np.float64), lon_g.ravel().astype(np.float64)
    H_ps = INT.haversine_km(pixlat[:, None], pixlon[:, None], slat[None, :], slon[None, :])
    e_s_km, e_p_km = sh / 1000.0, elev_g.ravel().astype(np.float64) / 1000.0
    dE = sh[:, None] - sh[None, :]
    fit_max = FIT_MAX_FRAC * H_ss.max()
    edges = np.linspace(0.0, fit_max, N_BINS + 1)
    elev_terc = np.digitize(sh, np.quantile(sh, [1 / 3, 2 / 3]))

    dates = pd.date_range(TRAIN_START, TRAIN_END, freq="D")
    print(f"[setup] S={S} stations | grid {gh}x{gw} -> P={gh * gw} | train days={len(dates)} "
          f"({TRAIN_START}..{TRAIN_END}) | max pair dist={H_ss.max():.0f} km "
          f"(variogram fit <= {fit_max:.0f} km, {N_BINS} bins, day stride {VARIO_STRIDE})")
    t0 = time.time()
    V = load_all_series(order, feats, dates)
    print(f"[load ] station series {V.shape} in {time.time() - t0:.1f}s | "
          f"finite: {100 * np.isfinite(V).mean():.1f}%")

    vario_full, results, neg_loso = {}, {}, {}
    for f in run_feats:
        c = feats.index(f)
        t0 = time.time()
        Vc = V[:, :, c].astype(np.float64)
        Vsub = Vc[::VARIO_STRIDE]

        # full-data drift + variogram (for the deliverable W and as fold fallback)
        n_s, sum_s = station_stats(Vc)
        b_full = pooled_beta(n_s, sum_s, sh)
        M1, M2, N = pair_stats(Vsub)
        G = semivar_for_beta(M1, M2, dE, b_full)
        c0, c1, r = fit_exp_variogram(*bin_variogram(H_ss, G, N, edges))
        vario_full[f] = (c0, c1, r)
        print(f"\n=== {f} (agg={_agg(f)}, k_f={kf_per_feat[f]:.2f}, paper gamma*="
              f"{GAMMA_OPT[f]:.0f}) ===")
        print(f"[drift] pooled lapse b={1000 * b_full:+.2f} degC/km | "
              f"[variogram] nugget={c0:.3f} psill={c1:.3f} sill={c0 + c1:.3f} degC^2, "
              f"range={r:.0f} km")

        # ---------------- LOSO ----------------
        pap_key, asc_key = f"paper_g{GAMMA_OPT[f]:.0f}*", "as_coded(K=0.2)"
        errs = {"kriging_ked": [], pap_key: [], asc_key: []}
        negs = []
        for j in range(S):
            donors = np.arange(S) != j
            has_truth = np.isfinite(Vc[:, j])
            if has_truth.sum() < MIN_DAYS:
                continue
            Vd = Vc[:, donors]
            # fold-honest drift + variogram from the 67 donors
            b_j = pooled_beta(n_s, sum_s, sh, keep=donors)
            Gj = semivar_for_beta(M1, M2, dE, b_j)[np.ix_(donors, donors)]
            try:
                pj = fit_exp_variogram(*bin_variogram(H_ss[np.ix_(donors, donors)], Gj,
                                                      N[np.ix_(donors, donors)], edges))
            except Exception:
                pj = (c0, c1, r)
            pred_k, negfrac = kriging_loso_station(
                Vd, H_ss[np.ix_(donors, donors)], H_ss[j, donors],
                e_s_km[donors], e_s_km[j], pj)
            if np.isfinite(negfrac):
                negs.append(negfrac)
            # champions, exactly like loso_interpolation.py
            pix_t = np.array([[slon[j], slat[j], sh[j]]])
            w_pap = INT.weights_paper(pix_t, slat[donors], slon[donors], sh[donors],
                                      kf_per_feat[f], gamma=GAMMA_OPT[f], elev=True)[0]
            w_asc = INT.weights_as_coded(pix_t, slat[donors], slon[donors], sh[donors],
                                         kf_per_feat[f], kernel="exponential", k_exp=0.2)[0]
            for kname, pred in (("kriging_ked", pred_k),
                                (pap_key, renorm_pred(w_pap, Vd)),
                                (asc_key, renorm_pred(w_asc, Vd))):
                m = has_truth & np.isfinite(pred)
                if m.sum() >= MIN_DAYS:
                    errs[kname].append((j, float(np.abs(pred[m] - Vc[m, j]).mean())))
        neg_loso[f] = float(np.mean(negs)) if negs else np.nan

        print(f"{'kernel':18}{'MAE_all':>8}{'lo-elev':>9}{'mid':>7}{'hi-elev':>9}{'n_st':>6}")
        maes = {}
        for kname, ev in errs.items():
            if not ev:
                maes[kname] = np.nan
                continue
            js = np.array([e[0] for e in ev])
            ms = np.array([e[1] for e in ev])
            terc = elev_terc[js]
            maes[kname] = float(ms.mean())
            tm = [ms[terc == t].mean() if (terc == t).any() else np.nan for t in (0, 1, 2)]
            print(f"{kname:18}{ms.mean():8.3f}{tm[0]:9.3f}{tm[1]:7.3f}{tm[2]:9.3f}{len(ev):6d}")
        results[f] = {"ked": maes["kriging_ked"], "paper": maes[pap_key],
                      "as_coded": maes[asc_key]}
        print(f"[LOSO ] delta(KED - paper*) = {maes['kriging_ked'] - maes[pap_key]:+.3f} degC | "
              f"%neg weights (fold mean) = {100 * neg_loso[f]:.1f}% | {time.time() - t0:.0f}s")

    # ---------------- summary + decision ----------------
    print(f"\n{'=' * 84}\nLOSO summary (TRAIN {TRAIN_START}..{TRAIN_END}, MAE degC, "
          f"lower=better; paper*=train-selected gamma)")
    print(f"{'feature':17}{'KED':>7}{'paper*':>8}{'as_coded':>9}{'d(KED-pap)':>11}"
          f"{'nugget':>8}{'psill':>7}{'range':>7}{'%neg':>6}")
    wins = []
    for f in run_feats:
        d = results[f]["ked"] - results[f]["paper"]
        c0, c1, r = vario_full[f]
        tag = ""
        if d < -WIN_MARGIN:
            wins.append(f)
            tag = "  << KED wins"
        print(f"{f:17}{results[f]['ked']:7.3f}{results[f]['paper']:8.3f}"
              f"{results[f]['as_coded']:9.3f}{d:+11.3f}{c0:8.3f}{c1:7.3f}{r:7.0f}"
              f"{100 * neg_loso[f]:6.1f}{tag}")

    if not wins:
        print(f"\n[DECISION] kriging does NOT beat paper-gamma* by >{WIN_MARGIN} degC on any "
              f"feature -> NO weight file written (negative result for the ablation).")
        print(f"[scope] wrote nothing; repo untouched. total {time.time() - t00:.0f}s")
        return

    print(f"\n[DECISION] kriging beats paper-gamma* by >{WIN_MARGIN} degC on: {wins} "
          f"-> building {os.path.basename(OUT_NPZ)} (all {len(feats)} features)")
    W = np.empty((len(feats), gh * gw, S), np.float32)
    negfrac = np.zeros(len(feats), np.float32)
    for c, f in enumerate(feats):
        params = vario_full.get(f)
        if params is None:   # feature not LOSO'd in a KRIG_FEATURES run: fit now
            Vc = V[:, :, c].astype(np.float64)
            n_s, sum_s = station_stats(Vc)
            M1, M2, N = pair_stats(Vc[::VARIO_STRIDE])
            G = semivar_for_beta(M1, M2, dE, pooled_beta(n_s, sum_s, sh))
            params = fit_exp_variogram(*bin_variogram(H_ss, G, N, edges))
            vario_full[f] = params
        Wc = ked_grid_weights(H_ss, H_ps, e_s_km, e_p_km, params)
        rs_dev = np.abs(Wc.sum(1) - 1.0).max()
        el_dev = np.abs(Wc @ e_s_km - e_p_km).max() * 1000.0
        negfrac[c] = (Wc < 0).mean()
        assert np.isfinite(Wc).all() and rs_dev < 1e-6 and el_dev < 1e-3, f"{f}: bad solve"
        W[c] = Wc.astype(np.float32)
        print(f"  [{f:16s}] %neg={100 * negfrac[c]:5.1f}  min_w={Wc.min():+.3f}  "
              f"max_w={Wc.max():+.3f}  |rowsum-1|<{rs_dev:.1e}  |elev constr|<{el_dev:.1e} m")

    np.savez(OUT_NPZ, W=W, kernel="kriging_ked",
             gamma=np.full(len(feats), np.nan, np.float32),   # n/a for kriging (schema parity)
             elev=True, variant="kriging_ked",
             station_names=np.array(order, object), features=np.array(feats, object),
             k_f=np.array([kf_per_feat[f] for f in feats], np.float32), grid_h=gh, grid_w=gw,
             vario_nugget=np.array([vario_full[f][0] for f in feats], np.float32),
             vario_psill=np.array([vario_full[f][1] for f in feats], np.float32),
             vario_range_km=np.array([vario_full[f][2] for f in feats], np.float32),
             neg_frac=negfrac, drift="[1, station_elev]  (universal kriging / KED)",
             train_period=f"{TRAIN_START}..{TRAIN_END}")
    print(f"\n[saved] {OUT_NPZ} ({os.path.getsize(OUT_NPZ) / 1e6:.1f} MB)")
    print("[note ] W has NEGATIVE weights (kriging): fine for the plain synth path "
          "(_synth_frames matmul); do NOT combine with PAPERW_MASKS (masked renorm "
          "assumes nonnegative weights).")
    print(f"[scope] wrote ONLY the npz above. total {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
