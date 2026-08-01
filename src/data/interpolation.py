"""
Elevation-aware station->grid interpolation.

Two implementations:
  * weights_as_coded(...)  -- reproduces the ORIGINAL preprocess notebook
                              (cell 13): d_eff = sqrt(d_km^2 + (0.001*k_f*dh_m)^2),
                              kernel exp(-k_exp * d_eff) with raw km. No per-axis
                              max-normalization.
  * weights_paper(...)     -- the formula actually WRITTEN in the paper
                              (paper.tex eq. d3d / exp_weights):
                                 d3d = sqrt( (d_h/d_h_max)^2 + (k_f*|d_e|/d_e_max)^2 )
                                 w  = exp(-gamma * d3d), renormalized per pixel.
                              gamma is the tunable bandwidth the reviewers asked
                              to justify; set elev=False for the no-elevation ablation.

The mismatch (no max-normalization + the magic 0.001 vs the paper's normalized
k_f, and exp over raw-km vs normalized d3d) is documented in
docs/INTERPOLATION_FINDING.md. weights_paper is what the A3 ablation rebuilds
frames with.

k_f comes from k_values.csv (column 'k') = |StdEffect(height)/StdEffect(distance)|
of a per-feature pair regression -- exactly the paper's dimensionless ratio.
"""
import numpy as np

R_EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs broadcast (degrees)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _pixel_station_distances(pix, st_lat, st_lon, st_h):
    """
    pix: (P,3) array of [lon,lat,height] per pixel (P=W*H flattened).
    Returns d_km (P,S) horizontal and dh_m (P,S) |elevation diff| in meters.
    """
    plon, plat, ph = pix[:, 0][:, None], pix[:, 1][:, None], pix[:, 2][:, None]
    d_km = haversine_km(plat, plon, st_lat[None, :], st_lon[None, :])  # (P,S)
    dh_m = np.abs(st_h[None, :] - ph)                                   # (P,S)
    dh_m = np.where(np.isnan(ph), 0.0, dh_m)  # missing pixel elev -> ignore height
    return d_km, dh_m


def _normalize_rows(w, eps=1e-12):
    s = w.sum(axis=1, keepdims=True)
    out = np.where(s > eps, w / np.maximum(s, eps), 1.0 / w.shape[1])
    return out


def weights_as_coded(pix, st_lat, st_lon, st_h, k_f,
                     kernel="exponential", p=2, k_exp=1.0,
                     alpha_km_per_m=0.001, eps_km=1e-6):
    """Original notebook behaviour (no max-normalization)."""
    d_km, dh_m = _pixel_station_distances(pix, st_lat, st_lon, st_h)
    alpha_eff = alpha_km_per_m * float(k_f)
    d_eff = np.sqrt(d_km ** 2 + (alpha_eff * dh_m) ** 2)
    if kernel == "idw":
        w = 1.0 / np.power(d_eff + eps_km, p)
    elif kernel == "exponential":
        w = np.exp(-k_exp * d_eff)
    else:
        raise ValueError(kernel)
    # exact-hit safeguard
    hit = d_eff <= eps_km
    if hit.any():
        w = np.where(hit, 1.0, np.where(hit.any(axis=1, keepdims=True), 0.0, w))
    return _normalize_rows(w)


def weights_paper(pix, st_lat, st_lon, st_h, k_f, gamma=4.0,
                  elev=True, norm="global", eps=1e-12):
    """
    Paper eq. d3d + exp_weights.
      norm="global": d_horiz_max / d_elev_max over ALL pixel-station pairs (per frame).
      norm="pixel":  per-pixel max over its stations.
    elev=False -> no-elevation ablation (drop the elevation term entirely).
    gamma -> bandwidth (tunable; A3 ablation).
    """
    d_km, dh_m = _pixel_station_distances(pix, st_lat, st_lon, st_h)
    if norm == "global":
        dh_max = max(d_km.max(), eps)
        de_max = max(dh_m.max(), eps)
        dh_n = d_km / dh_max
        de_n = dh_m / de_max
    elif norm == "pixel":
        dh_n = d_km / np.maximum(d_km.max(axis=1, keepdims=True), eps)
        de_n = dh_m / np.maximum(dh_m.max(axis=1, keepdims=True), eps)
    else:
        raise ValueError(norm)
    if elev:
        d3d = np.sqrt(dh_n ** 2 + (float(k_f) * de_n) ** 2)
    else:
        d3d = dh_n
    w = np.exp(-gamma * d3d)
    return _normalize_rows(w)


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    # toy 1-pixel demo: 4 stations, one on a hill, to show elevation sensitivity
    np.random.seed(0)
    st_lat = np.array([32.0, 32.1, 31.9, 32.05])
    st_lon = np.array([34.8, 34.9, 34.7, 34.85])
    st_h = np.array([10.0, 800.0, 20.0, 50.0])     # station 1 is on a hill
    vals = np.array([30.0, 22.0, 31.0, 29.0])      # cooler on the hill
    pix = np.array([[34.82, 32.02, 600.0]])        # a high pixel near the hill

    for k_f in (2.3, 20.9):  # max_dry_temp vs tmp_air_wet-ish
        wc = weights_as_coded(pix, st_lat, st_lon, st_h, k_f)[0]
        wp = weights_paper(pix, st_lat, st_lon, st_h, k_f, gamma=4.0)[0]
        wpe = weights_paper(pix, st_lat, st_lon, st_h, k_f, gamma=4.0, elev=False)[0]
        print(f"\nk_f={k_f}")
        print("  as-coded  w:", np.round(wc, 3), "-> T=%.2f" % (wc @ vals))
        print("  paper     w:", np.round(wp, 3), "-> T=%.2f" % (wp @ vals))
        print("  paper noel:", np.round(wpe, 3), "-> T=%.2f" % (wpe @ vals))
    print("\n[note] as-coded barely shifts weight to the hill station even for the")
    print("       high pixel; paper-accurate (max-normalized, k_f on elevation)")
    print("       pulls the high pixel toward the cooler hill station -> genuinely")
    print("       elevation-aware. gamma controls smoothness (A3 bandwidth ablation).")
