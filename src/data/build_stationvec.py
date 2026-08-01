"""
Build the STATION-VECTOR datasets (data/stationvec_<Region>, data/stationvec_MIN_<Region>)
from the per-station daily CSVs — the documented spec in each dataset's dim_order.json:

  X (180, 612) float32   station-major: dim = station_index*9 + feature_index;
                         window = pd.date_range(end=tag-1day, periods=180)
  imputation             per (station,feature): forward-fill within the 180-day window;
                         remaining NaN -> that station-feature's TRAIN-split mean; if a
                         station-feature is all-missing in TRAIN -> the cross-station
                         TRAIN mean of that feature. No future/test info used.
  y (4,) float32         raw degC order statistics (p1/p3/p7/p15 hottest — or coldest for
                         MIN) of the next-30-day cluster-mean series. Taken verbatim from
                         the canonical frame dataset when available; otherwise recomputed
                         from the daily CSVs via the identical cluster-mean protocol
                         (src/baselines/series.py — validated to reproduce the stored
                         targets to 0.0000).
  splits                 the canonical chronological split CSVs (bundled under
                         data/dataset_meta/<Region>[_MIN]/), paths rewritten.
  norm_stats.npz         x_mean/x_std (612,) over TRAIN samples only; y_median/y_iqr/
                         y_transform copied from the canonical norm stats.

Inputs (all resolved via src/repo_paths.py):
  - per-station daily CSVs           (STATIONS_DIR / DATA_ROOT — from preprocessing/01)
  - grid_metadata.npz                (bundled: station order)
  - features_order.txt               (bundled: 9-channel order)
  - data/dataset_meta/<R>[_MIN]/     (bundled: split CSVs + norm stats)

Run:
  REGION=Center            python src/data/build_stationvec.py             # hot
  REGION=Center COLD=1     python src/data/build_stationvec.py             # winter/MIN
  OUT_DIR=/path            ... override the output dir (default data/stationvec_<R>)
  VALIDATE=/path/to/ref    ... additionally compare every sample against a reference
                               stationvec dir (max |dX|, |dy| reported)
"""
import os
import sys
import json
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                                         # <repo>/src
import repo_paths as RP

T_HIST = 180
KEEP_DOM = (1, 7, 14, 21, 28)


def _agg(feature):
    """Generator convention: min_* -> min, max_* -> max, else mean."""
    return "min" if feature.startswith("min_") else ("max" if feature.startswith("max_") else "mean")


def load_daily_matrix(station_order, features, start, end):
    """(D, S, F) float32 daily value cube from the per-station CSVs."""
    csv_dir = RP.stations_daily_dir()
    dates = pd.date_range(start, end, freq="D")
    V = np.full((len(dates), len(station_order), len(features)), np.nan, np.float32)
    for si, st in enumerate(station_order):
        p = os.path.join(csv_dir, st.replace(" ", "_") + ".csv")
        if not os.path.exists(p):
            print(f"  [warn] missing station CSV: {p}")
            continue
        df = pd.read_csv(p)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Date"] = df["Date"].dt.normalize()
        for fi, feat in enumerate(features):
            if feat not in df.columns:
                continue
            s = pd.to_numeric(df[feat], errors="coerce")
            s.index = df["Date"]
            g = s.groupby(level=0)
            how = _agg(feat)
            s = g.max() if how == "max" else (g.min() if how == "min" else g.mean())
            V[:, si, fi] = s.reindex(dates).to_numpy(np.float32)
    return dates, V


def load_split_tags(meta_dir, split, tagged):
    name = f"split_{split}_yNONE_v1.csv" if tagged else f"split_{split}.csv"
    df = pd.read_csv(os.path.join(meta_dir, name))
    df["tag"] = pd.to_datetime(df["tag"])
    return df.sort_values("tag").reset_index(drop=True)


def canonical_y(region, cold, tags):
    """{tag -> y (4,)}: verbatim from the canonical frame dataset when available,
    else recomputed from the daily CSVs via the identical series.py protocol."""
    can = RP.canonical_dataset(region, cold=cold, check=False)
    if os.path.isdir(can):
        out = {}
        for t in tags:
            p = os.path.join(can, f"sample_{t.strftime('%Y-%m-%d')}.npz")
            if os.path.exists(p):
                with np.load(p, allow_pickle=True) as z:
                    out[t] = z["y"].astype(np.float32)
        if len(out) == len(tags):
            print(f"  [y] verbatim from canonical dataset ({len(out)} tags)")
            return out
    # recompute from the daily CSVs (identical protocol; series.py reproduces to 0.0000)
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "baselines"))
    import series as S
    col = "min_dry_temp" if cold else "max_dry_temp"
    y = S.build_cluster_daily(region, cols=[col])[col].dropna().sort_index()
    out = {}
    for t in tags:
        fut = y.reindex(pd.date_range(t, t + pd.Timedelta(days=29), freq="D")).dropna()
        s = fut.sort_values(ascending=cold)          # descending hot / ascending cold
        if len(s) >= 15:
            out[t] = np.array([s.iloc[0], s.iloc[2], s.iloc[6], s.iloc[14]], np.float32)
    print(f"  [y] recomputed from daily CSVs ({len(out)}/{len(tags)} tags)")
    return out


def main():
    region = os.environ.get("REGION", "Center")
    cold = os.environ.get("COLD", "0") == "1"
    meta_dir = os.path.join(RP.REPO_DATA, "dataset_meta", region + ("_MIN" if cold else ""))
    if not os.path.isdir(meta_dir):
        meta_dir = RP.canonical_dataset(region, cold=cold)   # fall back to the real dataset
    tagged = os.path.exists(os.path.join(meta_dir, "split_train_yNONE_v1.csv"))
    out_dir = os.environ.get(
        "OUT_DIR", os.path.join(RP.REPO_DATA, ("stationvec_MIN_" if cold else "stationvec_") + region))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[build_stationvec] region={region} cold={cold}\n  meta={meta_dir}\n  out={out_dir}")

    g = np.load(RP.grid_metadata_npz(), allow_pickle=True)
    station_order = [str(s) for s in g["station_names"]]
    features = open(RP.features_order_txt()).read().split()
    S_, F_ = len(station_order), len(features)
    assert S_ * F_ == 612, (S_, F_)

    splits = {s: load_split_tags(meta_dir, s, tagged) for s in ("train", "val", "test")}
    all_tags = pd.concat([splits[s]["tag"] for s in splits]).sort_values()
    start = (all_tags.min() - pd.Timedelta(days=T_HIST)).normalize()
    end = (all_tags.max() + pd.Timedelta(days=31)).normalize()
    dates, V = load_daily_matrix(station_order, features, start, end)
    dix = {d: i for i, d in enumerate(dates)}

    # ---- TRAIN-only imputation means (no future/test info) ------------------------
    # "TRAIN-split mean" = mean over the union of the TRAIN samples' 180-day windows.
    tr_tags = splits["train"]["tag"]
    tr_lo = dix[(tr_tags.min() - pd.Timedelta(days=T_HIST)).normalize()]
    tr_hi = dix[(tr_tags.max() - pd.Timedelta(days=1)).normalize()]
    Vtr = V[tr_lo:tr_hi + 1]                                        # (Dtr, S, F)
    with np.errstate(invalid="ignore"):
        sf_mean = np.nanmean(Vtr, axis=0)                           # (S, F) station-feature
        f_mean = np.nanmean(Vtr, axis=(0, 1))                       # (F,)  cross-station
    fully_missing = [station_order[si] for si in range(S_) if np.isnan(sf_mean[si]).all()]
    sf_fill = np.where(np.isnan(sf_mean), f_mean[None, :], sf_mean).astype(np.float32)

    ymap = canonical_y(region, cold, list(all_tags))

    # ---- per-sample build ---------------------------------------------------------
    def build_X(tag):
        w_end = dix[(tag - pd.Timedelta(days=1)).normalize()]
        Xw = V[w_end - T_HIST + 1:w_end + 1].copy()                 # (180, S, F)
        # forward-fill within the window (per station-feature column)
        for t in range(1, T_HIST):
            m = np.isnan(Xw[t])
            Xw[t][m] = Xw[t - 1][m]
        Xw = np.where(np.isnan(Xw), sf_fill[None], Xw)              # TRAIN-mean fill
        return Xw.reshape(T_HIST, S_ * F_).astype(np.float32)

    n_missing_all = n_missing_tr = n_missing_te = 0
    n_vals_all = n_vals_tr = n_vals_te = 0
    written = {}
    for split, df in splits.items():
        kept = 0
        for tag in df["tag"]:
            if tag not in ymap:
                continue
            w_end_date = (tag - pd.Timedelta(days=1)).normalize()
            if w_end_date not in dix or (tag - pd.Timedelta(days=T_HIST)).normalize() not in dix:
                continue
            X = build_X(tag)
            raw = V[dix[w_end_date] - T_HIST + 1: dix[w_end_date] + 1]
            miss = int(np.isnan(raw).sum()); tot = raw.size
            n_missing_all += miss; n_vals_all += tot
            if split == "train":
                n_missing_tr += miss; n_vals_tr += tot
            if split == "test":
                n_missing_te += miss; n_vals_te += tot
            np.savez_compressed(os.path.join(out_dir, f"sample_{tag.strftime('%Y-%m-%d')}.npz"),
                                X=X, y=ymap[tag])
            kept += 1
        written[split] = kept
        rel = os.path.basename(os.path.normpath(out_dir))
        out_df = df[df["tag"].isin([t for t in df["tag"] if t in ymap])].copy()
        out_df["path"] = ["./" + rel + f"/sample_{t.strftime('%Y-%m-%d')}.npz" for t in out_df["tag"]]
        out_df["tag"] = out_df["tag"].dt.strftime("%Y-%m-%d")
        out_df[["path", "tag"]].to_csv(os.path.join(out_dir, f"split_{split}.csv"), index=False)
        print(f"  [{split}] {kept} samples")

    # ---- norm stats (TRAIN only) --------------------------------------------------
    tr_paths = pd.read_csv(os.path.join(out_dir, "split_train.csv"))["path"]
    acc_s = np.zeros(S_ * F_, np.float64); acc_q = np.zeros(S_ * F_, np.float64); n = 0
    for p in tr_paths:
        X = np.load(os.path.join(out_dir, os.path.basename(p)))["X"].astype(np.float64)
        acc_s += X.sum(0); acc_q += (X ** 2).sum(0); n += X.shape[0]
    x_mean = (acc_s / n).astype(np.float32)
    x_std = np.sqrt(np.maximum(acc_q / n - (acc_s / n) ** 2, 1e-8)).astype(np.float32)
    cn = np.load(os.path.join(
        meta_dir, "norm_stats_extremes_full_MIN.npz" if cold else "norm_stats_extremes_full_yNONE_v1.npz"),
        allow_pickle=True)
    np.savez(os.path.join(out_dir, "norm_stats.npz"),
             x_mean=x_mean, x_std=x_std,
             y_median=cn["y_median"].astype(np.float32), y_iqr=cn["y_iqr"].astype(np.float32),
             y_transform=np.array([str(cn["y_transform"]) if cn["y_transform"].shape == ()
                                   else str(cn["y_transform"][0])], object),
             C=S_ * F_, T=T_HIST,
             features=np.array(features, object), station_names=np.array(station_order, object))

    json.dump({
        "layout": "station-major: dim = station_index*9 + feature_index",
        "shape": [T_HIST, S_ * F_], "T": T_HIST, "C": S_ * F_,
        "n_stations": S_, "n_features": F_,
        "station_order": station_order, "feature_order": features,
        "windowing": "input window = pd.date_range(end=tag-1day, periods=180); matches frame input_days",
        "imputation": "per (station,feature): forward-fill within the 180-day window; remaining NaN "
                      "filled with that station-feature TRAIN-split mean; if station-feature all-missing "
                      "in TRAIN, filled with cross-station TRAIN mean for that feature. No future/test "
                      "info used.",
        "y_source": "reused verbatim from frame dataset sample_<tag>.npz y (raw degC order stats)",
        "normalization": "x_mean/x_std (612,) computed over TRAIN samples only; y_median/y_iqr/"
                         "y_transform copied from frame norm_stats",
    }, open(os.path.join(out_dir, "dim_order.json"), "w"), indent=1)
    json.dump({
        "missing_pct_overall": 100.0 * n_missing_all / max(n_vals_all, 1),
        "missing_pct_train": 100.0 * n_missing_tr / max(n_vals_tr, 1),
        "missing_pct_test": 100.0 * n_missing_te / max(n_vals_te, 1),
        "stations_fully_missing_in_train": fully_missing,
    }, open(os.path.join(out_dir, "imputation_summary.json"), "w"), indent=1)
    print(f"  [norm] x_mean/x_std over {written['train']} TRAIN samples; y stats copied")
    print(f"  [DONE] {sum(written.values())} samples -> {out_dir}")

    # ---- optional validation vs a reference stationvec dir ------------------------
    ref = os.environ.get("VALIDATE")
    if ref:
        dmax_x = dmax_y = 0.0; n_cmp = 0
        for split in ("train", "val", "test"):
            for p in pd.read_csv(os.path.join(out_dir, f"split_{split}.csv"))["path"]:
                b = os.path.basename(p)
                rp = os.path.join(ref, b)
                if not os.path.exists(rp):
                    continue
                a = np.load(os.path.join(out_dir, b)); r = np.load(rp)
                dmax_x = max(dmax_x, float(np.abs(a["X"] - r["X"]).max()))
                dmax_y = max(dmax_y, float(np.abs(a["y"] - r["y"]).max()))
                n_cmp += 1
        print(f"  [VALIDATE vs {ref}] {n_cmp} samples: max|dX|={dmax_x:.6f} max|dy|={dmax_y:.6f} "
              f"({'PASS' if dmax_x < 1e-3 and dmax_y < 1e-3 else 'CHECK'})")


if __name__ == "__main__":
    main()
