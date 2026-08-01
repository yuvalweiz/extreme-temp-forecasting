"""
Daily station-availability masks for mask-aware frame synthesis (PAPERW_MASKS).

For each of the 68 stations (order = stationvec dim_order.json "station_order")
and each calendar day 2005-01-01..2025-06-30, mask[d, s, f] = True iff the RAW
station CSV has a real (non-NaN) value that day for feature f, using the SAME
daily aggregation convention as the original frame generator (min_* -> min,
max_* -> max, else mean — mirrors src/eval/loso_interpolation.py
load_station_series). Masks are per-feature because availability differs per
feature.

Output: station_masks.npz with
  mask          (D, 68, 9) bool
  dates         (D,) datetime64[D]  (contiguous daily index)
  station_names (68,) str
  features      (9,) str

Usage:
  python -m pipeline.build_station_masks
  (env: MASKS_CSV_DIR, MASKS_DIM_ORDER, MASKS_OUT override the defaults)
"""
import os
import json
import numpy as np
import pandas as pd

import sys
_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                                        # <repo>/src
import repo_paths as RP

CSV_DIR = os.environ.get("MASKS_CSV_DIR") or RP.stations_daily_dir(check=False)
DIM_ORDER = os.environ.get(
    "MASKS_DIM_ORDER",
    os.path.join(RP.stationvec_dir("Center", check=False), "dim_order.json"))
OUT = os.environ.get(
    "MASKS_OUT", os.path.join(RP.interp_weights_dir(), "station_masks.npz"))
START, END = "2005-01-01", "2025-06-30"


def _agg(feature):
    """Generator convention: min_* -> min, max_* -> max, else mean."""
    return "min" if feature.startswith("min_") else ("max" if feature.startswith("max_") else "mean")


def build_masks(csv_dir=CSV_DIR, dim_order_path=DIM_ORDER, start=START, end=END):
    with open(dim_order_path) as f:
        dim = json.load(f)
    names = [str(s) for s in dim["station_order"]]
    features = [str(f) for f in dim["feature_order"]]
    dates = pd.date_range(start, end, freq="D")
    mask = np.zeros((len(dates), len(names), len(features)), dtype=bool)

    for j, nm in enumerate(names):
        p = os.path.join(csv_dir, nm.replace(" ", "_") + ".csv")
        cols = set(features) | {"Date"}
        df = pd.read_csv(p, usecols=lambda c: c in cols)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        idx = df["Date"].dt.normalize()
        for k, feat in enumerate(features):
            if feat not in df.columns:
                continue
            s = pd.to_numeric(df[feat], errors="coerce")
            s.index = idx
            g = s.groupby(level=0)
            a = _agg(feat)
            s = g.max() if a == "max" else (g.min() if a == "min" else g.mean())
            s = s[(s.index >= start) & (s.index <= end)]
            v = s.reindex(dates).to_numpy(np.float32)
            mask[:, j, k] = np.isfinite(v)
        print(f"  [{j + 1:2d}/{len(names)}] {nm:24s} avail={mask[:, j].mean() * 100:5.1f}%", flush=True)
    return dates, names, features, mask


def main():
    dates, names, features, mask = build_masks()
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez_compressed(
        OUT,
        mask=mask,
        dates=dates.to_numpy().astype("datetime64[D]"),
        station_names=np.array(names),
        features=np.array(features),
        start=START, end=END,
        aggregation="min_*->min, max_*->max, else mean (per calendar day)")
    print(f"\n[masks] saved {mask.shape} bool -> {OUT} "
          f"({os.path.getsize(OUT) / 1e6:.2f} MB)")
    print(f"[masks] days={len(dates)} ({START}..{END}) stations={len(names)} features={len(features)}")
    print("\nper-feature availability:")
    for k, feat in enumerate(features):
        print(f"  {feat:18s} {mask[:, :, k].mean() * 100:6.2f}%")
    print(f"  {'OVERALL':18s} {mask.mean() * 100:6.2f}%")
    zero_days = (~mask.any(axis=1)).sum(axis=0)          # per-feature days with 0 stations
    print("days with ZERO available stations, per feature:",
          {f: int(z) for f, z in zip(features, zero_days)})


if __name__ == "__main__":
    main()
