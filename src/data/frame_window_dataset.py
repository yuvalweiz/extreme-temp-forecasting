"""
Train directly from an on-disk frames directory (no per-sample npz, disk-friendly).

A "frames dir" holds per-feature/day rasters:  <prefix>_<feature>_<YYYY-MM-DD>.npy
each (44,137). For a prediction point, the input window is the 180 daily frames
ending at that day, stacked over the 9 feature channels -> X (180, 9, 44, 137),
exactly matching the existing sample_*.npz. The y targets come from the cluster
order-statistic series (kernel-independent), so the SAME class trains on the
EXP / IDW / EXP_V2 / future paper-accurate frame sets just by changing frames_dir.

This is the infrastructure for the A3 interpolation ablation: point at a different
frames dir, recompute train-only norm stats, retrain, compare.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP

FRAME_DIRS = {k: RP.frames_dir(k, check=False) for k in ("EXP", "IDW", "EXP_V2")}
PREFIX = {"EXP": "exponential", "IDW": "idw", "EXP_V2": "exponential"}
HISTORY = 180


def _frame_path(frames_dir, prefix, feat, date):
    return os.path.join(frames_dir, f"{prefix}_{feat}_{pd.Timestamp(date).strftime('%Y-%m-%d')}.npy")


def load_window(frames_dir, prefix, feats, pred_point, history=HISTORY, cache=None):
    """Stack the `history` daily frames ending at pred_point -> (T, C, H, W).

    If `cache` (dict date_str -> (C,H,W) array) is given, read from it (fast,
    used for training); otherwise read .npy from disk (used for validation)."""
    days = pd.date_range(pd.Timestamp(pred_point) - pd.Timedelta(days=history - 1),
                         pd.Timestamp(pred_point), freq="D")
    T, C = len(days), len(feats)
    if cache is not None:
        Cc, H, W = next(iter(cache.values())).shape  # infer channels (may be 2C if anomaly-augmented)
        X = np.full((T, Cc, H, W), np.nan, np.float32)
        for ti, d in enumerate(days):
            ds = d.strftime("%Y-%m-%d")
            if ds in cache:
                X[ti] = cache[ds]
        return X
    h0 = np.load(_frame_path(frames_dir, prefix, feats[0], days[-1]))
    X = np.empty((T, C, *h0.shape), np.float32)
    for ti, d in enumerate(days):
        for ci, f in enumerate(feats):
            p = _frame_path(frames_dir, prefix, f, d)
            X[ti, ci] = np.load(p) if os.path.exists(p) else np.nan
    return X


def preload_frames(frames_dir, prefix, feats, date_lo, date_hi):
    """Load every daily (C,H,W) frame in [date_lo, date_hi] into a dict
    date_str -> (C,H,W). Each unique frame read once (windows overlap)."""
    cache = {}
    for d in pd.date_range(pd.Timestamp(date_lo), pd.Timestamp(date_hi), freq="D"):
        ds = d.strftime("%Y-%m-%d")
        chans = []
        ok = True
        for f in feats:
            p = _frame_path(frames_dir, prefix, f, d)
            if not os.path.exists(p):
                ok = False
                break
            chans.append(np.load(p))
        if ok:
            cache[ds] = np.stack(chans, 0).astype(np.float32)
    return cache


class FrameWindowDataset(Dataset):
    def __init__(self, frames_dir, prefix, feats, items, x_mean, x_std,
                 y_median, y_iqr, history=HISTORY, eps=1e-8, cache=None, spatial_mode="full"):
        """items: list of dicts with keys pred_point, tag, y (4,).
        cache: optional dict date_str -> (C,H,W) for fast in-memory loading.
        spatial_mode: 'full' | 'mean' (flatten each channel to its spatial mean ->
        destroys spatial structure, keeps per-channel value: tests if spatial info is used)."""
        self.frames_dir, self.prefix, self.feats = frames_dir, prefix, list(feats)
        self.items = items
        self.x_mean = x_mean.reshape(1, -1, 1, 1).astype(np.float32)
        self.x_std = x_std.reshape(1, -1, 1, 1).astype(np.float32)
        self.y_median, self.y_iqr = y_median.astype(np.float32), y_iqr.astype(np.float32)
        self.history, self.eps, self.cache = history, eps, cache
        self.spatial_mode = spatial_mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        X = load_window(self.frames_dir, self.prefix, self.feats, it["pred_point"],
                        self.history, cache=self.cache)
        X = np.nan_to_num(X, nan=0.0)
        if self.spatial_mode == "mean":  # flatten spatial structure (control: keeps channel value only)
            X = np.broadcast_to(X.mean(axis=(2, 3), keepdims=True), X.shape).copy()
        Xn = (X - self.x_mean) / (self.x_std + self.eps)
        y_raw = np.asarray(it["y"], np.float32)
        y_norm = (y_raw - self.y_median) / np.maximum(self.y_iqr, 1e-6)
        return (torch.from_numpy(Xn.astype(np.float32)), torch.from_numpy(y_norm),
                torch.from_numpy(y_raw), it["tag"])


# ---------------------------------------------------------------- validation
if __name__ == "__main__":
    DS = RP.canonical_dataset("Center")
    feats = open(f"{DS}/features_order.txt").read().split()
    for variant in ["EXP", "IDW", "EXP_V2"]:
        fd, px = FRAME_DIRS[variant], PREFIX[variant]
        for sd in ["sample_2005-07-01.npz", "sample_2015-08-21.npz"]:
            p = f"{DS}/{sd}"
            if not os.path.exists(p):
                continue
            s = np.load(p, allow_pickle=True)
            pp = pd.Timestamp(str(s["pred_point"]))
            Xw = load_window(fd, px, feats, pp)
            d = np.nanmax(np.abs(Xw - s["X"]))
            tag = "  <== SAMPLE SOURCE" if d < 1e-3 else ""
            print(f"  {variant:7s} {sd}: pred_point={pp.date()} max|Δ(window,sample)|={d:.5f}{tag}")
