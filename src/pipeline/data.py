"""
Data — ported 1:1 from run_grid.py (CELL 1). Reads the per-sample .npz store
(X=(T,C,H,W) exp-interpolated frames, y=(4,) raw °C order stats), applies the
DOM filter {1,7,14,21,28}, per-channel x-normalization, and y-normalization
(median/IQR). Extreme threshold = TRAIN quantile (p95 published).
"""
import os
import json
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def load_norm_stats(dataset_dir, tag_norm="yNONE_v1"):
    norm_path = os.path.join(dataset_dir, f"norm_stats_extremes_full_{tag_norm}.npz")
    ns = np.load(norm_path, allow_pickle=True)
    return dict(
        x_mean=ns["x_mean"].astype(np.float32), x_std=ns["x_std"].astype(np.float32),
        y_median=ns["y_median"].astype(np.float32), y_iqr=ns["y_iqr"].astype(np.float32),
        y_transform=str(ns["y_transform"][0]),
        T=int(ns["T"]), C=int(ns["C"]), H=int(ns["H"]), W=int(ns["W"]),
    )


def load_split(dataset_dir, split, tag_norm="yNONE_v1", keep_dom=(1, 7, 14, 21, 28)):
    df = pd.read_csv(os.path.join(dataset_dir, f"split_{split}_{tag_norm}.csv"))
    df["tag"] = pd.to_datetime(df["tag"])
    df = df[df["tag"].dt.day.isin(set(keep_dom))].copy()
    return df.sort_values("tag").reset_index(drop=True)


def _abspath(dataset_dir, p):
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(os.path.dirname(dataset_dir), p)
    if os.path.exists(cand):
        return cand
    return os.path.join(dataset_dir, os.path.basename(p))


class FullSampleDataset(Dataset):
    """history_window: keep only the last W frames of the (T,C,H,W) clip (window ablation,
    stays inside the spatial pipeline — no rebuild). None = full T.
    y_override: optional {tag(YYYY-MM-DD): np.array(raw targets)} replacing the .npz y
    (target-set ablation; frames X unchanged)."""
    def __init__(self, df, dataset_dir, ns, keep_dom, history_window=None, y_override=None, cache=False, geo=None, tabular=False, lapse=None):
        self.ns = ns
        self.history_window = history_window
        self.y_override = y_override
        self.geo = geo                  # (G,H,W) static geo channels appended to every frame
        self.tabular = tabular          # True -> return spatial-mean series (T,C), no spatial structure
        self.lapse = lapse              # (H,W) lapse-rate correction added to raw temp frames (A3 fix)
        self._cache = None
        good, bad = [], 0
        for row in df.itertuples(index=False):
            p = _abspath(dataset_dir, row.path)
            try:
                with np.load(p, allow_pickle=True) as z:
                    if "X" in z.files and "y" in z.files:
                        good.append((p, row.tag))
                    else:
                        bad += 1
            except (zipfile.BadZipFile, OSError, EOFError, ValueError):
                bad += 1
        self.df = pd.DataFrame(good, columns=["path", "tag"]).sort_values("tag").reset_index(drop=True)
        if bad:
            print(f"[Dataset] skipped corrupted/missing-key files: {bad}")
        if len(self.df) == 0:
            raise RuntimeError("Dataset empty after filtering corrupted files.")
        if cache:   # preload normalized X + raw y into RAM once (kills per-epoch npz decompression)
            self._cache = [self._materialize(i) for i in range(len(self.df))]

    def __len__(self):
        return len(self.df)

    def _normalize_y(self, y_raw):
        y0 = (y_raw.astype(np.float32) - self.ns["y_median"]) / np.maximum(self.ns["y_iqr"], 1e-6)
        if self.ns["y_transform"] == "asinh":
            y0 = np.arcsinh(y0).astype(np.float32)
        return y0.astype(np.float32)

    def _materialize(self, idx):
        row = self.df.iloc[idx]
        tag = row["tag"].strftime("%Y-%m-%d")
        with np.load(row["path"], allow_pickle=True) as z:
            X = z["X"].astype(np.float32)
            y_raw = z["y"].astype(np.float32)
        if self.y_override is not None:
            y_raw = self.y_override[tag].astype(np.float32)
        C = self.ns["C"]
        if self.lapse is not None:
            X = X + self.lapse[None, None]                     # raw lapse-rate correction (A3), pre-norm
        X = (X - self.ns["x_mean"].reshape(1, C, 1, 1)) / (self.ns["x_std"].reshape(1, C, 1, 1) + 1e-8)
        if self.history_window is not None and self.history_window < X.shape[0]:
            X = X[-int(self.history_window):]
        if self.tabular:
            X = X.mean(axis=(2, 3)).astype(np.float32)        # (T,C) spatial-mean series
        elif self.geo is not None:
            g = np.broadcast_to(self.geo[None], (X.shape[0], *self.geo.shape))
            X = np.concatenate([X, g], axis=1).astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(self._normalize_y(y_raw)), torch.from_numpy(y_raw), tag

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._materialize(idx)


class MmapDataset(Dataset):
    """Reads samples from the precomputed mmap arrays (uncompressed fp16) — shared OS
    page cache across runs, normalizes on the fly (cheap). Supports window slice + y_override."""
    def __init__(self, cdir, split, ns, history_window=None, y_override=None, geo=None, tabular=False, lapse=None):
        self.ns = ns; self.history_window = history_window; self.y_override = y_override; self.geo = geo; self.tabular = tabular; self.lapse = lapse
        self.X = np.load(os.path.join(cdir, f"{split}_X.npy"), mmap_mode="r")
        self.Y = np.load(os.path.join(cdir, f"{split}_y.npy"), mmap_mode="r")
        self.tags = json.load(open(os.path.join(cdir, f"{split}_tags.json")))
        self.xm = ns["x_mean"].reshape(1, ns["C"], 1, 1); self.xs = ns["x_std"].reshape(1, ns["C"], 1, 1)

    def __len__(self): return len(self.tags)

    def _normalize_y(self, y_raw):
        y0 = (y_raw.astype(np.float32) - self.ns["y_median"]) / np.maximum(self.ns["y_iqr"], 1e-6)
        return (np.arcsinh(y0) if self.ns["y_transform"] == "asinh" else y0).astype(np.float32)

    def __getitem__(self, idx):
        tag = self.tags[idx]
        X = np.asarray(self.X[idx], dtype=np.float32)
        y_raw = (self.y_override[tag].astype(np.float32) if self.y_override is not None
                 else np.asarray(self.Y[idx], dtype=np.float32))
        if self.lapse is not None:
            X = X + self.lapse[None, None]
        X = (X - self.xm) / (self.xs + 1e-8)
        if self.history_window is not None and self.history_window < X.shape[0]:
            X = X[-int(self.history_window):]
        if self.tabular:
            X = X.mean(axis=(2, 3)).astype(np.float32)
        elif self.geo is not None:
            g = np.broadcast_to(self.geo[None], (X.shape[0], *self.geo.shape))
            X = np.concatenate([X, g], axis=1).astype(np.float32)
        return (torch.from_numpy(X), torch.from_numpy(self._normalize_y(y_raw)),
                torch.from_numpy(y_raw), tag)


class StationVecDataset(Dataset):
    """Fair non-spatial control: reads (T, n_stations*n_features) flat station vectors —
    SAME 68 stations x 9 features x 180-day daily history as the spatial frames, same tags/
    targets, differing ONLY in representation (no grid/interpolation). Returns (T,F) for the
    temporal head (HeadOnlyModel).
    y_override: optional {tag(YYYY-MM-DD): np.array(raw targets)} replacing the .npz y
    (target-set ablation, TARGETS env; X unchanged). None -> behavior identical to before."""
    def __init__(self, df, dataset_dir, ns, y_override=None):
        self.df = df.reset_index(drop=True); self.dir = dataset_dir; self.ns = ns
        self.y_override = y_override
        self.xm = ns["x_mean"].reshape(1, -1).astype(np.float32)
        self.xs = ns["x_std"].reshape(1, -1).astype(np.float32)

    def __len__(self): return len(self.df)

    def _norm_y(self, yr):
        y0 = (yr.astype(np.float32) - self.ns["y_median"]) / np.maximum(self.ns["y_iqr"], 1e-6)
        return (np.arcsinh(y0) if str(self.ns["y_transform"]) == "asinh" else y0).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; tag = str(row["tag"])
        p = os.path.join(self.dir, os.path.basename(row["path"]))
        z = np.load(p); X = z["X"].astype(np.float32); y_raw = z["y"].astype(np.float32)
        if self.y_override is not None:
            y_raw = self.y_override[tag].astype(np.float32)
        X = (X - self.xm) / (self.xs + 1e-8)
        return torch.from_numpy(X), torch.from_numpy(self._norm_y(y_raw)), torch.from_numpy(y_raw), tag


def build_stationvec_data(cfg, dataset_dir, y_override=None):
    """Loaders/ns/hot_thr for the flat station-vector dataset (fair tabular ablation).
    y_transform is read element-wise (['asinh'] -> 'asinh'), so cold (MIN) stationvec dirs
    normalize y exactly like the frame pipeline; hot dirs ('none') are unaffected.
    y_override (TARGETS env): {tag: raw target vector}; y-norm stats (median/IQR) and
    hot_thr are recomputed from the TRAIN override targets (mirrors build_data), split
    rows without an override entry are dropped (printed). None -> identical to before."""
    z = np.load(os.path.join(dataset_dir, "norm_stats.npz"), allow_pickle=True)
    ns = {"x_mean": z["x_mean"], "x_std": z["x_std"], "y_median": z["y_median"],
          "y_iqr": z["y_iqr"],
          "y_transform": str(z["y_transform"][0]) if z["y_transform"].shape else str(z["y_transform"]),
          "C": int(z["C"]), "T": int(z["T"]), "H": 1, "W": 1}
    dfs = {s: pd.read_csv(os.path.join(dataset_dir, f"split_{s}.csv")) for s in ["train", "val", "test"]}
    if y_override is not None:   # TARGETS: swap targets + TRAIN-override y-stats/threshold
        n0 = {s: len(dfs[s]) for s in dfs}
        dfs = {s: dfs[s][[str(t) in y_override for t in dfs[s]["tag"]]].reset_index(drop=True)
               for s in dfs}
        drops = {s: n0[s] - len(dfs[s]) for s in dfs if n0[s] != len(dfs[s])}
        if drops:
            print(f"[stationvec] y_override: dropped tags without override targets {drops}")
        Ytr = np.stack([y_override[str(t)] for t in dfs["train"]["tag"]]).astype(np.float32)
        ns = dict(ns)
        ns["y_median"] = np.median(Ytr, 0).astype(np.float32)
        ns["y_iqr"] = (np.quantile(Ytr, 0.75, 0) - np.quantile(Ytr, 0.25, 0)).astype(np.float32)
        hot_thr = float(np.quantile(Ytr[:, 0], cfg.extreme_q))
    ds = {s: StationVecDataset(dfs[s], dataset_dir, ns, y_override=y_override) for s in dfs}
    if y_override is None:
        ytr = np.stack([np.load(os.path.join(dataset_dir, os.path.basename(p)))["y"].astype(np.float32)
                        for p in dfs["train"]["path"]])
        hot_thr = float(np.quantile(ytr[:, 0], cfg.extreme_q))
    def dl(split, shuf): return DataLoader(ds[split], batch_size=cfg.batch_size, shuffle=shuf,
                                           num_workers=0, pin_memory=True)
    loaders = {"train": dl("train", True), "val": dl("val", False), "test": dl("test", False)}
    print(f"[stationvec] train/val/test={len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])} "
          f"| F={ns['C']} (68 stations x 9 feat) T={ns['T']} | hot_thr={hot_thr:.3f}")
    return loaders, ns, hot_thr


# ---------------------------------------------------------------- synthesized frames (PAPERW)
def _synth_frames(sv, Wt, gh, gw, feature_map=None):
    """Synthesize raw frames from one flat station-vector sample.
    sv (T, S*C) station-major (dim = station*C + feature); Wt (C, S, P) = W.transpose(0,2,1)
    with W row-stochastic (C, P=gh*gw, S). Returns X (T, C, gh, gw) float32:
    X[t,c] = (W[c] @ sv[t,:,c]).reshape(gh, gw). One BLAS batched matmul — no frame bank.
    feature_map (opt-in, MULTI-KERNEL STACKED weights from stack_weights.py): int array
    (C,) mapping each W-channel to its source sv-feature index, so C may exceed the sv
    feature count C0: X[t,c] = W[c] @ sv[t,:,feature_map[c]]. None -> unchanged."""
    C, S, P = Wt.shape
    T = sv.shape[0]
    if feature_map is None:
        s = np.ascontiguousarray(sv.reshape(T, S, C).transpose(2, 0, 1)).astype(np.float32)  # (C,T,S)
    else:
        C0 = sv.shape[1] // S
        s = np.ascontiguousarray(
            sv.reshape(T, S, C0).transpose(2, 0, 1)[feature_map]).astype(np.float32)         # (C,T,S)
    X = np.matmul(s, Wt)                                                                 # (C,T,P)
    return np.ascontiguousarray(X.transpose(1, 0, 2)).reshape(T, C, gh, gw)


# ---------------------------------------------------------------- mask-aware synthesis (PAPERW_MASKS)
_MASK_LOGGED = set()


def _mask_log_once(key, msg):
    if key not in _MASK_LOGGED:
        _MASK_LOGGED.add(key)
        print(msg, flush=True)


def _load_station_masks(masks_path, station_names=None, features=None):
    """Load build_station_masks.py output -> (mask (D,S,C) bool, first-date Timestamp).
    Validates station/feature order against the weights file; requires a contiguous
    daily date index (so window rows can be sliced arithmetically)."""
    mz = np.load(masks_path, allow_pickle=True)
    mask = mz["mask"].astype(bool)
    dates = pd.DatetimeIndex(mz["dates"])
    assert len(dates) == mask.shape[0] and \
        (dates == pd.date_range(dates[0], periods=len(dates), freq="D")).all(), \
        "station masks date index must be contiguous daily"
    if station_names is not None:
        assert [str(s) for s in mz["station_names"]] == [str(s) for s in station_names], \
            "masks/weights station order mismatch"
    if features is not None:
        assert [str(f) for f in mz["features"]] == [str(f) for f in features], \
            "masks/weights feature order mismatch"
    return mask, dates[0]


def _sample_mask_window(masks, tag, T):
    """(T,S,C) bool availability rows for one sample: the station-vector window is
    pd.date_range(end=tag-1d, periods=T), so its first day is tag - T days.
    Returns None when the window falls outside the mask date range."""
    mask, start = masks
    i0 = (pd.Timestamp(tag) - start).days - T
    if i0 < 0 or i0 + T > mask.shape[0]:
        return None
    return mask[i0:i0 + T]


def _synth_frames_masked(sv, Wt, gh, gw, M, feature_map=None):
    """Mask-aware synthesis — the ORIGINAL frame generator's missing-data convention:
    per day/feature, drop missing stations and renormalize the weights over the
    available ones: X[t,c] = (W[c] * m) @ v / max((W[c] * m) @ 1, eps).
    sv (T, S*C) station-major; Wt (C, S, P); M (T, S, C) bool availability.
    Imputed sv values get ZERO weight wherever M is False (only real values count).
    Guard: a day+feature with zero available stations falls back to unmasked
    synthesis (W is row-stochastic), logged once.
    feature_map (opt-in, stacked weights): as in _synth_frames — sv columns AND mask
    columns are indexed per W-channel (masks stay per BASE feature). None -> unchanged."""
    C, S, P = Wt.shape
    T = sv.shape[0]
    if feature_map is None:
        s = np.ascontiguousarray(sv.reshape(T, S, C).transpose(2, 0, 1)).astype(np.float32)   # (C,T,S)
        m = np.ascontiguousarray(M.transpose(2, 0, 1)).astype(np.float32)                     # (C,T,S)
        empty_ct = (~M.any(axis=1)).T                                                         # (C,T)
    else:
        C0 = sv.shape[1] // S
        s = np.ascontiguousarray(
            sv.reshape(T, S, C0).transpose(2, 0, 1)[feature_map]).astype(np.float32)          # (C,T,S)
        m = np.ascontiguousarray(M.transpose(2, 0, 1)[feature_map]).astype(np.float32)        # (C,T,S)
        empty_ct = (~M.any(axis=1)).T[feature_map]                                            # (C,T)
    num = np.matmul(s * m, Wt)                                                            # (C,T,P)
    den = np.matmul(m, Wt)                                                                # (C,T,P)
    X = num / np.maximum(den, 1e-8)
    if empty_ct.any():
        _mask_log_once("empty", f"[synth mask] {int(empty_ct.sum())} day-feature slot(s) with 0 "
                                "available stations -> unmasked fallback for those slots")
        ec, et = np.nonzero(empty_ct)                                                     # (C,T) coords
        X[ec, et] = np.matmul(s[ec, et][:, None, :], Wt[ec])[:, 0]
    return np.ascontiguousarray(X.transpose(1, 0, 2)).reshape(T, C, gh, gw)


# ------------------------------------------------- frame-representation upgrades (opt-in)
def _window_doy_idx(tag, T):
    """Day-of-year index (0..365) for each day of a sample's station-vector window
    pd.date_range(end=tag-1d, periods=T) — the SAME window convention as
    _sample_mask_window. Indexes the (366, C, H, W) DOY climatology (anomaly mode)."""
    days = pd.date_range(end=pd.Timestamp(tag) - pd.Timedelta(days=1), periods=T)
    return days.dayofyear.to_numpy() - 1


def _coverage_channel(M, cov_anchor, gh, gw, T):
    """DYNAMIC daily coverage channel (PAPERW_COV=1): per-day per-pixel effective kernel
    coverage of the anchor feature's elevation-aware kernel — the masked-synthesis
    denominator (W_anchor ⊙ mask_t) @ 1, in [0,1] because W is row-stochastic. Tells the
    CNN where today's interpolation is data-poor. cov_anchor = (Wt_anchor (S,P) float32,
    base-feature column into M). M=None (no masks / window outside range) -> all-ones."""
    if M is None:
        return np.ones((T, 1, gh, gw), np.float32)
    Wa, fidx = cov_anchor
    cov = M[:, :, fidx].astype(np.float32) @ Wa                    # (T,S)@(S,P) -> (T,P)
    return np.ascontiguousarray(cov.reshape(T, 1, gh, gw))


def _synth_doy_climatology(sv_dir, weights_path, Wt, gh, gw, df_train, region, masks=None,
                           feature_map=None):
    """Per-pixel DAY-OF-YEAR climatology of the RAW synthesized frames (anomaly mode,
    PAPERW_ANOM=1) — computed ONCE per variant+region from TRAIN samples only (same ::4
    streaming as _synth_norm_stats), accumulating per-DOY sums over every window day,
    then smoothed with a ±7-day CIRCULAR window (fills sparse DOYs, e.g. Feb 29); any
    DOY still empty falls back to the all-train mean. Subtracting clim[doy] removes the
    seasonal cycle so the CNN sees synoptic anomalies (what drives extremes).
    Cached as <weights stem>_clim_<region>[_masked].npz, clim fp16 (366, C, H, W);
    masks!=None -> climatology of the MASK-AWARE frames (never mixes caches).
    Returns (clim fp16 (366,C,H,W), path)."""
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    suffix = "_masked" if masks is not None else ""
    cpath = os.path.join(os.path.dirname(os.path.abspath(weights_path)),
                         f"{stem}_clim_{region}{suffix}.npz")
    if os.path.exists(cpath):
        return np.load(cpath)["clim"], cpath
    C = Wt.shape[0]
    csum = np.zeros((366, C, gh * gw), np.float32)
    cnt = np.zeros(366, np.float64)
    sub = df_train.iloc[::4]
    for i, (_, row) in enumerate(sub.iterrows()):
        sv = np.load(os.path.join(sv_dir, os.path.basename(row["path"])))["X"].astype(np.float32)
        M = _sample_mask_window(masks, str(row["tag"]), sv.shape[0]) if masks is not None else None
        xf = (_synth_frames_masked(sv, Wt, gh, gw, M, feature_map) if M is not None
              else _synth_frames(sv, Wt, gh, gw, feature_map))
        didx = _window_doy_idx(str(row["tag"]), sv.shape[0])
        xf = xf.reshape(sv.shape[0], C, gh * gw)
        if didx.size == np.unique(didx).size:               # window days are distinct DOYs (T<=366)
            csum[didx] += xf
        else:                                               # window > 1 year: duplicate-safe path
            np.add.at(csum, didx, xf)
        cnt += np.bincount(didx, minlength=366)
        if i % 40 == 0:
            print(f"  [synth clim] {i}/{len(sub)}", flush=True)
    ssum = sum(np.roll(csum, sft, axis=0) for sft in range(-7, 8))     # ±7d circular box
    scnt = sum(np.roll(cnt, sft) for sft in range(-7, 8))
    fallback = csum.sum(0) / max(cnt.sum(), 1.0)                       # all-train mean (C,P)
    clim = np.where(scnt[:, None, None] > 0,
                    ssum / np.maximum(scnt, 1.0)[:, None, None], fallback[None])
    assert np.isfinite(clim).all(), "DOY climatology has non-finite values"
    clim = clim.reshape(366, C, gh, gw).astype(np.float16)
    np.savez(cpath, clim=clim, doy_smooth_halfwin=7, region=region, weights=stem,
             masked=(masks is not None), n_samples=len(sub))
    print(f"[synth clim] computed from {len(sub)} TRAIN samples -> {cpath} "
          f"({os.path.getsize(cpath) / 1e6:.1f} MB)", flush=True)
    return clim, cpath


class SynthFrameDataset(Dataset):
    """MEMORY-WISE interpolation ablation: wraps a station-vector dataset dir + a precomputed
    per-feature weight matrix (compute_interp_weights.py) and synthesizes the (T,C,H,W) frames
    on the fly — the paper-accurate (or IDW) frames are never materialized on disk.
    Same tags/targets/y-normalization as the frame datasets; x_mean/x_std are per-VARIANT.
    cache=True stores the normalized frames fp16 in RAM (fp32 on __getitem__ exit).
    masks=(mask (D,S,C) bool, first-date) from _load_station_masks -> MASK-AWARE synthesis
    (per-day renormalization over available stations); masks=None -> unchanged behavior.
    return_sv=True (HYBRID mode, opt-in): the first element becomes the PAIR
    (frames (T,C,H,W), raw station vector (T,S*C) normalized with sv_norm=(mean,std) —
    the stationvec dir's norm_stats). Use hybrid_collate in the DataLoader so batches
    arrive as a HybridBatch (duck-types .to/.float, so train/eval loops run unchanged).
    return_sv=False -> byte-identical to before.
    Frame-representation upgrades (each opt-in, unset -> byte-identical to before):
    feature_map (stacked multi-kernel weights): per-W-channel sv-feature index;
    clim (anomaly mode): (366,C,H,W) fp16 DOY climatology subtracted from the RAW frames
    (before normalization; ns must hold `_anom` stats);
    coverage + cov_anchor (coverage mode): appends the per-day kernel-coverage channel
    (ns must hold C+1 `_cov` stats). All composable with masks/hybrid/cache."""
    def __init__(self, df, dataset_dir, ns, Wt, cache=False, masks=None,
                 return_sv=False, sv_norm=None, feature_map=None, clim=None,
                 coverage=False, cov_anchor=None, y_override=None):
        self.y_override = y_override        # {tag: raw targets} (TARGETS ablation)
        self.df = df.reset_index(drop=True); self.dir = dataset_dir; self.ns = ns
        self.Wt = Wt                                        # (C, S, H*W) float32
        self.masks = masks
        self.feature_map = feature_map      # stacked weights: W-channel -> sv-feature index
        self.clim = clim                    # (366,C,H,W) fp16 DOY climatology (anomaly mode)
        self.coverage = bool(coverage)      # append per-day kernel-coverage channel
        self.cov_anchor = cov_anchor        # (Wt_anchor (S,P), base-feature col) for coverage
        self.return_sv = bool(return_sv)
        if self.return_sv:
            assert sv_norm is not None, "return_sv=True requires sv_norm=(mean, std)"
            self.sv_mean = np.asarray(sv_norm[0], np.float32).reshape(1, -1)
            self.sv_std = np.asarray(sv_norm[1], np.float32).reshape(1, -1)
        self._cache = None
        if cache:
            if self.return_sv:   # frames fp16 + sv fp32 (sv is small) in RAM
                self._cache = [((X[0].half(), X[1]), yn, yr, t) for X, yn, yr, t in
                               (self._materialize(i) for i in range(len(self.df)))]
            else:
                self._cache = [(X.half(), yn, yr, t) for X, yn, yr, t in
                               (self._materialize(i) for i in range(len(self.df)))]

    def __len__(self): return len(self.df)

    def _norm_y(self, yr):
        y0 = (yr.astype(np.float32) - self.ns["y_median"]) / np.maximum(self.ns["y_iqr"], 1e-6)
        return (np.arcsinh(y0) if str(self.ns["y_transform"]) == "asinh" else y0).astype(np.float32)

    def _materialize(self, idx):
        row = self.df.iloc[idx]; tag = str(row["tag"])
        z = np.load(os.path.join(self.dir, os.path.basename(row["path"])))
        sv = z["X"].astype(np.float32)
        M = None
        if self.masks is not None:
            M = _sample_mask_window(self.masks, tag, sv.shape[0])
            if M is None:
                _mask_log_once("range", f"[synth mask] window for tag={tag} outside mask "
                                        "date range -> unmasked synthesis for such samples")
        X = (_synth_frames_masked(sv, self.Wt, self.ns["H"], self.ns["W"], M, self.feature_map)
             if M is not None
             else _synth_frames(sv, self.Wt, self.ns["H"], self.ns["W"], self.feature_map))
        if self.clim is not None:   # ANOMALY: subtract DOY climatology in RAW space, pre-norm
            X = X - self.clim[_window_doy_idx(tag, sv.shape[0])].astype(np.float32)
        if self.coverage:           # +1 dynamic channel: today's effective kernel coverage
            X = np.concatenate([X, _coverage_channel(M, self.cov_anchor, self.ns["H"],
                                                     self.ns["W"], sv.shape[0])], axis=1)
        y_raw = (self.y_override[tag].astype(np.float32) if self.y_override is not None
                 else z["y"].astype(np.float32))
        C = self.ns["C"]
        X = (X - self.ns["x_mean"].reshape(1, C, 1, 1)) / (self.ns["x_std"].reshape(1, C, 1, 1) + 1e-8)
        Xt = torch.from_numpy(X)
        if self.return_sv:   # HYBRID: pair (frames, normalized raw station vector)
            sv_n = (sv - self.sv_mean) / (self.sv_std + 1e-8)
            Xt = (Xt, torch.from_numpy(sv_n.astype(np.float32)))
        return (Xt, torch.from_numpy(self._norm_y(y_raw)),
                torch.from_numpy(y_raw), tag)

    def __getitem__(self, idx):
        if self._cache is not None:
            X, yn, yr, tag = self._cache[idx]
            if self.return_sv:
                return (X[0].float(), X[1]), yn, yr, tag
            return X.float(), yn, yr, tag
        return self._materialize(idx)


# ---------------------------------------------------------------- HYBRID (frames + raw sv)
class HybridBatch:
    """Batch container for HYBRID mode: frames (B,T,C,H,W) + station vectors (B,T,F_sv).
    Duck-types the tensor calls the existing train/eval loops make on X (.to(...),
    .float(), indexing, pin_memory), so run_one_config / collect_pred_table need NO
    changes — model(X) receives the pair and ConvNeXtTiny_Hybrid.forward unpacks it."""
    __slots__ = ("frames", "sv")

    def __init__(self, frames, sv):
        self.frames, self.sv = frames, sv

    def to(self, *args, **kwargs):
        return HybridBatch(self.frames.to(*args, **kwargs), self.sv.to(*args, **kwargs))

    def float(self):
        return HybridBatch(self.frames.float(), self.sv.float())

    def pin_memory(self):
        return HybridBatch(self.frames.pin_memory(), self.sv.pin_memory())

    def __getitem__(self, idx):
        return HybridBatch(self.frames[idx], self.sv[idx])

    def __len__(self):
        return len(self.frames)

    @property
    def shape(self):
        return self.frames.shape


def hybrid_collate(batch):
    """Default collate on ((Xf, Xsv), yn, yr, tag) samples, then wrap X as a HybridBatch."""
    from torch.utils.data import default_collate
    X, yn, yr, tags = default_collate(batch)
    return HybridBatch(X[0], X[1]), yn, yr, list(tags)


def _synth_norm_stats(sv_dir, weights_path, Wt, gh, gw, df_train, region, masks=None,
                      feature_map=None, clim=None, coverage=False, cov_anchor=None):
    """Per-channel mean/std of the RAW synthesized frames for THIS weights variant —
    streamed over every 4th TRAIN sample (compute_lapse.py convention), cached as
    <weights stem>_norm_<region>.npz next to the weights file (small).
    masks!=None -> stats of the MASK-AWARE frames, cached with a `_masked` suffix
    (never reuses/overwrites the unmasked cache).
    Opt-in frame-representation modes, each with its OWN cache suffix (composable,
    never mixing caches): feature_map (stacked weights — flows through synthesis, C
    follows Wt.shape[0]); clim (`_anom`: stats of the climatology-subtracted frames);
    coverage+cov_anchor (`_cov`: stats over C+1 channels incl. the coverage channel)."""
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    suffix = (("_masked" if masks is not None else "") + ("_cov" if coverage else "")
              + ("_anom" if clim is not None else ""))
    npath = os.path.join(os.path.dirname(os.path.abspath(weights_path)),
                         f"{stem}_norm_{region}{suffix}.npz")
    if os.path.exists(npath):
        z = np.load(npath)
        return z["x_mean"].astype(np.float32), z["x_std"].astype(np.float32), npath
    C = Wt.shape[0] + (1 if coverage else 0)
    xsum = np.zeros(C, np.float64); xsq = np.zeros(C, np.float64); n = 0
    sub = df_train.iloc[::4]
    for i, (_, row) in enumerate(sub.iterrows()):
        sv = np.load(os.path.join(sv_dir, os.path.basename(row["path"])))["X"].astype(np.float32)
        M = _sample_mask_window(masks, str(row["tag"]), sv.shape[0]) if masks is not None else None
        xf = (_synth_frames_masked(sv, Wt, gh, gw, M, feature_map) if M is not None
              else _synth_frames(sv, Wt, gh, gw, feature_map))              # (T,C,H,W) raw
        if clim is not None:        # ANOMALY: same RAW-space subtraction as at synth time
            xf = xf - clim[_window_doy_idx(str(row["tag"]), sv.shape[0])].astype(np.float32)
        if coverage:                # coverage channel gets its own train-only mean/std
            xf = np.concatenate([xf, _coverage_channel(M, cov_anchor, gh, gw, sv.shape[0])],
                                axis=1)
        xf = xf.astype(np.float64)
        xsum += xf.sum((0, 2, 3)); xsq += (xf ** 2).sum((0, 2, 3)); n += xf[:, 0].size
        if i % 40 == 0:
            print(f"  [synth norm] {i}/{len(sub)}", flush=True)
    x_mean = (xsum / n).astype(np.float32)
    x_std = np.sqrt(np.maximum(xsq / n - (xsum / n) ** 2, 1e-8)).astype(np.float32)
    np.savez(npath, x_mean=x_mean, x_std=x_std, n_frames=n, region=region, weights=stem,
             masked=(masks is not None), coverage=bool(coverage), anom=(clim is not None))
    print(f"[synth norm] computed from {len(sub)} TRAIN samples -> {npath}")
    return x_mean, x_std, npath


def build_synth_data(cfg, sv_dir, weights_path, cache=False, masks_path=None, hybrid=False,
                     y_override=None,
                     coverage=None, anomaly=None):
    """Loaders/ns/hot_thr for SYNTHESIZED frames (PAPERW mode): station-vector dataset x
    precomputed interpolation weights — the interpolation ablation without frame banks.
    y-normalization/targets identical to stationvec (== frame dataset y); x_mean/x_std
    computed-or-loaded per variant; hot_thr = TRAIN extreme_q quantile of y[:,0]
    (same as build_stationvec_data).
    masks_path (opt-in, env PAPERW_MASKS at the train.py call site): station availability
    masks from build_station_masks.py -> MASK-AWARE synthesis (per-day weight
    renormalization over available stations, the original frame generator's convention);
    None -> behavior identical to before.
    hybrid=True (opt-in, env HYBRID at the train.py call site): each batch is a
    HybridBatch(frames, raw station vectors) — sv normalized with the stationvec dir's
    norm_stats (x_mean/x_std, 612) — for ConvNeXtTiny_Hybrid; ns gains 'sv_dim'.
    hybrid=False -> behavior identical to before.
    FRAME-REPRESENTATION upgrades (all elevation-aware, opt-in, unset -> byte-identical):
    * STACKED multi-kernel weights (stack_weights.py output, auto-detected via the
      `feature_map` key): W (C,P,S) concatenates several kernels per feature; channel c
      interpolates sv feature feature_map[c]; ns['C']/in_chans follow W.shape[0].
    * coverage (env PAPERW_COV=1, read HERE so train.py stays untouched; None=env):
      append ONE dynamic channel = per-day effective kernel coverage of the
      max_dry_temp anchor kernel, (W_anchor ⊙ mask_t) @ 1 in [0,1] — all-ones + warning
      without masks — with its own train-only stats (`_cov` norm suffix); C -> C+1.
    * anomaly (env PAPERW_ANOM=1; None=env): subtract the per-pixel DOY climatology
      (TRAIN-only, _synth_doy_climatology, <variant>_clim_<region>[_masked].npz) from
      each RAW frame day before normalization (`_anom` norm suffix) — the CNN sees
      synoptic anomalies instead of the seasonal cycle. Composes with masks/coverage."""
    if coverage is None:
        coverage = os.environ.get("PAPERW_COV", "0") == "1"
    if anomaly is None:
        anomaly = os.environ.get("PAPERW_ANOM", "0") == "1"
    wz = np.load(weights_path, allow_pickle=True)
    W = wz["W"].astype(np.float32)                                  # (C, H*W, S) row-stochastic
    gh, gw = int(wz["grid_h"]), int(wz["grid_w"])
    variant = str(wz["variant"]) if "variant" in wz.files else os.path.basename(weights_path)
    feature_map = wz["feature_map"].astype(np.int64) if "feature_map" in wz.files else None
    # stacked files carry per-kernel channel names in `features`; validation against the
    # stationvec/masks feature order uses the shared base list (`base_features`)
    base_features = (wz["base_features"] if feature_map is not None and "base_features" in wz.files
                     else wz["features"])
    z = np.load(os.path.join(sv_dir, "norm_stats.npz"), allow_pickle=True)
    # the weight station/feature order MUST match the station-vector layout
    assert [str(s) for s in wz["station_names"]] == [str(s) for s in z["station_names"]], \
        "weights/stationvec station order mismatch"
    assert [str(f) for f in base_features] == [str(f) for f in z["features"]], \
        "weights/stationvec feature order mismatch"
    if feature_map is not None:
        assert feature_map.shape == (W.shape[0],) and feature_map.min() >= 0 \
            and feature_map.max() < len(base_features), "bad feature_map in stacked weights"
        src = (f" ({', '.join(str(v) for v in wz['source_variants'])})"
               if "source_variants" in wz.files else "")
        print(f"[synth] STACKED multi-kernel weights: C={W.shape[0]} channels over "
              f"{len(base_features)} base features{src}")
    Wt = np.ascontiguousarray(W.transpose(0, 2, 1))                 # (C, S, H*W) for matmul

    masks = None
    if masks_path:
        masks = _load_station_masks(masks_path, wz["station_names"], base_features)
        print(f"[synth] MASK-AWARE synthesis ON: {os.path.basename(masks_path)} "
              f"(days={masks[0].shape[0]}, availability={masks[0].mean() * 100:.1f}%)")

    cov_anchor = None
    if coverage:   # anchor = the max_dry_temp channel's elevation-aware kernel
        feats = [str(f) for f in base_features]
        fidx = feats.index("max_dry_temp") if "max_dry_temp" in feats else 0
        ch = int(np.nonzero(feature_map == fidx)[0][0]) if feature_map is not None else fidx
        cov_anchor = (np.ascontiguousarray(Wt[ch]), fidx)
        if masks is None:
            print("[synth] WARNING: coverage=1 (PAPERW_COV) without station masks -> coverage "
                  "channel is ALL-ONES (set PAPERW_MASKS for a meaningful coverage map)")
        print(f"[synth] COVERAGE channel ON: anchor='{feats[fidx]}' (W channel {ch}) "
              f"-> C={W.shape[0]}+1")

    dfs = {s: pd.read_csv(os.path.join(sv_dir, f"split_{s}.csv")) for s in ["train", "val", "test"]}
    if os.environ.get("TRAINVAL", "0") == "1":   # final full-fit: train on train+val (fixed stop)
        dfs["train"] = pd.concat([dfs["train"], dfs["val"]], ignore_index=True)
        print(f"[synth] TRAINVAL=1: training on train+val = {len(dfs['train'])} samples "
              f"(selection must use FIXED_STOP)")
    region = os.path.basename(os.path.normpath(sv_dir)).replace("stationvec_", "")

    clim = None
    if anomaly:    # per-variant+region DOY climatology (TRAIN-only), computed once + cached
        clim, cpath = _synth_doy_climatology(sv_dir, weights_path, Wt, gh, gw, dfs["train"],
                                             region, masks=masks, feature_map=feature_map)
        print(f"[synth] ANOMALY mode ON (PAPERW_ANOM): frames = raw - DOY climatology "
              f"[{os.path.basename(cpath)}]")

    x_mean, x_std, npath = _synth_norm_stats(sv_dir, weights_path, Wt, gh, gw, dfs["train"], region,
                                             masks=masks, feature_map=feature_map, clim=clim,
                                             coverage=coverage, cov_anchor=cov_anchor)
    ns = {"x_mean": x_mean, "x_std": x_std,
          "y_median": z["y_median"].astype(np.float32), "y_iqr": z["y_iqr"].astype(np.float32),
          "y_transform": str(z["y_transform"][0]) if z["y_transform"].shape else str(z["y_transform"]),
          "C": W.shape[0] + (1 if coverage else 0), "T": int(z["T"]), "H": gh, "W": gw}

    if y_override is not None:   # TARGETS ablation: y stats/threshold from the override ranks
        ytr = np.stack([y_override[str(t)] for t in dfs["train"]["tag"]]).astype(np.float32)
        q1, q2, q3 = np.percentile(ytr, [25, 50, 75], axis=0)
        ns["y_median"] = q2.astype(np.float32)
        ns["y_iqr"] = np.maximum((q3 - q1).astype(np.float32), 1e-3)
        print(f"[synth] y_override: {ytr.shape[1]} outputs, y stats recomputed from train override")
    else:
        ytr = np.stack([np.load(os.path.join(sv_dir, os.path.basename(p)))["y"].astype(np.float32)
                        for p in dfs["train"]["path"]])
    hot_thr = float(np.quantile(ytr[:, 0], cfg.extreme_q))

    sv_norm = None
    if hybrid:   # raw station-vector branch normalized with the stationvec dir's stats (612)
        sv_norm = (z["x_mean"].astype(np.float32), z["x_std"].astype(np.float32))
        ns["sv_dim"] = int(z["C"])
        print(f"[synth] HYBRID ON: frames + raw station vectors (sv_dim={ns['sv_dim']}, "
              f"stationvec norm stats)")
    ds = {s: SynthFrameDataset(dfs[s], sv_dir, ns, Wt, cache=cache, masks=masks,
                               return_sv=hybrid, sv_norm=sv_norm, feature_map=feature_map,
                               clim=clim, coverage=coverage, cov_anchor=cov_anchor,
                               y_override=y_override) for s in dfs}
    tr_sampler = None
    osf = float(os.environ.get("EXT_OVERSAMPLE", "0") or "0")
    if osf > 1.0:   # opt-in: oversample extreme-anchor train samples by factor osf
        is_ext = (ytr[:, 0] <= hot_thr) if os.environ.get("COLD", "0") == "1" else (ytr[:, 0] >= hot_thr)
        wts = np.where(is_ext, osf, 1.0).astype(np.float64)
        tr_sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(wts), num_samples=len(wts), replacement=True)
        print(f"[synth] EXT_OVERSAMPLE={osf:g}: {int(is_ext.sum())}/{len(wts)} extreme train samples upweighted")
    def dl(split, shuf): return DataLoader(ds[split], batch_size=cfg.batch_size,
                                           shuffle=(shuf and tr_sampler is None),
                                           sampler=tr_sampler if (shuf and tr_sampler is not None) else None,
                                           num_workers=0, pin_memory=True,
                                           collate_fn=hybrid_collate if hybrid else None)
    loaders = {"train": dl("train", True), "val": dl("val", False), "test": dl("test", False)}
    flags = (f"{' +stacked' if feature_map is not None else ''}"
             f"{' +masks' if masks is not None else ''}{' +cov' if coverage else ''}"
             f"{' +anom' if clim is not None else ''}")
    print(f"[synth] variant={variant}{flags} region={region} | train/val/test="
          f"{len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])} | T={ns['T']} C={ns['C']} "
          f"{gh}x{gw} | hot_thr(p{int(cfg.extreme_q*100)})={hot_thr:.3f} | norm={os.path.basename(npath)}")
    return loaders, ns, hot_thr


def build_data(cfg, history_window=None, y_override=None, cache=False, geo=None, tabular=False, lapse=False):
    """Returns (loaders dict, ns, hot_thr). hot_thr = TRAIN extreme_q quantile of y[:,0].
    y_override: {tag: raw target vector} for the target-set ablation; if given, the y-norm
    stats (median/IQR) and hot_thr are recomputed from the TRAIN override targets.
    lapse=True: add the saved lapse-rate correction to raw frames + use lapse norm stats (A3).
    cache=True preloads normalized samples into RAM (kills per-epoch npz decompression)."""
    ns = load_norm_stats(cfg.dataset_dir, cfg.tag_norm)
    dfs = {s: load_split(cfg.dataset_dir, s, cfg.tag_norm, cfg.keep_dom) for s in ["train", "val", "test"]}

    lapse_corr = None
    if lapse:
        _d = os.path.dirname(__file__)
        lapse_corr = np.load(os.path.join(_d, "lapse_correction.npy")).astype(np.float32)
        ln = np.load(os.path.join(_d, "lapse_norm.npz"))
        ns = dict(ns); ns["x_mean"] = ln["x_mean"]; ns["x_std"] = ln["x_std"]   # lapse-frame stats
        print(f"[lapse] A3 fix ON: correction std={lapse_corr.std():.2f}°C, lapse norm stats loaded")

    if y_override is not None:
        tr_tags = [t.strftime("%Y-%m-%d") for t in dfs["train"]["tag"]]
        Ytr = np.stack([y_override[t] for t in tr_tags if t in y_override]).astype(np.float32)
        ns = dict(ns)
        ns["y_median"] = np.median(Ytr, 0).astype(np.float32)
        ns["y_iqr"] = (np.quantile(Ytr, 0.75, 0) - np.quantile(Ytr, 0.25, 0)).astype(np.float32)
        hot_thr = float(np.quantile(Ytr[:, 0], cfg.extreme_q))
    cdir = os.path.join(cfg.dataset_dir, "_mmcache")
    use_mmap = cache and os.path.exists(os.path.join(cdir, "train_X.npy"))
    if use_mmap:
        ds = {s: MmapDataset(cdir, s, ns, history_window, y_override, geo, tabular, lapse_corr) for s in dfs}
        if y_override is None:
            ytr = np.load(os.path.join(cdir, "train_y.npy"))
            hot_thr = float(np.quantile(ytr[:, 0], cfg.extreme_q))
    else:
        ds = {s: FullSampleDataset(dfs[s], cfg.dataset_dir, ns, cfg.keep_dom, history_window, y_override, cache, geo, tabular, lapse_corr) for s in dfs}
        if y_override is None:
            train_hot = np.array([float(np.load(p, allow_pickle=True)["y"].astype(np.float32)[0])
                                  for p in ds["train"].df["path"].tolist()], dtype=np.float32)
            hot_thr = float(np.quantile(train_hot, cfg.extreme_q))

    nw = cfg.num_workers if use_mmap else (0 if cache else cfg.num_workers)
    def dl(split, shuffle):
        return DataLoader(ds[split], batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
    loaders = {"train": dl("train", True), "val": dl("val", False), "test": dl("test", False)}
    print(f"[data] train/val/test={len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])} "
          f"| hot_thr(p{int(cfg.extreme_q*100)})={hot_thr:.3f} | T={ns['T']} C={ns['C']} {ns['H']}x{ns['W']}"
          + (f" | window={history_window}" if history_window else ""))
    return loaders, ns, hot_thr
