"""
Clean, reproducible evaluation + significance library for the
extreme-temperature forecasting paper.

Ported from the original Models Evaluations/Evaluations.ipynb (cells 3,4,6,9,10)
with two corrections documented in docs/STRATEGY.md:

  (1) Cross-region pooling for significance concatenates each region's
      per-sample loss differential exactly ONCE (the original notebook
      tripled the pooled "ALL" rows: n=2148=3x716, n_ext=285=3x95).
  (2) A single, explicit extreme mask definition: a test sample is
      "extreme" iff its true hottest-day value y[0] >= the TRAIN p90 of y[0]
      for that region (p10 for the cold/winter setting). This is the same
      mask used for hot and soft targets, computed on TRAIN only (no leakage).

Metrics:
  hot target  : err on output 0 (hottest day)                 -> MAE/RMSE
  soft target : mean over outputs {p3,p7,p15} of |pred-true|   -> MAE/RMSE
  all-case    : over all test samples
  extreme-case: over the extreme subset (mask above)

The module consumes the per-region prediction CSVs (same schema used by the
paper and emitted by repo/src/train/train_grid_hot.py), so the identical code
evaluates baselines and any newly trained model.
"""
import os
import numpy as np
import pandas as pd

KEEP_DOM = {1, 7, 14, 21, 28}
EXTREME_Q = 0.90  # hot p90 (cold uses 0.10)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def mae(t, p):
    t = np.asarray(t, float); p = np.asarray(p, float)
    m = np.isfinite(t) & np.isfinite(p)
    return (float(np.mean(np.abs(t[m] - p[m]))) if m.any() else np.nan, int(m.sum()))


def rmse(t, p):
    t = np.asarray(t, float); p = np.asarray(p, float)
    m = np.isfinite(t) & np.isfinite(p)
    return (float(np.sqrt(np.mean((t[m] - p[m]) ** 2))) if m.any() else np.nan, int(m.sum()))


# --------------------------------------------------------------------------
# extreme threshold from TRAIN only (no leakage)
# --------------------------------------------------------------------------
def _load_split(csv_path):
    df = pd.read_csv(csv_path)
    df["tag"] = pd.to_datetime(df["tag"], errors="coerce")
    df = df[df["tag"].dt.day.isin(KEEP_DOM)].copy()
    return df.sort_values("tag").reset_index(drop=True)


def _resolve(path, dataset_dir):
    p = str(path).strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p
    pr = p[2:] if p.startswith("./") else p
    root = os.path.dirname(os.path.normpath(dataset_dir))
    for c in (os.path.join(root, pr), os.path.join(dataset_dir, pr),
              os.path.join(root, os.path.basename(pr)),
              os.path.join(dataset_dir, os.path.basename(pr))):
        if os.path.exists(c):
            return c
    return os.path.join(root, pr)


def split_csv(dataset_dir, split, tag="yNONE_v1"):
    """Resolve a split CSV: prefer the tagged file (canonical frame datasets,
    split_<s>_yNONE_v1.csv); fall back to the untagged one (station-vector
    datasets, split_<s>.csv — identical tags/dates, verified)."""
    cands = ([f"split_{split}_{tag}.csv"] if tag else []) + [f"split_{split}.csv"]
    for name in cands:
        p = os.path.join(dataset_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no split_{split}*.csv found in {dataset_dir}")


def train_extreme_threshold(dataset_dir, tag="yNONE_v1", q=EXTREME_Q):
    """TRAIN-only quantile of y[0] (hottest day). q=0.90 hot, 0.10 cold.

    y source, in order: per-sample npz (canonical frame dataset or station-vector
    dataset) -> bundled y_values_*.csv (data/dataset_meta/<Region>/, extracted
    verbatim from the dataset so a bare git clone can regenerate the tables)."""
    df = _load_split(split_csv(dataset_dir, "train", tag))
    vals = []
    for p in df["path"].tolist():
        rp = _resolve(p, dataset_dir)
        if os.path.exists(rp):
            with np.load(rp, allow_pickle=True) as z:
                vals.append(float(z["y"].astype(np.float32)[0]))
    if not vals:
        yv = load_y_values(dataset_dir, tag)
        if yv is not None:
            m = yv.merge(df[["tag"]], on="tag")
            vals = m.loc[m["split"] == "train", "y_p1"].astype(float).tolist()
    if not vals:
        raise RuntimeError(f"no train y-values resolved under {dataset_dir}")
    return float(np.quantile(np.asarray(vals, np.float32), q))


def load_y_values(dataset_dir, tag="yNONE_v1"):
    """Bundled per-sample y table (split,tag,y_p1,y_p3,y_p7,y_p15) or None."""
    for name in ([f"y_values_{tag}.csv"] if tag else []) + ["y_values.csv"]:
        p = os.path.join(dataset_dir, name)
        if os.path.exists(p):
            yv = pd.read_csv(p)
            yv["tag"] = pd.to_datetime(yv["tag"])
            return yv
    return None


# --------------------------------------------------------------------------
# prediction loaders  (region-level, schema-tolerant)
# --------------------------------------------------------------------------
def _read(fp):
    df = pd.read_csv(fp)
    if "tag" not in df.columns and "tag_dt" in df.columns:
        df = df.rename(columns={"tag_dt": "tag"})
    df["tag"] = pd.to_datetime(df["tag"], errors="coerce")
    return df.dropna(subset=["tag"]).sort_values("tag").reset_index(drop=True)


def load_region_hot(preds_dir, region):
    """Returns dict model-> DataFrame[tag,true,pred] for the HOT (output-0) target."""
    out = {}
    base = os.path.join(preds_dir, region)
    f = os.path.join(base, f"preds_hot_{region}.csv")
    if os.path.exists(f):
        d = _read(f)
        for col, name in [("pred_temporalfusion_hot", "temporalfusion"),
                          ("pred_lstm_hot", "lstm")]:
            if col in d.columns:
                out[name] = d[["tag", "true_m1_hot", col]].rename(
                    columns={"true_m1_hot": "true", col: "pred"})
    for fn, name in [(f"preds_prophet_{region}.csv", "prophet"),
                     (f"preds_sarimax_{region}_test_wide.csv", "sarimax"),
                     (f"preds_tab_lstm_hot_israel_wide_{region}.csv", "tab_lstm")]:
        fp = os.path.join(base, fn)
        if os.path.exists(fp):
            d = _read(fp)
            if "pred_m1_hot" in d.columns:
                out[name] = d[["tag", "true_m1_hot", "pred_m1_hot"]].rename(
                    columns={"true_m1_hot": "true", "pred_m1_hot": "pred"})
    return out


SOFT = ["p3", "p7", "p15"]


def load_region_soft(preds_dir, region):
    """Returns dict model-> DataFrame[tag,true_hot,true_p3.., pred_p3..] for SOFT."""
    out = {}
    base = os.path.join(preds_dir, region)
    f = os.path.join(base, f"preds_soft_{region}.csv")
    if os.path.exists(f):
        d = _read(f)
        keep = ["tag", "true_m1_hot"] + [f"true_m1_{s}" for s in SOFT]
        for name in ["temporalfusion", "lstm"]:
            cols = [f"pred_{name}_soft_{s}" for s in SOFT]
            if all(c in d.columns for c in cols):
                ren = {f"pred_{name}_soft_{s}": f"pred_{s}" for s in SOFT}
                out[name] = d[keep + cols].rename(columns=ren)
    for fn, name in [(f"preds_prophet_{region}.csv", "prophet"),
                     (f"preds_sarimax_{region}_test_wide.csv", "sarimax"),
                     (f"preds_tab_lstm_soft_israel_wide_{region}.csv", "tab_lstm")]:
        fp = os.path.join(base, fn)
        if os.path.exists(fp):
            d = _read(fp)
            cols = [f"pred_m1_{s}" for s in SOFT]
            if all(c in d.columns for c in cols):
                ren = {f"pred_m1_{s}": f"pred_{s}" for s in SOFT}
                keep = ["tag", "true_m1_hot"] + [f"true_m1_{s}" for s in SOFT]
                out[name] = d[keep + cols].rename(columns=ren)
    return out


# --------------------------------------------------------------------------
# per-sample loss arrays  (for metrics AND significance)
# --------------------------------------------------------------------------
def hot_losses(df, thr):
    """Return DataFrame[tag, abs_err, is_ext] for the hot target."""
    e = (df["pred"] - df["true"]).abs().to_numpy(float)
    ext = (df["true"].to_numpy(float) >= float(thr))
    return pd.DataFrame({"tag": df["tag"].values, "abs_err": e, "is_ext": ext})


def soft_losses(df, thr):
    """Soft target = the AVERAGE of the 3rd/7th/15th extreme days.

    The error is |mean(preds) - mean(trues)| (abs-of-mean), matching the
    summer evaluation in the original notebook (cell 3/6). NOTE: the winter
    cell used mean-of-abs instead -- that season inconsistency is documented
    in docs/STRATEGY.md and is unified here to abs-of-mean.
    Extreme subset is keyed on the true hottest day (same mask as hot).
    """
    true_avg = df[[f"true_m1_{s}" for s in SOFT]].to_numpy(float).mean(axis=1)
    pred_avg = df[[f"pred_{s}" for s in SOFT]].to_numpy(float).mean(axis=1)
    e = np.abs(pred_avg - true_avg)
    ext = (df["true_m1_hot"].to_numpy(float) >= float(thr))
    return pd.DataFrame({"tag": df["tag"].values, "abs_err": e, "is_ext": ext})


def summarize(loss_df):
    a = loss_df["abs_err"].to_numpy(float)
    e = loss_df.loc[loss_df["is_ext"], "abs_err"].to_numpy(float)
    mae_all = float(np.mean(a)) if len(a) else np.nan
    rmse_all = float(np.sqrt(np.mean(a ** 2))) if len(a) else np.nan
    mae_ext = float(np.mean(e)) if len(e) else np.nan
    rmse_ext = float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan
    return dict(n_all=len(a), n_ext=len(e),
                mae_all=mae_all, rmse_all=rmse_all,
                mae_ext=mae_ext, rmse_ext=rmse_ext,
                avg_mae=np.nanmean([mae_all, mae_ext]),
                avg_rmse=np.nanmean([rmse_all, rmse_ext]))


# --------------------------------------------------------------------------
# moving-block bootstrap + Holm  (ported verbatim from the notebook)
# --------------------------------------------------------------------------
def moving_block_bootstrap(diff, block_len=4, n_boot=10000, seed=0):
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < max(8, 2 * block_len):
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    blocks = [diff[s:s + block_len] for s in range(0, n - block_len + 1)]
    n_blocks = int(np.ceil(n / block_len))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(blocks), size=n_blocks)
        boot[b] = np.mean(np.concatenate([blocks[i] for i in pick])[:n])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
    return float(np.mean(diff)), float(lo), float(hi), p


def holm_bonferroni(pvals):
    p = pd.Series(pvals, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return p
    adj = pd.Series(index=valid.index, dtype=float)
    for i, (k, pv) in enumerate(valid.items()):
        adj[k] = min(1.0, (m - i) * pv)
    adj = adj.cummax()
    out = p.copy()
    out.loc[adj.index] = adj.values
    return out
