"""
Metrics + selection — ported 1:1 from run_grid.py (CELL 3). The published model
selection is `avg(val_mae_all, val_mae_ext, underrate_ext_all_outputs)`.
Out_dim-agnostic: the anchor (output 0) is always the rank-1 hottest day; for the
published 4-output model the columns are named hot/p3/p7/p15 (back-compat), otherwise
o0..o{k-1}. All extreme/selection logic keys on output 0.
"""
import numpy as np
import pandas as pd
import torch
from .model import denorm_y_torch

NAMED4 = ["hot", "p3", "p7", "p15"]


def _names(k):
    return NAMED4 if k == 4 else [f"o{i}" for i in range(k)]


@torch.no_grad()
def collect_pred_table(model, loader, ns, device, use_amp=True):
    model.eval()
    ym = torch.as_tensor(ns["y_median"], device=device)
    yi = torch.as_tensor(ns["y_iqr"], device=device)
    yt = ns["y_transform"]
    rows, cols = [], None
    for X, y_norm, y_raw, tags in loader:
        X = X.to(device, non_blocking=True).float()
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_norm = model(X)
        pred_raw = denorm_y_torch(pred_norm.float(), ym, yi, yt).detach().cpu().numpy()
        true_raw = y_raw.numpy()
        if cols is None:
            cols = _names(pred_raw.shape[1])
        for i in range(pred_raw.shape[0]):
            rows.append({"tag": tags[i],
                         **{f"true_m1_{c}": float(true_raw[i, j]) for j, c in enumerate(cols)},
                         **{f"pred_m1_{c}": float(pred_raw[i, j]) for j, c in enumerate(cols)}})
    df = pd.DataFrame(rows)
    df.attrs["out_cols"] = cols
    df["tag_dt"] = pd.to_datetime(df["tag"])
    return df.sort_values("tag_dt").reset_index(drop=True)


def _out_cols(df):
    return [c[len("true_m1_"):] for c in df.columns if c.startswith("true_m1_")]


def _arrs(df):
    cols = _out_cols(df)
    t = df[[f"true_m1_{c}" for c in cols]].to_numpy(np.float32)
    p = df[[f"pred_m1_{c}" for c in cols]].to_numpy(np.float32)
    return t, p, cols


def _anchor_true(df):
    cols = _out_cols(df)
    return df[f"true_m1_{cols[0]}"].to_numpy(np.float32)


def mae_metrics(df, hot_thr):
    t, p, cols = _arrs(df)
    mae_out = np.mean(np.abs(p - t), axis=0)
    m_ext = (_anchor_true(df) >= float(hot_thr))
    mae_out_e = np.mean(np.abs(p[m_ext] - t[m_ext]), axis=0) if m_ext.any() else np.full((len(cols),), np.nan, np.float32)
    return dict(mae_out_all=mae_out, mae_mean_all=float(mae_out.mean()),
                mae_out_ext=mae_out_e, mae_mean_ext=float(mae_out_e.mean()) if m_ext.any() else float("nan"),
                n_all=int(len(df)), n_ext=int(m_ext.sum()))


def under_rate_extreme_all_outputs(df, hot_thr):
    t, p, _ = _arrs(df)
    m_ext = (_anchor_true(df) >= float(hot_thr))
    if not m_ext.any():
        return float("nan")
    return float((p[m_ext] < t[m_ext]).mean())


def val_select_score_avg3(df_val, hot_thr):
    m = mae_metrics(df_val, hot_thr)
    mae_all = float(m["mae_mean_all"]); mae_ext = float(m["mae_mean_ext"])
    under_ext = float(under_rate_extreme_all_outputs(df_val, hot_thr))
    if not np.isfinite(mae_ext) or not np.isfinite(under_ext):
        return mae_all
    return float((mae_all + mae_ext + under_ext) / 3.0)


def val_select_score_paper(df_val, q=0.90):
    """Selects on the PAPER metric: Mean MAE = (MAE_all + MAE_ext)/2 on the ANCHOR output,
    extreme = top-q (default top-10%) of the VAL anchor-true values. This directly targets what
    the article ranks on (unlike avg3, whose under-rate term is near-inert). Use with a low/zero
    warmup so the early-epoch extreme optimum is reachable."""
    t, p, _ = _arrs(df_val)
    ta = _anchor_true(df_val)
    err0 = np.abs(p[:, 0] - t[:, 0])            # anchor (hottest/coldest-day) output only
    mae_all = float(err0.mean())
    thr = float(np.quantile(ta, q))
    m_ext = ta >= thr
    if not m_ext.any():
        return mae_all
    mae_ext = float(err0[m_ext].mean())
    return float((mae_all + mae_ext) / 2.0)


def suite(df, hot_thr):
    m = mae_metrics(df, hot_thr)
    return dict(n=m["n_all"], n_ext=m["n_ext"],
                mae_mean=m["mae_mean_all"], mae_out=[float(x) for x in m["mae_out_all"]],
                mae_mean_ext=m["mae_mean_ext"],
                mae_hot_ext=float(m["mae_out_ext"][0]) if np.isfinite(m["mae_out_ext"][0]) else float("nan"),
                mae_hot_all=float(m["mae_out_all"][0]),
                under_rate_ext_all_outs=under_rate_extreme_all_outputs(df, hot_thr))


# ============================================================
# COLD (MIN targets) metrics — ported 1:1 from run_grid_cold_winter.py (CELL 3).
# Opt-in only; reuses collect_pred_table above (output 0 = coldest day; the df
# column is named *_hot for back-compat but is the anchor). Extreme = LOW tail:
# true_cold <= cold_thr (train p10). Direction = "too-warm rate" = P(pred > true).
# ============================================================
def compute_cold_metrics(df, cold_thr):
    """Output 0 only, for selection + reporting. cold_thr = train p10 (low tail)."""
    t = _anchor_true(df)
    cols = _out_cols(df)
    p = df[f"pred_m1_{cols[0]}"].to_numpy(np.float32)
    mae_all = float(np.abs(p - t).mean())
    ext_mask = t <= float(cold_thr)
    n_ext = int(ext_mask.sum())
    if n_ext > 0:
        te, pe = t[ext_mask], p[ext_mask]
        mae_ext = float(np.abs(pe - te).mean())
        too_warm_mae = float(np.maximum(0.0, pe - te).mean())
        too_warm_rate = float((pe > te).mean())
    else:
        mae_ext = too_warm_mae = too_warm_rate = float("nan")
    return dict(n_all=int(len(df)), n_ext=n_ext,
                mae_cold_all=mae_all, mae_cold_ext=mae_ext,
                too_warm_mae_ext=too_warm_mae, too_warm_rate_ext=too_warm_rate)


def val_select_score_cold(df_val, cold_thr):
    """Selection score = avg(val_mae_cold_all, val_mae_cold_ext) on output 0
    (falls back to mae_cold_all when no extremes present)."""
    m = compute_cold_metrics(df_val, cold_thr)
    if m["n_ext"] == 0 or not np.isfinite(m["mae_cold_ext"]):
        return float(m["mae_cold_all"])
    return float((m["mae_cold_all"] + m["mae_cold_ext"]) / 2.0)


def val_select_score_component(df_val, q=0.90, component="all", cold_thr=None, low=False):
    """SELECT ablation: single-component selection on the anchor output.
    component='all' -> val all-sample MAE only; 'ext' -> val extreme-subset MAE only.
    low=True (cold): extreme = anchor-true <= cold_thr (or bottom-(1-q) quantile)."""
    t, p, _ = _arrs(df_val)
    ta = _anchor_true(df_val)
    err0 = np.abs(p[:, 0] - t[:, 0])
    if component == "all":
        return float(err0.mean())
    if low:
        thr = float(cold_thr) if cold_thr is not None else float(np.quantile(ta, 1.0 - q))
        m_ext = ta <= thr
    else:
        m_ext = ta >= float(np.quantile(ta, q))
    return float(err0[m_ext].mean()) if m_ext.any() else float(err0.mean())


def suite_cold(df, cold_thr):
    m = compute_cold_metrics(df, cold_thr)
    t, p, _ = _arrs(df)
    mae_out = np.mean(np.abs(p - t), axis=0)
    return dict(n=m["n_all"], n_ext=m["n_ext"],
                mae_cold_all=m["mae_cold_all"], mae_cold_ext=m["mae_cold_ext"],
                too_warm_mae_ext=m["too_warm_mae_ext"], too_warm_rate_ext=m["too_warm_rate_ext"],
                mae_out_all=[float(x) for x in mae_out], mae_mean_all=float(mae_out.mean()))
