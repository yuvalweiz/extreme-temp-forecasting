"""
Reproducible training directly from an on-disk frames directory (no per-sample
npz). Same model/loss as the grid (models.py). Targets + splits come from the
existing region dataset (kernel-independent), so pointing --frames at EXP / IDW /
EXP_V2 / a future paper-accurate dir gives a clean interpolation ablation.

Env params: REGION, FRAMES (EXP|IDW|EXP_V2), FRAMES_DIR/PREFIX (override),
HEAD, K, ALPHA, BETA, EPOCHS, PATIENCE, SEED, CKPT_ROOT, SAVE_CKPT.

Validation built in: for FRAMES=EXP the assembled window equals the stored
sample (max|Δ|=0), so a reproduction run must match the paper's numbers.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import models as M
import frame_window_dataset as FW

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
OUT_WEIGHTS = (1.0, 0.5, 0.25, 0.125)
EXTREME_Q = 0.90
KEEP_DOM = {1, 7, 14, 21, 28}

REGION = os.environ.get("REGION", "Center")
FRAMES = os.environ.get("FRAMES", "EXP")
FRAMES_DIR = os.environ.get("FRAMES_DIR", FW.FRAME_DIRS.get(FRAMES, ""))
PREFIX = os.environ.get("PREFIX", FW.PREFIX.get(FRAMES, "exponential"))
HEAD = os.environ.get("HEAD", "temporalfusion")
K = float(os.environ.get("K", "2")); ALPHA = float(os.environ.get("ALPHA", "2")); BETA = float(os.environ.get("BETA", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "120")); PATIENCE = int(os.environ.get("PATIENCE", "10"))
SEED = int(os.environ.get("SEED", "333"))
HISTORY = int(os.environ.get("HISTORY", str(FW.HISTORY)))  # A2 window ablation knob
POOL = os.environ.get("POOL", "avg")  # spatial pooling: avg (published) | attn | avgmax
ANOMALY = os.environ.get("ANOMALY", "off")  # off | augment (concat day-of-year climatology anomaly channels)
SPATIAL = os.environ.get("SPATIAL", "full")  # full | mean (flatten spatial structure -> tests if spatial info is used)
GRAD = os.environ.get("GRAD", "off")  # off | on (append per-channel spatial gradient-magnitude channels -> structure survives GAP)
TARGETS = os.environ.get("TARGETS", "1,3,7,15")  # 1-based order-stat ranks; #outputs=len; rank-1 = anchor metric
RANKS = [int(x) for x in TARGETS.split(",")]
RES = int(os.environ.get("RES", "1"))  # effective-resolution ablation: downsample frames by RES then upsample back
CKPT_ROOT = os.environ.get("CKPT_ROOT", f"./ckpt_frames_{REGION}_{FRAMES}")
DS_DIR = f"/home/weizyuv/Deep Learning Models/Cluster {REGION}/dataset_FULL_h180_next30_DOM_1_7_14_21_28"
TAG = "yNONE_v1"


def load_items(split):
    df = pd.read_csv(os.path.join(DS_DIR, f"split_{split}_{TAG}.csv"))
    df["tag"] = pd.to_datetime(df["tag"]); df = df[df["tag"].dt.day.isin(KEEP_DOM)].sort_values("tag")
    items = []
    for _, r in df.iterrows():
        p = os.path.join(DS_DIR, f"sample_{r['tag'].strftime('%Y-%m-%d')}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        items.append({"tag": r["tag"].strftime("%Y-%m-%d"),
                      "pred_point": pd.Timestamp(str(z["pred_point"])), "y": z["y"].astype(np.float32)})
    return items


def compute_x_stats(items, cache, feats, n_in=None):
    """Train-only per-channel mean/std over the unique frames in train windows."""
    npz = os.path.join(CKPT_ROOT, f"xstats_{FRAMES}_{ANOMALY}.npz")
    if os.path.exists(npz):
        d = np.load(npz); return d["mean"], d["std"]
    dates = set()
    for it in items:
        for d in pd.date_range(it["pred_point"] - pd.Timedelta(days=HISTORY - 1), it["pred_point"]):
            dates.add(d.strftime("%Y-%m-%d"))
    C = int(n_in or len(feats)); s = np.zeros(C); s2 = np.zeros(C); n = 0
    for ds in dates:
        if ds in cache:
            a = cache[ds].reshape(C, -1); s += a.sum(1); s2 += (a ** 2).sum(1); n += a.shape[1]
    mean = (s / n).astype(np.float32); std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8)).astype(np.float32)
    os.makedirs(CKPT_ROOT, exist_ok=True); np.savez(npz, mean=mean, std=std)
    return mean, std


@torch.no_grad()
def predict(model, loader, y_med, y_iqr):
    model.eval(); rows = []
    for X, yn, yr, tags in loader:
        X = X.to(DEVICE).float()
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pc = M.denorm_y_torch(model(X).float(), y_med, y_iqr).cpu().numpy()
        tr = yr.numpy()
        for i in range(pc.shape[0]):
            r = {"tag": tags[i], "true_m1_hot": float(tr[i, 0]), "pred_m1_hot": float(pc[i, 0])}
            if pc.shape[1] > 1:
                r["true_soft"] = float(tr[i, 1:].mean()); r["pred_soft"] = float(pc[i, 1:].mean())
            rows.append(r)
    return pd.DataFrame(rows)


def metrics(df, thr):
    # anchor (rank-1, output 0) metric — comparable across any target construction
    a = (df.pred_m1_hot - df.true_m1_hot).abs()
    ext = df[df.true_m1_hot >= thr]
    me = float((ext.pred_m1_hot - ext.true_m1_hot).abs().mean()) if len(ext) else float("nan")
    return float(a.mean()), me, len(ext)


def main():
    assert FRAMES_DIR and os.path.isdir(FRAMES_DIR), f"bad FRAMES_DIR {FRAMES_DIR}"
    os.makedirs(CKPT_ROOT, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    feats = open(os.path.join(DS_DIR, "features_order.txt")).read().split()
    ns = np.load(os.path.join(DS_DIR, f"norm_stats_extremes_full_{TAG}.npz"), allow_pickle=True)
    y_med = ns["y_median"].astype(np.float32); y_iqr = ns["y_iqr"].astype(np.float32)

    items = {s: load_items(s) for s in ["train", "val", "test"]}
    out_w = tuple(0.5 ** i for i in range(len(RANKS)))  # geometric decay for #targets
    if RANKS != [1, 3, 7, 15]:  # output-construction ablation: recompute y from the cluster order statistics
        sys.path.insert(0, os.path.join(HERE, "..", "baselines"))
        import series as S
        ser = S.build_cluster_daily(REGION)["max_dry_temp"].dropna().sort_index()
        for s in items:
            keep = []
            for it in items[s]:
                fut = ser.reindex(pd.date_range(pd.Timestamp(it["tag"]), it["pred_point"] + pd.Timedelta(days=30)))
                fut = fut.dropna().sort_values(ascending=False).to_numpy()
                if len(fut) >= max(RANKS):
                    it["y"] = np.array([fut[r - 1] for r in RANKS], np.float32); keep.append(it)
            items[s] = keep
        ys = np.array([it["y"] for it in items["train"]], np.float32)
        y_med = np.median(ys, 0).astype(np.float32)
        y_iqr = (np.quantile(ys, .75, 0) - np.quantile(ys, .25, 0)).astype(np.float32)
        print(f"[targets] ranks={RANKS} -> out_dim={len(RANKS)} (y recomputed from cluster order stats)")
    train_hot = np.sort(np.array([it["y"][0] for it in items["train"]], np.float32))
    thr = float(np.quantile(train_hot, EXTREME_Q))
    print(f"[{REGION}/{FRAMES}] train/val/test = {len(items['train'])}/{len(items['val'])}/{len(items['test'])} | p90={thr:.2f}")

    lo = min(it["pred_point"] for s in items for it in items[s]) - pd.Timedelta(days=HISTORY)
    hi = max(it["pred_point"] for s in items for it in items[s])
    print(f"[preload] frames {lo.date()}..{hi.date()} from {os.path.basename(FRAMES_DIR)}")
    t0 = time.time(); cache = FW.preload_frames(FRAMES_DIR, PREFIX, feats, lo, hi)
    print(f"[preload] {len(cache)} daily frames in {time.time()-t0:.0f}s")

    if RES > 1:  # effective-resolution ablation: downsample then upsample back (same size, coarser detail)
        import torch.nn.functional as Fnn
        for ds in list(cache):
            t = torch.from_numpy(cache[ds]).unsqueeze(0); H, W = t.shape[2:]
            cache[ds] = Fnn.interpolate(Fnn.avg_pool2d(t, RES, ceil_mode=True), size=(H, W),
                                        mode="nearest").squeeze(0).numpy().astype(np.float32)
        print(f"[res] effective resolution /{RES} (44x137 -> ~{44//RES}x{137//RES} -> upsampled back)")

    n_in = len(feats)
    if ANOMALY == "augment":
        # day-of-year climatology from TRAIN period only (no leakage), then
        # append per-pixel anomaly channels: X -> concat(raw, raw - clim[doy]).
        train_end = max(it["pred_point"] for it in items["train"])
        from collections import defaultdict
        acc, cnt = defaultdict(lambda: 0.0), defaultdict(int)
        for ds, arr in cache.items():
            d = pd.Timestamp(ds)
            if d <= train_end:
                doy = d.dayofyear
                acc[doy] = acc[doy] + arr; cnt[doy] += 1
        clim = {k: (acc[k] / cnt[k]).astype(np.float32) for k in acc}
        gmean = np.mean([v for v in clim.values()], axis=0).astype(np.float32)
        for ds in list(cache):
            doy = pd.Timestamp(ds).dayofyear
            c = clim.get(doy, gmean)
            cache[ds] = np.concatenate([cache[ds], cache[ds] - c], axis=0).astype(np.float32)
        n_in = 2 * len(feats)
        print(f"[anomaly] augmented to {n_in} channels (raw + DOY-climatology anomaly)")

    if GRAD == "on":
        # append per-channel spatial gradient magnitude: injects structure that
        # survives GAP (mean|grad| != mean|value|). Computed per frame, no leakage.
        for ds in list(cache):
            f = cache[ds]
            gy, gx = np.gradient(f, axis=1), np.gradient(f, axis=2)
            mag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
            cache[ds] = np.concatenate([f, mag], axis=0).astype(np.float32)
        n_in = cache[next(iter(cache))].shape[0]
        print(f"[grad] augmented to {n_in} channels (raw + spatial gradient magnitude)")

    if FRAMES == "EXP" and ANOMALY == "off" and GRAD == "off":  # reuse paper's exact stats for true reproduction
        x_mean, x_std = ns["x_mean"].astype(np.float32), ns["x_std"].astype(np.float32)
    else:
        x_mean, x_std = compute_x_stats(items["train"], cache, feats, n_in)

    def mk(split, shuffle):
        ds = FW.FrameWindowDataset(FRAMES_DIR, PREFIX, feats, items[split], x_mean, x_std,
                                   y_med, y_iqr, history=HISTORY, cache=cache, spatial_mode=SPATIAL)
        return DataLoader(ds, batch_size=4, shuffle=shuffle, num_workers=0)
    tr, va, te = mk("train", True), mk("val", False), mk("test", False)

    model = M.ConvNeXtTiny_WithHead(in_chans=n_in, T=HISTORY, head_type=HEAD,
                                    d_model=256, nhead=4, layers=2, dropout=0.1,
                                    backbone_pool=POOL, out_dim=len(RANKS)).to(DEVICE)
    base = M.PercentileWeightedMAE(y_med, y_iqr, "none", out_w, train_hot, k=K, alpha=ALPHA).to(DEVICE)
    loss_fn = M.PercentileWeightedHinge(base, beta_hot=BETA).to(DEVICE)
    y_med_t = torch.as_tensor(y_med, device=DEVICE); y_iqr_t = torch.as_tensor(y_iqr, device=DEVICE)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": 1e-5, "weight_decay": 1e-3},
                             {"params": hd, "lr": 1e-4, "weight_decay": 1e-4}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best, best_ep, best_state, bad, ema = 1e9, -1, None, 0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for X, yn, yr, tags in tr:
            X = X.to(DEVICE).float(); yr = yr.to(DEVICE).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                loss = loss_fn(model(X).float(), yr)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        dfv = predict(model, va, y_med_t, y_iqr_t)
        ma, me, ne = metrics(dfv, thr)
        raw = ma if (ne == 0 or not np.isfinite(me)) else 0.5 * (ma + me)
        ema = raw if ema is None else 0.35 * raw + 0.65 * ema
        if ema < best - 1e-6:
            best, best_ep, best_state, bad = ema, ep, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        print(f"ep{ep:03d} val_all={ma:.3f} val_ext={me:.3f} ema={ema:.4f} best={best:.4f}@{best_ep} bad={bad}/{PATIENCE}")
        if bad >= PATIENCE:
            break
    if best_state:
        model.load_state_dict(best_state)
    dft = predict(model, te, y_med_t, y_iqr_t)
    ta, tee, tne = metrics(dft, thr)
    dfv = predict(model, va, y_med_t, y_iqr_t); va_a, va_e, _ = metrics(dfv, thr)
    dft.to_csv(os.path.join(CKPT_ROOT, "preds_test.csv"), index=False)
    dfv.to_csv(os.path.join(CKPT_ROOT, "preds_val.csv"), index=False)
    meta = {"region": REGION, "frames": FRAMES, "head": HEAD, "k": K, "alpha": ALPHA, "beta": BETA,
            "best_ep": best_ep, "val_mae_all": va_a, "val_mae_ext": va_e,
            "test_mae_all": ta, "test_mae_ext": tee, "test_n_ext": tne}
    json.dump(meta, open(os.path.join(CKPT_ROOT, "meta.json"), "w"), indent=2)
    print(f"[DONE] {FRAMES} TEST mae_all={ta:.3f} mae_ext={tee:.3f} (val {va_a:.3f}/{va_e:.3f})")


def _getitem_cached(self, i):
    it = self.items[i]
    X = FW.load_window(self.frames_dir, self.prefix, self.feats, it["pred_point"], self.history,
                       cache=getattr(self, "cache", None))
    X = np.nan_to_num(X, nan=0.0)
    Xn = (X - self.x_mean) / (self.x_std + self.eps)
    yr = np.asarray(it["y"], np.float32)
    yn = (yr - self.y_median) / np.maximum(self.y_iqr, 1e-6)
    return (torch.from_numpy(Xn.astype(np.float32)), torch.from_numpy(yn), torch.from_numpy(yr), it["tag"])


if __name__ == "__main__":
    main()
