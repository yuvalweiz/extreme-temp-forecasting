"""
Build an IDW-frame dataset that mirrors the EXP dataset EXACTLY (same tags, same
splits, same y targets, same 180-day window = date_range(end=tag-1day, periods=180)),
but with weather values from the IDW (2D, distance-only) frames instead of EXP
(elevation-aware). Lets us answer "is the elevation-aware interpolation good?" by
comparing EXP vs IDW (and IDW+geo) under the identical pipeline.

Outputs a space-free dataset dir with _mmcache/{split}_{X,y}.npy + tags + norm_stats
(x-stats recomputed on IDW train; y-stats copied from EXP since targets are identical).
"""
import os, sys, json, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP
from pipeline.data import load_split, _abspath, load_norm_stats

# needs the FULL data release: canonical EXP dataset + the on-disk IDW frame bank
EXP_DS = os.environ.get("EXP_DS") or RP.canonical_dataset("Center")
IDW_FRAMES = os.environ.get("IDW_FRAMES") or RP.frames_dir("IDW")
OUT = os.environ.get("OUT", os.path.join(RP.DATA_ROOT, "idw_Center"))   # space-free
FEATS = open(f"{EXP_DS}/features_order.txt").read().split()
HIST = 180


def idw_frame(date_str, cache):
    if date_str in cache:
        return cache[date_str]
    chans = []
    for f in FEATS:
        p = os.path.join(IDW_FRAMES, f"idw_{f}_{date_str}.npy")
        chans.append(np.load(p).astype(np.float16) if os.path.exists(p) else None)
    stack = None if any(c is None for c in chans) else np.stack(chans, 0)  # (9,44,137)
    cache[date_str] = stack
    return stack


def main():
    os.makedirs(f"{OUT}/_mmcache", exist_ok=True)
    # mirror metadata so the pipeline can read it
    for fn in os.listdir(EXP_DS):
        if fn.startswith("split_") or fn.startswith("norm_stats") or fn == "features_order.txt":
            shutil.copy(os.path.join(EXP_DS, fn), os.path.join(OUT, fn))
    ns = load_norm_stats(EXP_DS)
    cache = {}
    xsum = np.zeros(len(FEATS), np.float64); xsq = np.zeros(len(FEATS), np.float64); xn = 0
    for split in ["train", "val", "test"]:
        df = load_split(EXP_DS, split)
        Xs, Ys, tags = [], [], []
        for _, row in df.iterrows():
            tag = row["tag"].strftime("%Y-%m-%d")
            exp_p = _abspath(EXP_DS, row["path"])
            if not os.path.exists(exp_p):
                continue
            y = np.load(exp_p)["y"].astype(np.float32)        # identical targets
            anchor = row["tag"] - pd.Timedelta(days=1)
            dates = pd.date_range(end=anchor, periods=HIST, freq="D")
            frames = [idw_frame(d.strftime("%Y-%m-%d"), cache) for d in dates]
            if any(f is None for f in frames):
                # fill missing day with nearest available (forward/back) to keep shape
                last = None
                for i, f in enumerate(frames):
                    if f is None: frames[i] = last
                    else: last = f
                for i in range(len(frames)-1, -1, -1):
                    if frames[i] is None: frames[i] = next(f for f in frames if f is not None)
            X = np.stack(frames, 0).astype(np.float16)        # (180,9,44,137)
            Xs.append(X); Ys.append(y); tags.append(tag)
            if split == "train":
                xf = X.astype(np.float64)
                xsum += xf.sum((0, 2, 3)); xsq += (xf**2).sum((0, 2, 3)); xn += xf[:, 0].size
        X = np.stack(Xs); Y = np.stack(Ys)
        np.save(f"{OUT}/_mmcache/{split}_X.npy", X)
        np.save(f"{OUT}/_mmcache/{split}_y.npy", Y)
        json.dump(tags, open(f"{OUT}/_mmcache/{split}_tags.json", "w"))
        print(f"[IDW/{split}] {len(tags)} samples X{X.shape} {X.nbytes/1e9:.1f}GB", flush=True)
    # IDW x-stats (train); y-stats copied from EXP
    xmean = (xsum/xn).astype(np.float32); xstd = np.sqrt(np.maximum(xsq/xn - (xsum/xn)**2, 1e-12)).astype(np.float32)
    np.savez(f"{OUT}/norm_stats_extremes_full_yNONE_v1.npz",
             x_mean=xmean, x_std=xstd, y_median=ns["y_median"], y_iqr=ns["y_iqr"],
             y_transform=np.array(["none"]), T=np.int64(HIST), C=np.int64(len(FEATS)),
             H=np.int64(44), W=np.int64(137))
    print(f"[DONE] IDW dataset -> {OUT}  | IDW x_mean(maxdry)={xmean[7]:.2f} vs EXP {ns['x_mean'][7]:.2f}", flush=True)


if __name__ == "__main__":
    main()
