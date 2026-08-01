"""
One-time: materialize each split's samples into a single uncompressed fp16 array
(mmap-able). Kills the per-epoch npz decompression — every training run then mmaps
these and reads at RAM speed via the shared OS page cache (no per-run preload, no
I/O contention across concurrent runs). Run once per region.

env: REGION
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
from pipeline.config import REGION_DATASET
from pipeline.data import load_split, _abspath


def main():
    region = os.environ.get("REGION", "Center")
    dsdir = REGION_DATASET[region]
    cdir = os.path.join(dsdir, "_mmcache")
    os.makedirs(cdir, exist_ok=True)
    for split in ["train", "val", "test"]:
        df = load_split(dsdir, split)
        paths, tags, good = [], [], 0
        Xs, Ys = [], []
        for _, row in df.iterrows():
            p = _abspath(dsdir, row["path"]); tag = row["tag"].strftime("%Y-%m-%d")
            try:
                with np.load(p, allow_pickle=True) as z:
                    if "X" not in z.files or "y" not in z.files:
                        continue
                    Xs.append(z["X"].astype(np.float16)); Ys.append(z["y"].astype(np.float32)); tags.append(tag); good += 1
            except Exception:
                continue
        X = np.stack(Xs); Y = np.stack(Ys)
        np.save(os.path.join(cdir, f"{split}_X.npy"), X)
        np.save(os.path.join(cdir, f"{split}_y.npy"), Y)
        json.dump(tags, open(os.path.join(cdir, f"{split}_tags.json"), "w"))
        print(f"[{region}/{split}] {good} samples -> X{X.shape} {X.dtype} ({X.nbytes/1e9:.1f} GB)", flush=True)
    print(f"[DONE] mmap cache -> {cdir}", flush=True)


if __name__ == "__main__":
    main()
