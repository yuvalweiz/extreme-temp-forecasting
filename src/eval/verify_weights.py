"""
Prove "ours" is reproducible 1-to-1 from the saved weights: load each FINAL
ConvNeXt-TFT checkpoint, run inference on the Center test set, and match the
predictions against the paper's saved preds_hot_Center.csv. A match (max|Δ|~0)
means we can regenerate the paper's model outputs exactly without retraining.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                                    # <repo>/src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                                "legacy", "train_reimpl"))
import repo_paths as RP
import models as M

# needs the FULL data release: frame dataset (X) + published checkpoints
DS = RP.canonical_dataset("Center")
# search every grid_ckpts_FINAL* sibling by default (the published runs span them);
# narrow with e.g. CKPT_GLOB='grid_ckpts_FINAL_v3'
CKPT_GLOB = os.environ.get("CKPT_GLOB", "grid_ckpts_FINAL*")
CKDIRS = sorted(glob.glob(os.path.join(os.path.dirname(RP.checkpoints_dir("Center", check=False)),
                                       CKPT_GLOB)))
SAVED = os.path.join(RP.published_preds("Summer"), "Center", "preds_hot_Center.csv")
TAG = "yNONE_v1"
KEEP_DOM = {1, 7, 14, 21, 28}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_samples():
    df = pd.read_csv(os.path.join(DS, f"split_test_{TAG}.csv"))
    df["tag"] = pd.to_datetime(df["tag"]); df = df[df["tag"].dt.day.isin(KEEP_DOM)].sort_values("tag")
    rows = []
    for _, r in df.iterrows():
        p = os.path.join(DS, f"sample_{r['tag'].strftime('%Y-%m-%d')}.npz")
        if os.path.exists(p):
            rows.append((r["tag"].strftime("%Y-%m-%d"), p))
    return rows


def main():
    ns = np.load(os.path.join(DS, f"norm_stats_extremes_full_{TAG}.npz"), allow_pickle=True)
    xm = ns["x_mean"].astype(np.float32).reshape(1, -1, 1, 1)
    xs = ns["x_std"].astype(np.float32).reshape(1, -1, 1, 1)
    ymed = torch.tensor(ns["y_median"].astype(np.float32), device=DEV)
    yiqr = torch.tensor(ns["y_iqr"].astype(np.float32), device=DEV)
    C = int(ns["C"]); T = int(ns["T"])
    samples = test_samples()
    saved = pd.read_csv(SAVED)[["tag", "pred_temporalfusion_hot"]]

    cks = sorted(ck for d0 in CKDIRS
                 for ck in glob.glob(os.path.join(d0, "*temporalfusion*", "best.pt")))
    print(f"[verify] {len(samples)} test samples | {len(cks)} TFT checkpoints "
          f"in {len(CKDIRS)} dirs ({CKPT_GLOB}) | device={DEV}")

    # preload + normalize the test X ONCE (RAM ~9 GB) so each checkpoint is seconds
    print("[verify] preloading test samples ...", flush=True)
    tags, Xs = [], []
    for tag, p in samples:
        X = np.load(p, allow_pickle=True)["X"].astype(np.float32)
        Xs.append((X - xm) / (xs + 1e-8)); tags.append(tag)
    print(f"[verify] loaded {len(Xs)} samples", flush=True)

    best = (1e9, None)
    for ck in cks:
        d = torch.load(ck, map_location=DEV, weights_only=False)
        sd = d.get("state_dict", d)
        model = M.ConvNeXtTiny_WithHead(in_chans=C, T=T, head_type="temporalfusion",
                                        d_model=256, nhead=4, layers=2, dropout=0.1,
                                        pretrained=False).to(DEV).eval()
        miss, unexp = model.load_state_dict(sd, strict=False)
        rows = []
        with torch.no_grad():
            for tag, X in zip(tags, Xs):
                xb = torch.from_numpy(X[None]).to(DEV)
                with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                    pn = model(xb).float()
                pc = (pn * yiqr + ymed)[0, 0].item()
                rows.append({"tag": tag, "pred_mine": pc})
        m = pd.DataFrame(rows).merge(saved, on="tag")
        dmax = float((m.pred_mine - m.pred_temporalfusion_hot).abs().max())
        dmean = float((m.pred_mine - m.pred_temporalfusion_hot).abs().mean())
        name = os.path.join(os.path.basename(os.path.dirname(os.path.dirname(ck))),
                            os.path.basename(os.path.dirname(ck)))
        flag = "  <== MATCH (paper's model)" if dmax < 0.05 else ""
        print(f"  {name[:70]:70s} missing={len(miss)} max|Δ|={dmax:.4f} mean|Δ|={dmean:.4f}{flag}",
              flush=True)
        if dmax < best[0]:
            best = (dmax, name)
    print(f"\n[best match] {best[1]}  max|Δ|={best[0]:.4f}")
    print("=> 'ours' is reproducible 1-to-1 from saved weights" if best[0] < 0.05
          else "=> closest checkpoint above; exact paper config may be in another ckpt dir")


if __name__ == "__main__":
    main()
