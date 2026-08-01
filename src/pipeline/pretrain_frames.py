"""
In-domain self-supervised pretraining (masked denoising autoencoder) on the author's
own daily interpolated frames — the standout upside vs out-of-domain ImageNet weights
(literature: in-domain MAE > ImageNet transfer for <10k labels).

Encoder = the SAME ConvNeXt backbone used in the forecasting pipeline, so its weights
load straight into model.backbone.m for fine-tuning. Masked input -> reconstruct the
full 9-channel frame. NO LEAKAGE: only frames dated <= the region's last TRAIN pred_point.

env: REGION, BACKBONE(convnext_tiny|convnextv2_tiny|convnextv2_nano), EPOCHS, MASK_RATIO, BATCH, OUT
     TASK A opt-in: PAPERW=<interp_weights .npz> + DATASET_DIR=<stationvec dir> ->
     frames SYNTHESIZED from station-vector daily rows (train-cutoff-safe); LIMIT=<n> caps days.
"""
import os, sys, glob, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP
from pipeline.config import REGION_DATASET
from pipeline.data import load_norm_stats

FDIR = os.environ.get("FRAMES_DIR") or RP.frames_dir("EXP", check=False)
FEAT_ORDER = ["max_wet_temp","tmp_air_wet","min_wet_temp","tmp_air_dry","tmp_dew_pnt",
              "max_heat_stress","min_heat_stress","max_dry_temp","min_dry_temp"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_frame(date):
    chans = []
    for feat in FEAT_ORDER:
        p = os.path.join(FDIR, f"exponential_{feat}_{date}.npy")
        if not os.path.exists(p):
            return None
        chans.append(np.load(p).astype(np.float32))
    return np.stack(chans, 0)                                  # (9,44,137)


class FrameDS(torch.utils.data.Dataset):
    def __init__(self, dates, xm, xs):
        self.dates, self.xm, self.xs = dates, xm.reshape(-1,1,1), xs.reshape(-1,1,1)
    def __len__(self): return len(self.dates)
    def __getitem__(self, i):
        f = load_frame(self.dates[i])
        f = np.nan_to_num((f - self.xm) / (self.xs + 1e-8), nan=0.0).astype(np.float32)
        return torch.from_numpy(f)


class FrameMAE(nn.Module):
    def __init__(self, in_chans=9, backbone="convnext_tiny"):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=True, in_chans=in_chans,
                                     num_classes=0, global_pool="")
        Cf = int(self.enc.num_features)
        self.dec = nn.Sequential(
            nn.Conv2d(Cf, 256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, in_chans, 1))
    def forward(self, x, out_hw):
        fm = self.enc.forward_features(x)                      # (B,Cf,h',w')
        fm = F.interpolate(fm, size=out_hw, mode="bilinear", align_corners=False)
        return self.dec(fm)


# ============================================================
# TASK A (opt-in): masked-AE pretraining on SYNTHESIZED frames.
# env PAPERW=<interp_weights .npz> + DATASET_DIR=<stationvec dir> -> pretraining frames
# are synthesized on the fly (W @ station-vector daily rows, same _synth_frames math as
# train.py's PAPERW mode) instead of read from FDIR. Normalization = the SAME per-variant
# synth norm stats the fine-tune uses (_synth_norm_stats cache next to the weights file).
# LEAK SAFETY: rows come ONLY from TRAIN samples (split_train.csv); each sample's 180
# history days end at tag-1day (dim_order.json windowing), so every day predates the
# train cutoff = max TRAIN tag (same convention as the FDIR path below).
# Checkpoint format IDENTICAL: {"backbone","region","enc_state_dict"} -> train.py
# PRETRAINED loading works unchanged. BACKBONE env: convnext_tiny / convnextv2_tiny /
# convnextv2_nano (any timm convnext*). Extra env: LIMIT=<n unique days> (CPU smokes).
# PAPERW unset -> this file behaves byte-identically to before.
# ============================================================
class SynthPretrainDS(torch.utils.data.Dataset):
    """One item = one unique calendar day's synthesized (C,H,W) frame, normalized."""
    def __init__(self, day_rows, Wt, gh, gw, xm, xs):
        self.rows = day_rows                                   # (N, S*C) float32
        self.Wt, self.gh, self.gw = Wt, gh, gw                 # (C,S,H*W)
        self.xm, self.xs = xm.reshape(-1, 1, 1), xs.reshape(-1, 1, 1)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        from pipeline.data import _synth_frames
        f = _synth_frames(self.rows[i:i + 1], self.Wt, self.gh, self.gw)[0]   # (C,gh,gw)
        f = np.nan_to_num((f - self.xm) / (self.xs + 1e-8), nan=0.0).astype(np.float32)
        return torch.from_numpy(f)


def _synth_pretrain_data(weights_path, sv_dir, limit=None):
    """Collect the unique-day station-vector rows of ALL TRAIN samples (<= train cutoff)
    and return (dataset, in_chans, cutoff). Rows are deduped by calendar day."""
    from pipeline.data import _synth_norm_stats
    wz = np.load(weights_path, allow_pickle=True)
    W = wz["W"].astype(np.float32)                              # (C, H*W, S) row-stochastic
    gh, gw = int(wz["grid_h"]), int(wz["grid_w"])
    Wt = np.ascontiguousarray(W.transpose(0, 2, 1))             # (C, S, H*W)
    z = np.load(os.path.join(sv_dir, "norm_stats.npz"), allow_pickle=True)
    assert [str(s) for s in wz["station_names"]] == [str(s) for s in z["station_names"]], \
        "weights/stationvec station order mismatch"
    assert [str(f) for f in wz["features"]] == [str(f) for f in z["features"]], \
        "weights/stationvec feature order mismatch"
    df_tr = pd.read_csv(os.path.join(sv_dir, "split_train.csv"))
    cutoff = pd.to_datetime(df_tr["tag"]).max()                 # no-leakage cutoff (train only)
    region = os.path.basename(os.path.normpath(sv_dir)).replace("stationvec_", "")
    x_mean, x_std, _ = _synth_norm_stats(sv_dir, weights_path, Wt, gh, gw, df_tr, region)

    rows = {}
    for _, r in df_tr.sort_values("tag").iterrows():
        tag = pd.Timestamp(r["tag"])
        sv = np.load(os.path.join(sv_dir, os.path.basename(r["path"])))["X"].astype(np.float32)
        days = pd.date_range(end=tag - pd.Timedelta(days=1), periods=sv.shape[0], freq="D")
        for i, dd in enumerate(days):
            if dd > cutoff:                                     # explicit guard (never true for train)
                continue
            k = dd.strftime("%Y-%m-%d")
            if k not in rows:
                rows[k] = sv[i]
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    day_rows = np.stack([rows[k] for k in sorted(rows)]).astype(np.float32)
    ds = SynthPretrainDS(day_rows, Wt, gh, gw, x_mean, x_std)
    return ds, W.shape[0], cutoff


def _main_synth():
    """TASK A alternative main: same masked-AE recipe/loop as main(), frames synthesized."""
    region = os.environ.get("REGION", "Center")
    backbone = os.environ.get("BACKBONE", "convnext_tiny")
    epochs = int(os.environ.get("EPOCHS", "60"))
    mask_ratio = float(os.environ.get("MASK_RATIO", "0.5"))
    batch = int(os.environ.get("BATCH", "64"))
    out = os.environ.get("OUT", os.path.join(RP.experiments_root(),
                                             f"pretrain_synth_{region}_{backbone}.pt"))
    weights_path = os.environ["PAPERW"]
    sv_dir = os.environ["DATASET_DIR"]
    limit = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None

    ds, in_chans, cutoff = _synth_pretrain_data(weights_path, sv_dir, limit=limit)
    print(f"[pretrain-synth/{region}/{backbone}] weights={os.path.basename(weights_path)} "
          f"days<=train_cutoff({cutoff.date()})={len(ds)}"
          + (f" (LIMIT={limit})" if limit else "") + f" | mask={mask_ratio} epochs={epochs}")
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                                     drop_last=(len(ds) >= 2 * batch))
    model = FrameMAE(in_chans=in_chans, backbone=backbone).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    use_cuda = (DEV == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    P = 4  # mask block size (pixels)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; n = 0
        for x in dl:
            x = x.to(DEV, non_blocking=True)                   # (B,C,gh,gw)
            B, C, H, W = x.shape
            mh, mw = (H + P - 1)//P, (W + P - 1)//P
            m = (torch.rand(B, 1, mh, mw, device=DEV) < mask_ratio).float()
            m = F.interpolate(m, size=(H, W), mode="nearest")  # (B,1,H,W) 1=masked
            xin = x * (1 - m)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                rec = model(xin, (H, W))
                loss = (((rec - x)**2) * m).sum() / (m.sum()*C + 1e-6)   # MSE on masked pixels
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss)*B; n += B
        sch.step()
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  ep{ep:03d} masked-MSE={tot/max(n,1):.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    # save ENCODER weights keyed to match ConvNeXtTinyBackbone.m — format identical to main()
    torch.save({"backbone": backbone, "region": region, "mode": "synth",
                "weights": weights_path, "enc_state_dict": model.enc.state_dict()}, out)
    print(f"[DONE] saved encoder -> {out}")


def main():
    if os.environ.get("PAPERW"):        # TASK A opt-in: synthesized-frame pretraining
        return _main_synth()
    region = os.environ.get("REGION", "Center")
    backbone = os.environ.get("BACKBONE", "convnext_tiny")
    epochs = int(os.environ.get("EPOCHS", "60"))
    mask_ratio = float(os.environ.get("MASK_RATIO", "0.5"))
    batch = int(os.environ.get("BATCH", "64"))
    out = os.environ.get("OUT", os.path.join(RP.experiments_root(),
                                             f"pretrain_{region}_{backbone}.pt"))
    dsdir = REGION_DATASET[region]
    ns = load_norm_stats(dsdir)

    # no-leakage cutoff = last TRAIN pred_point
    tr = pd.read_csv(os.path.join(dsdir, "split_train_yNONE_v1.csv"))
    cutoff = pd.to_datetime(tr["tag"]).max()
    all_dates = sorted(os.path.basename(x).split("_")[-1][:-4]
                       for x in glob.glob(f"{FDIR}/exponential_max_dry_temp_*.npy"))
    dates = [d for d in all_dates if pd.Timestamp(d) <= cutoff]
    print(f"[pretrain/{region}/{backbone}] frames<=train_cutoff({cutoff.date()})={len(dates)} "
          f"(of {len(all_dates)} total) | mask={mask_ratio} epochs={epochs}")

    dl = torch.utils.data.DataLoader(FrameDS(dates, ns["x_mean"], ns["x_std"]),
                                     batch_size=batch, shuffle=True, num_workers=6, drop_last=True)
    model = FrameMAE(in_chans=ns["C"], backbone=backbone).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    P = 4  # mask block size (pixels)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; n = 0
        for x in dl:
            x = x.to(DEV, non_blocking=True)               # (B,9,44,137)
            B, C, H, W = x.shape
            mh, mw = (H + P - 1)//P, (W + P - 1)//P
            m = (torch.rand(B,1,mh,mw,device=DEV) < mask_ratio).float()
            m = F.interpolate(m, size=(H,W), mode="nearest")    # (B,1,H,W) 1=masked
            xin = x * (1 - m)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                rec = model(xin, (H, W))
                loss = (((rec - x)**2) * m).sum() / (m.sum()*C + 1e-6)   # MSE on masked pixels
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss)*B; n += B
        sch.step()
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  ep{ep:03d} masked-MSE={tot/n:.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    # save ENCODER weights keyed to match ConvNeXtTinyBackbone.m
    torch.save({"backbone": backbone, "region": region,
                "enc_state_dict": model.enc.state_dict()}, out)
    print(f"[DONE] saved encoder -> {out}")


if __name__ == "__main__":
    main()
