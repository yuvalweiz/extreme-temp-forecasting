"""Framing-A staged figure kit — output to figures_prep/, NOT the paper figures dir.
1. pred-vs-actual anchor panels (3 regions x 2 seasons) with the NN pipeline featured
2. LOSO interpolation-fidelity bars (IDW/kernel/KED/NN) with downstream annotation
3. validation->test transfer scatter per family (the deployment-selection insight)
4. kernel vs NN frame example (one test-period day, max dry temp, station overlay)
Style matches the article panels (DejaVu Serif, same rc); identity double-encoded
(color + marker/hatch) so no series is color-alone; sequential maps use one ramp.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import glob
import os

OUT = "/home/weizyuv/expreal/figures_prep"
AVP = "/home/weizyuv/article /results/actual_vs_predicted_corrected"
AVP_O = "/home/weizyuv/article /results/actual_vs_predicted"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 15, "axes.linewidth": 0.9, "font.family": "DejaVu Serif"})

# ---------- 1. panels --------------------------------------------------------
MODELS = [  # label, dir, stem, color (Okabe-Ito), marker, ls
    ("Ours (learned interp.)", AVP, "ConvNeXtTiny-TFT-NN", "#0072B2", "o", "-"),
    ("Ours (kernel)",          AVP, "ConvNeXtTiny-TFT",    "#56B4E9", "s", "-"),
    ("Tab-TFT (ablation)",     AVP, "Tab-TFT",             "#E69F00", "^", "--"),
    ("Tab-LSTM (ablation)",    AVP, "Tab-LSTM",            "#CC79A7", "P", "--"),
    ("SARIMAX",                AVP_O, "SARIMAX",           "#009E73", "v", ":"),
]
SPECS = [("summer", r, f"nnpanel_{r.lower()}_hot") for r in ["Center", "Northwest", "Negev"]] + \
        [("winter", r, f"nnpanel_{r.lower()}_cold") for r in ["Center", "Northwest", "Negev"]]
for season, reg, stem in SPECS:
    merged = None
    for label, src, fs, *_ in MODELS:
        p = os.path.join(src, season, reg, f"{fs}__anchor.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)[["date", "actual", "predicted"]].rename(columns={"predicted": label})
        merged = d if merged is None else merged.merge(d.drop(columns="actual"), on="date")
    if merged is None:
        continue
    merged["date"] = pd.to_datetime(merged.date)
    m = merged.set_index("date").resample("MS").mean().dropna()
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(m.index, m.actual, color="black", lw=2.4, label="Observed", zorder=5)
    for label, src, fs, c, mk, ls in MODELS:
        if label in m:
            ax.plot(m.index, m[label], color=c, marker=mk, ls=ls, lw=1.6, ms=5.5, label=label)
    ax.set_ylabel("Temperature (°C)")
    ttl = "Maximum" if season == "summer" else "Minimum"
    ax.set_title(f"Monthly {ttl} Temperature — {reg.replace('Northwest','North-West')}")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(ncol=3, fontsize=11, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{stem}.{ext}", dpi=200)
    plt.close(fig)
print("panels done")

# ---------- 2. LOSO bars -----------------------------------------------------
methods = ["Inverse-distance", "Elevation-aware kernel", "Kriging (KED)", "Learned (NN)"]
vals = [1.71, 1.183, 0.909, 1.044]
colors = ["#BBBBBB", "#BBBBBB", "#BBBBBB", "#0072B2"]
fig, ax = plt.subplots(figsize=(8.6, 4.2))
bars = ax.bar(methods, vals, color=colors, width=0.62, edgecolor="white")
bars[2].set_hatch("//")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=13)
ax.set_ylabel("LOSO test MAE (°C)")
ax.set_title("Interpolation fidelity at held-out stations (max dry temp)")
ax.text(2, 0.45, "map-optimal,\nforecast-harmful", ha="center", fontsize=11, style="italic")
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/loso_fidelity.{ext}", dpi=200)
plt.close(fig)
print("loso done")

# ---------- 3. val->test transfer scatter -----------------------------------
T = "/home/weizyuv/expreal/Center_tune"
FAMS = [("ours", "Ours (kernel)", "#0072B2", "o"), ("nnf", "Ours (learned)", "#56B4E9", "s"),
        ("stf", "Tab-TFT", "#E69F00", "^"), ("stl", "Tab-LSTM", "#CC79A7", "P")]
def cand_points(fam):
    pts = {}
    for d in glob.glob(f"{T}/{fam}s2std_*__*") + glob.glob(f"{T}/{fam}p2std_*__*"):
        key = d.split("/")[-1].split("__")[0]
        try:
            m = json.load(open(d + "/meta.json"))
        except Exception:
            continue
        t = m.get("test_topk") or m.get("test")
        pts.setdefault(key, []).append((m["val"]["mae_mean"], t["mae_mean"]))
    return [(np.mean([p[0] for p in v]), np.mean([p[1] for p in v])) for v in pts.values() if len(v) >= 3]
fig, ax = plt.subplots(figsize=(7.6, 5.6))
for fam, label, c, mk in FAMS:
    pts = cand_points(fam)
    if not pts:
        continue
    v, t = zip(*pts)
    ax.scatter(v, t, color=c, marker=mk, s=95, label=label, edgecolor="white", lw=0.8, zorder=3)
    best = min(pts, key=lambda p: p[0])
    ax.scatter([best[0]], [best[1]], facecolor="none", edgecolor=c, s=260, lw=2.2, zorder=4)
lims = [1.35, 2.1]
ax.plot(lims, lims, color="#888888", lw=0.8, ls=":")
ax.set_xlabel("Validation MAE (°C, selection currency)")
ax.set_ylabel("Test MAE (°C)")
ax.set_title("Candidate selection transfer (Center, current outputs)\ncircled = each family's validation pick")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=11)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/val_test_transfer.{ext}", dpi=200)
plt.close(fig)
print("transfer done")

# ---------- 4. kernel vs NN frame example -----------------------------------
try:
    G = np.load("/home/weizyuv/article /repo/data/grid_metadata.npz", allow_pickle=True)
    W = np.load("/home/weizyuv/interp_weights/paper_opt.npz", allow_pickle=True)
    D = np.load("/home/weizyuv/expreal/nninterp/daily_sv.npz", allow_pickle=True)
    bank_dates = open("/home/weizyuv/expreal/framebank_nn/dates.txt").read().split()
    bank = np.load("/home/weizyuv/expreal/framebank_nn/bank.npy", mmap_mode="r")
    day = "2023-07-15"
    di_b = bank_dates.index(day)
    di_d = list(D["dates"].astype(str)).index(day)
    fmax = 1  # channel order in bank == features_order; max_dry_temp index
    feats = [l.strip() for l in open("/home/weizyuv/Deep Learning Models/Cluster Center/"
             "dataset_FULL_h180_next30_DOM_1_7_14_21_28/features_order.txt") if l.strip()]
    fmax = feats.index("max_dry_temp")
    nn_frame = np.asarray(bank[di_b, fmax], dtype=float)
    sv = np.nan_to_num(D["M"][di_d, :, fmax])
    kern = (W["W"][fmax] @ sv).reshape(int(W["grid_h"]), int(W["grid_w"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    vmin, vmax = np.nanpercentile(nn_frame, [2, 98])
    for ax, fr, ttl in [(axes[0], kern, "Elevation-aware kernel"), (axes[1], nn_frame, "Learned interpolation (NN)")]:
        if fr is None:
            ax.axis("off"); ax.set_title("kernel frame unavailable"); continue
        im = ax.imshow(fr, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.scatter((G["station_lon"] - G["lon_grid"].min()) / (G["lon_grid"].max() - G["lon_grid"].min()) * fr.shape[1],
                   (G["station_lat"] - G["lat_grid"].min()) / (G["lat_grid"].max() - G["lat_grid"].min()) * fr.shape[0],
                   s=12, c="white", edgecolor="black", lw=0.4)
        ax.set_title(f"{ttl} — {day}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, label="Max dry-bulb temp (°C)")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/frame_kernel_vs_nn.{ext}", dpi=200)
    plt.close(fig)
    print("frame example done")
except Exception as e:
    print("frame example skipped:", e)
