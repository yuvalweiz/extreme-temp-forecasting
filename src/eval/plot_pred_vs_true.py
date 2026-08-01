"""
p1 (hottest-day) actual vs predicted over the test period: ours (ConvNeXt-TFT)
vs TimesFM 2.5 zero-shot. Shows our model tracking inter-annual extremes while
the non-spatial foundation model misses the peaks.

Run: python plot_pred_vs_true.py  -> repo/results/figures/p1_actual_vs_pred_<...>.{png,pdf}
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP

PREDS = RP.published_preds("Summer")
TFM = RP.results_dir()          # timesfm_<Region>.csv live in repo/results
OUT = os.path.join(RP.results_dir(), "figures")
REGIONS = ["Center", "Negev", "Northwest"]


def load(region):
    ours = pd.read_csv(os.path.join(PREDS, region, f"preds_hot_{region}.csv"))
    ours["tag"] = pd.to_datetime(ours["tag"])
    ours = ours[["tag", "true_m1_hot", "pred_temporalfusion_hot"]]
    tfm = pd.read_csv(os.path.join(TFM, f"timesfm_{region}.csv"))
    tfm["tag"] = pd.to_datetime(tfm["tag"])
    tfm = tfm[["tag", "pred_m1_hot"]].rename(columns={"pred_m1_hot": "pred_timesfm_hot"})
    return ours.merge(tfm, on="tag").sort_values("tag").reset_index(drop=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(11, 9), sharex=False)
    for ax, region in zip(axes, REGIONS):
        d = load(region)
        x = d["tag"]
        mae_o = (d.pred_temporalfusion_hot - d.true_m1_hot).abs().mean()
        mae_t = (d.pred_timesfm_hot - d.true_m1_hot).abs().mean()
        ax.plot(x, d.true_m1_hot, "-o", color="black", ms=3, lw=1.4, label="Actual p1")
        ax.plot(x, d.pred_temporalfusion_hot, "-s", color="#1f77b4", ms=3, lw=1.3,
                label=f"ConvNeXt-TFT (ours)  MAE={mae_o:.2f}")
        ax.plot(x, d.pred_timesfm_hot, "-^", color="#d62728", ms=3, lw=1.1, alpha=0.85,
                label=f"TimesFM 2.5 0-shot  MAE={mae_t:.2f}")
        ax.set_title(f"{region} — month-ahead hottest day (p1), test set", fontsize=11)
        ax.set_ylabel("Temperature (°C)")
        ax.legend(fontsize=8, loc="lower right", ncol=1, framealpha=0.9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Prediction date (test period)")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT, f"p1_actual_vs_pred_3regions.{ext}"), dpi=160, bbox_inches="tight")
    print("[OK] wrote", os.path.join(OUT, "p1_actual_vs_pred_3regions.png/pdf"))
    # also Center-only, larger
    d = load("Center")
    fig2, ax = plt.subplots(figsize=(11, 4.2))
    mae_o = (d.pred_temporalfusion_hot - d.true_m1_hot).abs().mean()
    mae_t = (d.pred_timesfm_hot - d.true_m1_hot).abs().mean()
    ax.plot(d.tag, d.true_m1_hot, "-o", color="black", ms=4, lw=1.6, label="Actual p1")
    ax.plot(d.tag, d.pred_temporalfusion_hot, "-s", color="#1f77b4", ms=4, lw=1.4,
            label=f"ConvNeXt-TFT (ours)  MAE={mae_o:.2f}°C")
    ax.plot(d.tag, d.pred_timesfm_hot, "-^", color="#d62728", ms=4, lw=1.2, alpha=0.85,
            label=f"TimesFM 2.5 zero-shot  MAE={mae_t:.2f}°C")
    ax.set_title("Center (Tel Aviv) — month-ahead hottest day p1: actual vs predicted")
    ax.set_ylabel("Temperature (°C)"); ax.set_xlabel("Prediction date (test period)")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.25)
    fig2.tight_layout()
    for ext in ["png", "pdf"]:
        fig2.savefig(os.path.join(OUT, f"p1_actual_vs_pred_Center.{ext}"), dpi=160, bbox_inches="tight")
    print("[OK] wrote", os.path.join(OUT, "p1_actual_vs_pred_Center.png/pdf"))


if __name__ == "__main__":
    main()
