"""Residual distributions (predicted - actual, anchor target, pooled 3 regions):
one box per model per season. Visualizes the UPE finding: baselines' summer error
mass sits below zero (under-prediction of the monthly maximum). Colors/ordering
match the prediction panels; zero line = perfect calibration.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

OUT = "/home/weizyuv/article /dami_submission/figures"
AVP_C = "/home/weizyuv/article /results/actual_vs_predicted_corrected"
AVP_O = "/home/weizyuv/article /results/actual_vs_predicted"

plt.rcParams.update({"font.size": 13, "axes.linewidth": 0.9, "font.family": "DejaVu Serif"})

MODELS = [("ConvNeXtTiny-TFT", AVP_C, "ConvNeXtTiny-TFT", "#1f77b4"),
          ("ConvNeXtTiny-LSTM", AVP_C, "ConvNeXtTiny-LSTM", "#d62728"),
          ("Tab-LSTM", AVP_C, "Tab-LSTM", "#9467bd"),
          ("Prophet", AVP_O, "Prophet", "#ff7f0e"),
          ("SARIMAX", AVP_O, "SARIMAX", "#2ca02c")]

fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), sharey=False)
for ax, (season, title) in zip(axes, [("summer", "Maximum-temperature anchor (summer)"),
                                      ("winter", "Minimum-temperature anchor (winter)")]):
    data, colors = [], []
    for label, src, fs, color in MODELS:
        r = []
        for reg in ["Center", "Northwest", "Negev"]:
            d = pd.read_csv(os.path.join(src, season, reg, f"{fs}__anchor.csv"))
            r.append(d["predicted"] - d["actual"])
        data.append(pd.concat(r).to_numpy())
        colors.append(color)
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55,
                    whis=(5, 95), medianprops=dict(color="black", lw=1.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(c)
    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_xticklabels([m[0] for m in MODELS], rotation=25, ha="right", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25, axis="y")
    lo = min(np.percentile(x, 5) for x in data)
    for i, x in enumerate(data):
        ax.text(i + 1, lo, f"MAE\n{np.abs(x).mean():.2f}", ha="center", va="top",
                fontsize=9.5, color="0.25", linespacing=0.9)
    ax.margins(y=0.18)
axes[0].set_ylabel("Predicted $-$ actual (deg C)", labelpad=6)
fig.tight_layout(w_pad=2.5)
fig.savefig(os.path.join(OUT, "residuals_anchor.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "residuals_anchor.png"), dpi=150, bbox_inches="tight")
print("done: residuals_anchor")
