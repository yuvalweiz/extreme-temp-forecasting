"""Residual distributions for the featured manuscript figure (residuals_anchor_nn.pdf):
predicted - actual on the anchor target, pooled over the three regions, one box per
model per season, six models (both interpolation variants + per-station ablations +
classical). Print-sized fonts: >= ~6 pt effective at 0.98\\textwidth (371 pt) display.
Asterisks mark the per-station ablations, matching the caption.
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

plt.rcParams.update({"font.size": 15, "axes.linewidth": 0.9, "font.family": "DejaVu Serif"})

MODELS = [("CNX-TFT (learned)", AVP_C, "ConvNeXtTiny-TFT-NN", "#1f77b4"),
          ("CNX-TFT (kernel)",  AVP_C, "ConvNeXtTiny-TFT",    "#17becf"),
          ("Tab-TFT*",          AVP_C, "Tab-TFT",             "#8c564b"),
          ("Tab-LSTM*",         AVP_C, "Tab-LSTM",            "#9467bd"),
          ("Prophet",           AVP_O, "Prophet",             "#ff7f0e"),
          ("SARIMAX",           AVP_O, "SARIMAX",             "#2ca02c")]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
for ax, (season, title) in zip(axes, [("summer", "Summer (hottest day)"),
                                      ("winter", "Winter (coldest day)")]):
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
    ax.set_xticklabels([m[0] for m in MODELS], rotation=30, ha="right", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.25, axis="y")
    lo = min(np.percentile(x, 5) for x in data)
    for i, x in enumerate(data):
        ax.text(i + 1, lo, f"{np.abs(x).mean():.2f}", ha="center", va="top",
                fontsize=12.5, color="0.25")
    ax.margins(y=0.20)
axes[0].set_ylabel("Residual (\N{DEGREE SIGN}C)", labelpad=6)
fig.tight_layout(w_pad=2.0)
fig.savefig(os.path.join(OUT, "residuals_anchor_nn.pdf"), bbox_inches="tight")
print("done: residuals_anchor_nn (print-sized)")
