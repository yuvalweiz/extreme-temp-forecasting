"""Regenerate the six *_nn predicted-vs-actual anchor panels with the legend BELOW
the axes (the previous version's in-axes legend covered the winter minima).
Series identical to the published figures: Observed + both interpolation variants of
ConvNeXtTiny-TFT + Tab-TFT/Tab-LSTM ablations + Prophet + SARIMAX.
Identity double-encoded (color + marker/linestyle) for CVD/print safety.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

OUT = "/home/weizyuv/article /dami_submission/figures"
AVP_C = "/home/weizyuv/article /results/actual_vs_predicted_corrected"
AVP_O = "/home/weizyuv/article /results/actual_vs_predicted"

plt.rcParams.update({"font.size": 17, "axes.linewidth": 0.9, "font.family": "DejaVu Serif"})

MODELS = [  # (label, source dir, file stem, color, marker, linestyle)
    ("ConvNeXtTiny-TFT (learned)", AVP_C, "ConvNeXtTiny-TFT-NN", "#1f77b4", "o", "-"),
    ("ConvNeXtTiny-TFT (kernel)",  AVP_C, "ConvNeXtTiny-TFT",    "#17becf", "s", "-"),
    ("Tab-TFT (Ablation)",         AVP_C, "Tab-TFT",             "#8c564b", "x", "--"),
    ("Tab-LSTM (Ablation)",        AVP_C, "Tab-LSTM",            "#9467bd", "d", "--"),
    ("Prophet",                    AVP_O, "Prophet",             "#ff7f0e", "D", "--"),
    ("SARIMAX",                    AVP_O, "SARIMAX",             "#2ca02c", "v", ":"),
]

SPECS = [("summer", "Center", "region_center_hot_nn", "Monthly Maximum Temperature - Center"),
         ("summer", "Northwest", "region_nw_hot_nn", "Monthly Maximum Temperature - North-West"),
         ("summer", "Negev", "region_negev_hot_nn", "Monthly Maximum Temperature - Negev"),
         ("winter", "Center", "region_center_cold_nn", "Monthly Minimum Temperature - Center"),
         ("winter", "Northwest", "region_nw_cold_nn", "Monthly Minimum Temperature - North-West"),
         ("winter", "Negev", "region_negev_cold_nn", "Monthly Minimum Temperature - Negev")]

for season, reg, stem, title in SPECS:
    merged = None
    for label, src, fs, *_ in MODELS:
        d = pd.read_csv(os.path.join(src, season, reg, f"{fs}__anchor.csv"))
        d = d[["date", "actual", "predicted"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.rename(columns={"predicted": label})
        if merged is None:
            merged = d
        else:
            merged = merged.merge(d.drop(columns="actual"), on="date", how="inner")
    merged["month"] = merged["date"].dt.to_period("M")
    g = merged.groupby("month").mean(numeric_only=True).reset_index()
    x = g["month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    ax.plot(x, g["actual"], color="black", lw=2.6, marker="s", ms=7, label="Observed", zorder=10)
    for label, src, fs, color, marker, ls in MODELS:
        ax.plot(x, g[label], color=color, lw=1.8, ls=ls, marker=marker, ms=6,
                label=label, alpha=0.95)
    ax.set_ylabel("Temperature (\N{DEGREE SIGN}C)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.margins(x=0.01)
    for tl in ax.get_xticklabels():
        tl.set_rotation(30); tl.set_ha("right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4,
              frameon=False, fontsize=15, handlelength=1.8, columnspacing=0.9)
    fig.savefig(os.path.join(OUT, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"{season} {reg:10s} -> {stem}  (months={len(g)}, samples={len(merged)})")
print("done: 6 nn panels, legend below axes")
