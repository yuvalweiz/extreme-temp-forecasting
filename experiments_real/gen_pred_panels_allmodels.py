"""All-models predicted-vs-actual anchor panels, in the OLD article's figure style:
monthly-averaged samples, one panel per region/season, six series
(Observed + ConvNeXtTiny-TFT + ConvNeXtTiny-LSTM + SARIMAX + Prophet + Tab-LSTM).
Deep models read from actual_vs_predicted_corrected (Paper-B predictions);
classical baselines from the published actual_vs_predicted (identical preds in both
protocols). Series are joined on common dates before monthly averaging.
Identity is double-encoded (color + marker/linestyle) for CVD/print safety.
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

plt.rcParams.update({"font.size": 17, "axes.linewidth": 0.9, "font.family": "DejaVu Serif"})

MODELS = [  # (label, source dir, file stem, color, marker, linestyle)
    ("ConvNeXtTiny-TFT",  AVP_C, "ConvNeXtTiny-TFT",  "#1f77b4", "o", "-"),
    ("ConvNeXtTiny-LSTM", AVP_C, "ConvNeXtTiny-LSTM", "#d62728", "^", "--"),
    ("Prophet",           AVP_O, "Prophet",           "#ff7f0e", "D", "--"),
    ("SARIMAX",           AVP_O, "SARIMAX",           "#2ca02c", "v", ":"),
    ("Tab-LSTM (Ablation)", AVP_C, "Tab-LSTM",        "#9467bd", "P", "--"),
]

SPECS = [("summer", "Center", "region_center_hot_dom", "Monthly Maximum Temperature - Center"),
         ("summer", "Northwest", "region_nw_hot_dom", "Monthly Maximum Temperature - North-West"),
         ("summer", "Negev", "region_negev_hot_dom", "Monthly Maximum Temperature - Negev"),
         ("winter", "Center", "region_center_cold_dom", "Monthly Minimum Temperature - Center"),
         ("winter", "Northwest", "region_nw_cold_dom", "Monthly Minimum Temperature - North-West"),
         ("winter", "Negev", "region_negev_cold_dom", "Monthly Minimum Temperature - Negev")]

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

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.plot(x, g["actual"], color="black", lw=2.6, marker="s", ms=7, label="Observed", zorder=10)
    for label, src, fs, color, marker, ls in MODELS:
        ax.plot(x, g[label], color=color, lw=1.8, ls=ls, marker=marker, ms=6,
                label=label, alpha=0.95)
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title(f"{title} [Monthly Samples]")
    ax.grid(alpha=0.25)
    ax.margins(x=0.01)
    for t in ax.get_xticklabels():
        t.set_rotation(35); t.set_ha("right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False, fontsize=15, handlelength=2.2, columnspacing=1.2)
    fig.savefig(os.path.join(OUT, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{season} {reg:10s} -> {stem}  (months={len(g)}, samples={len(merged)})")
print("done: 6 all-model panels")
