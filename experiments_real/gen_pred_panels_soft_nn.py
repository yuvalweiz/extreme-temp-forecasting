"""Soft-target predicted-vs-actual panels (mean of the 3rd/7th/15th ranks), same
style and series as the anchor panels (gen_pred_panels_nn_v2.py): monthly-averaged
samples, one panel per region/season, legend below axes, print-sized fonts.
Outputs region_*_hotsoft_nn.pdf / region_*_coldsoft_nn.pdf.
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

MODELS = [
    ("ConvNeXtTiny-TFT (learned)", AVP_C, "ConvNeXtTiny-TFT-NN", "#1f77b4", "o", "-"),
    ("ConvNeXtTiny-TFT (kernel)",  AVP_C, "ConvNeXtTiny-TFT",    "#17becf", "s", "-"),
    ("Tab-TFT (Ablation)",         AVP_C, "Tab-TFT",             "#8c564b", "x", "--"),
    ("Tab-LSTM (Ablation)",        AVP_C, "Tab-LSTM",            "#9467bd", "d", "--"),
    ("Prophet",                    AVP_O, "Prophet",             "#ff7f0e", "D", "--"),
    ("SARIMAX",                    AVP_O, "SARIMAX",             "#2ca02c", "v", ":"),
]

RANKS = ["p3", "p7", "p15"]

SPECS = [("summer", "Center", "region_center_hotsoft_nn", "Soft Maximum Targets (3rd/7th/15th mean) - Center"),
         ("summer", "Northwest", "region_nw_hotsoft_nn", "Soft Maximum Targets (3rd/7th/15th mean) - North-West"),
         ("summer", "Negev", "region_negev_hotsoft_nn", "Soft Maximum Targets (3rd/7th/15th mean) - Negev"),
         ("winter", "Center", "region_center_coldsoft_nn", "Soft Minimum Targets (3rd/7th/15th mean) - Center"),
         ("winter", "Northwest", "region_nw_coldsoft_nn", "Soft Minimum Targets (3rd/7th/15th mean) - North-West"),
         ("winter", "Negev", "region_negev_coldsoft_nn", "Soft Minimum Targets (3rd/7th/15th mean) - Negev")]

for season, reg, stem, title in SPECS:
    merged = None
    for label, src, fs, *_ in MODELS:
        d = pd.read_csv(os.path.join(src, season, reg, f"{fs}__soft.csv"))
        d["date"] = pd.to_datetime(d["date"])
        d = d.assign(actual=d[[f"actual_{r}" for r in RANKS]].mean(axis=1),
                     pred=d[[f"predicted_{r}" for r in RANKS]].mean(axis=1))
        d = d[["date", "actual", "pred"]].rename(columns={"pred": label})
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
print("done: 6 soft panels")
