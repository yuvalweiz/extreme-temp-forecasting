"""Regenerate the 6 predicted-vs-true panels (hot+cold × 3 regions), full-width,
plotting the PUBLISHED spatial model (ConvNeXt-TFT) against the actual observations.
Fixes: (1) Summer-Center had duplicate LSTM/TFT columns; plotting vs Actual avoids it.
(2) wide blank right margin — legend moved on top, plot fills the panel."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, glob, os

OUT = "/home/weizyuv/article /dami_submission/figures"
SUM = "/home/weizyuv/Models Evaluations/Predictions VS Actuals/Summer"
plt.rcParams.update({"font.size": 15, "axes.linewidth": 0.8})

def winter_csv(reg):
    W = f"/home/weizyuv/Deep Learning Models/Cluster {reg}/Winter Models/Final Results Cluster {reg} Winter"
    return glob.glob(f"{W}/preds_cold_{reg}.csv")[0]

def panel(ax, tags, true, pred, mae, ylabel=True):
    x = pd.to_datetime(tags)
    order = np.argsort(x.values)
    x, true, pred = x[order], np.asarray(true)[order], np.asarray(pred)[order]
    ax.plot(x, true, color="0.6", lw=1.4, label="Actual", zorder=1)
    ax.plot(x, pred, color="#1f77b4", lw=1.6, marker="o", ms=2.5,
            label=f"ConvNeXtTiny–TFT (MAE {mae:.2f}°C)", zorder=2)
    if ylabel: ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.25); ax.margins(x=0.01)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2,
              frameon=False, fontsize=13, handlelength=1.8)

specs = [
    ("hot", "Center", f"{SUM}/Center/preds_hot_Center.csv", "true_m1_hot", "pred_temporalfusion_hot", "region_center_hot_dom"),
    ("hot", "Northwest", f"{SUM}/Northwest/preds_hot_Northwest.csv", "true_m1_hot", "pred_temporalfusion_hot", "region_nw_hot_dom"),
    ("hot", "Negev", f"{SUM}/Negev/preds_hot_Negev.csv", "true_m1_hot", "pred_temporalfusion_hot", "region_negev_hot_dom"),
    ("cold", "Center", winter_csv("Center"), "true_m1_cold", "pred_temporalfusion_cold", "region_center_cold_dom"),
    ("cold", "Northwest", winter_csv("Northwest"), "true_m1_cold", "pred_temporalfusion_cold", "region_nw_cold_dom"),
    ("cold", "Negev", winter_csv("Negev"), "true_m1_cold", "pred_temporalfusion_cold", "region_negev_cold_dom"),
]
for season, reg, csv, tcol, pcol, stem in specs:
    d = pd.read_csv(csv)
    mae = float(np.abs(d[pcol] - d[tcol]).mean())
    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    panel(ax, d["tag"], d[tcol], d[pcol], mae)
    fig.savefig(os.path.join(OUT, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{season} {reg:10} {stem}: MAE={mae:.3f}  (n={len(d)})")
print("done -> 6 panels regenerated (model vs actual, full-width)")
