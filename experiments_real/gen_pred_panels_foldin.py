"""Regenerate the 6 predicted-vs-true panels for the FOLDED-IN corrected model:
plots the seed-ensembled anchor predictions of the deployed per-region configurations
(from article /results/actual_vs_predicted_corrected/) against the actual observations.
Same style as gen_pred_panels.py (full width, legend on top). Run AFTER foldin_avp.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, os

OUT = "/home/weizyuv/article /dami_submission/figures"
AVP = "/home/weizyuv/article /results/actual_vs_predicted_corrected"
plt.rcParams.update({"font.size": 15, "axes.linewidth": 0.8})

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

specs = [("summer", "Center", "region_center_hot_dom"),
         ("summer", "Northwest", "region_nw_hot_dom"),
         ("summer", "Negev", "region_negev_hot_dom"),
         ("winter", "Center", "region_center_cold_dom"),
         ("winter", "Northwest", "region_nw_cold_dom"),
         ("winter", "Negev", "region_negev_cold_dom")]
for season, reg, stem in specs:
    d = pd.read_csv(os.path.join(AVP, season, reg, "ConvNeXtTiny-TFT__anchor.csv"))
    mae = float(d["abs_error"].mean())
    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    panel(ax, d["date"], d["actual"], d["predicted"], mae)
    fig.savefig(os.path.join(OUT, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{season} {reg:10} {stem}: MAE={mae:.3f} (n={len(d)})")
print("done -> 6 corrected-model panels regenerated")
