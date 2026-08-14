"""Rebuild every cell of the manuscript's two main result tables (anchor MAE/RMSE/UPE,
soft MAE3/RMSE3) from the released row-level prediction files in this repository, and
verify them against the printed values. Run from anywhere:

    python src/eval/rebuild_manuscript_tables.py

Writes results/actual_vs_predicted_corrected/manuscript_tables.csv and regenerates
metrics_summary_corrected.csv in the same directory. Exits non-zero on any mismatch.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AC = REPO / "results" / "actual_vs_predicted_corrected"
AO = REPO / "results" / "actual_vs_predicted"
RES = REPO / "results"
REGIONS = ["Center", "Northwest", "Negev"]

# The classical-baseline row-level files are generated (not stored): materialize if absent.
if not (AO / "summer" / "Center" / "Prophet__anchor.csv").exists():
    import os
    import subprocess
    print("results/actual_vs_predicted missing - generating via export_actual_vs_predicted.py ...")
    subprocess.run([sys.executable, str(REPO / "src" / "eval" / "export_actual_vs_predicted.py")],
                   check=True, env=dict(os.environ, PYTHONPATH=str(REPO / "src")))

def deep(root, season, reg, model):
    a = pd.read_csv(root / season / reg / f"{model}__anchor.csv")
    s = pd.read_csv(root / season / reg / f"{model}__soft.csv")
    e = (a.predicted - a.actual).to_numpy()
    er = np.concatenate([(s[f"predicted_p{k}"] - s[f"actual_p{k}"]).to_numpy() for k in (3, 7, 15)])
    return e, er

def foundation(name, season, reg):
    df = pd.read_csv(RES / f"{name}_{reg}{'' if season == 'summer' else '_min'}.csv")
    if "split" in df.columns:
        df = df[df.split == "test"]
    e = (df.pred_m1_hot - df.true_m1_hot).to_numpy()
    er = np.concatenate([(df[f"pred_m1_p{k}"] - df[f"true_m1_p{k}"]).to_numpy() for k in (3, 7, 15)])
    return e, er

MODELS = {
    "ConvNeXtTiny-TFT-NN": lambda s, r: deep(AC, s, r, "ConvNeXtTiny-TFT-NN"),
    "ConvNeXtTiny-TFT": lambda s, r: deep(AC, s, r, "ConvNeXtTiny-TFT"),
    "ConvNeXtTiny-LSTM": lambda s, r: deep(AC, s, r, "ConvNeXtTiny-LSTM"),
    "Tab-TFT": lambda s, r: deep(AC, s, r, "Tab-TFT"),
    "Tab-LSTM": lambda s, r: deep(AC, s, r, "Tab-LSTM"),
    "Prophet": lambda s, r: deep(AO, s, r, "Prophet"),
    "SARIMAX": lambda s, r: deep(AO, s, r, "SARIMAX"),
    "TimesFM-2.5": lambda s, r: foundation("timesfm", s, r),
    "Moirai-1.1": lambda s, r: foundation("moirai", s, r),
    "Climatology": lambda s, r: deep(AC, s, r, "Climatology"),
    "SeasonalNaive": lambda s, r: deep(AC, s, r, "SeasonalNaive"),
}

# Printed manuscript cells: (season, model) -> per-region [anchor MAE, RMSE, UPE, soft MAE3, RMSE3]
PRINTED = {
("summer","ConvNeXtTiny-TFT-NN"): {"Center":[1.852,2.405,None,1.276,1.669],"Northwest":[1.859,2.392,None,1.293,1.747],"Negev":[1.766,2.299,None,1.407,1.823]},
("summer","ConvNeXtTiny-TFT"):    {"Center":[2.113,2.713,None,1.267,1.683],"Northwest":[1.809,2.284,None,1.279,1.759],"Negev":[2.007,2.511,None,1.393,1.792]},
("summer","ConvNeXtTiny-LSTM"):   {"Center":[2.315,2.917,None,1.277,1.698],"Northwest":[1.985,2.513,None,1.250,1.748],"Negev":[2.535,3.038,None,1.531,2.041]},
("summer","Tab-TFT"):             {"Center":[1.914,2.488,None,1.304,1.670],"Northwest":[1.894,2.401,None,1.290,1.676],"Negev":[1.896,2.422,None,1.567,1.999]},
("summer","Tab-LSTM"):            {"Center":[2.058,2.639,None,1.337,1.740],"Northwest":[1.871,2.545,None,1.339,1.841],"Negev":[1.914,2.465,None,1.530,2.010]},
("summer","Prophet"):             {"Center":[4.739,5.619,None,1.924,2.603],"Northwest":[4.498,5.451,None,1.962,2.763],"Negev":[4.832,5.420,None,2.342,2.960]},
("summer","SARIMAX"):             {"Center":[4.744,5.631,None,1.913,2.606],"Northwest":[4.515,5.478,None,1.968,2.787],"Negev":[4.805,5.406,None,2.315,2.950]},
("summer","TimesFM-2.5"):         {"Center":[5.071,6.388,None,1.899,2.895],"Northwest":[4.810,6.232,None,2.016,3.161],"Negev":[4.854,5.784,None,2.169,2.988]},
("summer","Moirai-1.1"):          {"Center":[4.924,6.509,None,2.066,3.214],"Northwest":[4.594,6.236,None,2.164,3.375],"Negev":[4.930,6.392,None,2.517,3.828]},
("winter","ConvNeXtTiny-TFT-NN"): {"Center":[1.298,1.652,None,0.972,1.283],"Northwest":[1.393,1.814,None,0.987,1.335],"Negev":[1.372,1.745,None,1.003,1.360]},
("winter","ConvNeXtTiny-TFT"):    {"Center":[1.275,1.645,None,0.963,1.237],"Northwest":[1.440,1.837,None,1.006,1.331],"Negev":[1.331,1.730,None,0.986,1.297]},
("winter","ConvNeXtTiny-LSTM"):   {"Center":[1.278,1.623,None,0.953,1.206],"Northwest":[1.427,1.829,None,1.018,1.319],"Negev":[1.400,1.761,None,1.060,1.376]},
("winter","Tab-TFT"):             {"Center":[1.315,1.720,None,0.975,1.227],"Northwest":[1.483,1.938,None,1.062,1.393],"Negev":[1.511,1.900,None,1.105,1.444]},
("winter","Tab-LSTM"):            {"Center":[1.462,1.787,None,0.906,1.191],"Northwest":[1.741,2.157,None,1.118,1.427],"Negev":[1.558,1.920,None,1.103,1.393]},
("winter","Prophet"):             {"Center":[1.687,2.258,None,0.970,1.334],"Northwest":[1.690,2.313,None,0.986,1.392],"Negev":[2.021,2.606,None,1.132,1.573]},
("winter","SARIMAX"):             {"Center":[1.767,2.333,None,0.979,1.359],"Northwest":[1.744,2.374,None,1.004,1.420],"Negev":[2.123,2.708,None,1.161,1.608]},
("winter","TimesFM-2.5"):         {"Center":[2.067,2.570,None,1.037,1.403],"Northwest":[2.073,2.658,None,1.057,1.466],"Negev":[2.298,2.814,None,1.196,1.571]},
("winter","Moirai-1.1"):          {"Center":[2.535,3.042,None,1.518,2.002],"Northwest":[2.458,3.034,None,1.448,1.942],"Negev":[2.682,3.207,None,1.657,2.130]},
("summer","Climatology"):        {"Center":[4.716,5.591,None,1.955,2.623],"Northwest":[4.483,5.412,None,1.985,2.783],"Negev":[4.760,5.344,None,2.311,2.925]},
("summer","SeasonalNaive"):      {"Center":[2.746,3.404,None,1.581,2.041],"Northwest":[2.536,3.194,None,1.631,2.133],"Negev":[2.632,3.197,None,1.834,2.305]},
("winter","Climatology"):        {"Center":[1.739,2.305,None,0.983,1.362],"Northwest":[1.728,2.350,None,1.001,1.418],"Negev":[2.122,2.710,None,1.165,1.619]},
("winter","SeasonalNaive"):      {"Center":[1.510,2.041,None,1.177,1.525],"Northwest":[1.657,2.231,None,1.286,1.683],"Negev":[1.682,2.238,None,1.331,1.712]},
}

rows, fails = [], []
for (season, model), regs in PRINTED.items():
    for reg, want in regs.items():
        e, er = MODELS[model](season, reg)
        mae, rmse = np.abs(e).mean(), np.sqrt((e ** 2).mean())
        upe = np.maximum(-e, 0).mean() if season == "summer" else np.maximum(e, 0).mean()
        m3, r3 = np.abs(er).mean(), np.sqrt((er ** 2).mean())
        got = [mae, rmse, upe, m3, r3]
        rows.append(dict(season=season, region=reg, model=model, n=len(e),
                         anchor_mae=round(mae, 6), anchor_rmse=round(rmse, 6), upe=round(upe, 6),
                         soft_mae3=round(m3, 6), soft_rmse3=round(r3, 6)))
        for name, g, w in zip(["anchor_mae", "anchor_rmse", "upe", "soft_mae3", "soft_rmse3"], got, want):
            if w is not None and f"{g:.3f}" != f"{w:.3f}":
                fails.append(f"{season}/{reg}/{model}/{name}: recomputed {g:.3f} != printed {w:.3f}")

df = pd.DataFrame(rows)
df.to_csv(AC / "manuscript_tables.csv", index=False)
summary = df[df.model.isin(["ConvNeXtTiny-TFT-NN", "ConvNeXtTiny-TFT", "ConvNeXtTiny-LSTM", "Tab-TFT", "Tab-LSTM"])]
summary.to_csv(AC / "metrics_summary_corrected.csv", index=False)
print(f"{len(rows)} cells recomputed; mismatches: {len(fails)}")
for f in fails:
    print("  MISMATCH", f)
sys.exit(1 if fails else 0)
