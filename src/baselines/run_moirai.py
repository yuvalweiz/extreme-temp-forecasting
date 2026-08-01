"""
Strong baseline B2: Moirai-1.1-R (zero-shot) -> a SECOND foundational forecaster
(complements TimesFM 2.5) so the paper's "we beat foundational models" is plural.
Non-spatial: sees only the region's daily cluster series, forecasts the next 30
days, reduces to p1/p3/p7/p15 with the SAME sort-descending [0,2,6,14] protocol as
every other model (docs/BASELINE_PROTOCOL). No leakage: context is strictly the
past up to each prediction day -- mirrors run_timesfm.py exactly.

Run (env python):  REGION=Center python run_moirai.py
Writes: repo/results/moirai_<REGION>.csv + prints MAE vs our ConvNeXt-TFT.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))                    # this dir
import repo_paths as RP
import series as S

REGION = os.environ.get("REGION", "Center")
TARGET = os.environ.get("TARGET_COL", "max_dry_temp")
HOTTEST = TARGET == "max_dry_temp"
MAX_CONTEXT = int(os.environ.get("MAX_CONTEXT", "512"))
SIZE = os.environ.get("MOIRAI_SIZE", "large")   # base|large
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "100"))
OUT = RP.results_dir()


def order_stats(vals, hottest=True):
    s = np.sort(np.asarray(vals, float))[::-1] if hottest else np.sort(np.asarray(vals, float))
    return s[0], s[2], s[6], s[14]


def main():
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from gluonts.dataset.common import ListDataset

    cd = S.build_cluster_daily(REGION, cols=[TARGET])
    y = cd[TARGET].interpolate(limit_direction="both")
    df = S.build_targets(cd[TARGET], hottest=HOTTEST)
    test = df[df.split == "test"].reset_index(drop=True)
    train = df[df.split == "train"]
    thr = float(np.quantile(train.true_m1_hot, 0.90 if HOTTEST else 0.10))

    yi = y.dropna()
    contexts, ppoints = [], []
    for _, r in test.iterrows():
        pp = pd.Timestamp(r["pred_point"])
        ctx = yi.loc[:pp].to_numpy(float)[-MAX_CONTEXT:]
        if len(ctx) < 64:
            continue
        contexts.append(ctx)
        ppoints.append(r["tag"])

    print(f"[Moirai-{SIZE}/{REGION}] {len(contexts)} test contexts | extreme thr={thr:.2f}")
    module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}")
    model = MoiraiForecast(
        module=module, prediction_length=30, context_length=MAX_CONTEXT,
        patch_size="auto", num_samples=NUM_SAMPLES,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=32)

    # one gluonts series per strictly-past context; forecast 30 steps ahead
    ds = ListDataset(
        [{"start": pd.Period("2000-01-01", freq="D"), "target": c.astype(np.float32)} for c in contexts],
        freq="D",
    )
    forecasts = list(predictor.predict(ds))

    rows = []
    for tag, fc in zip(ppoints, forecasts):
        med = np.median(fc.samples, axis=0)  # (30,)
        p1, p3, p7, p15 = order_stats(med, HOTTEST)
        row = {"tag": tag, "pred_m1_hot": p1, "pred_m1_p3": p3, "pred_m1_p7": p7, "pred_m1_p15": p15}
        row.update({f"fc_d{i+1}": float(v) for i, v in enumerate(np.asarray(med, float)[:30])})
        rows.append(row)
    pred = pd.DataFrame(rows).merge(test, on="tag")
    os.makedirs(OUT, exist_ok=True)
    suffix = "" if HOTTEST else "_min"
    pred.to_csv(os.path.join(OUT, f"moirai_{REGION}{suffix}.csv"), index=False)

    ext = pred.true_m1_hot >= thr
    def mae(col_t, col_p, mask=None):
        e = (pred[col_p] - pred[col_t]).abs()
        return float(e[mask].mean()) if mask is not None else float(e.mean())
    print(f"\n=== Moirai-1.1-R-{SIZE} zero-shot (non-spatial) — MAE C ({REGION}) ===")
    print(f"  p1(hot): all={mae('true_m1_hot','pred_m1_hot'):.3f}  ext={mae('true_m1_hot','pred_m1_hot',ext):.3f}")
    for t in ["p3", "p7", "p15"]:
        print(f"  {t:3s}    : all={mae(f'true_m1_{t}',f'pred_m1_{t}'):.3f}")
    print(f"  (bar to beat: ConvNeXt-TFT {REGION} hot all ~2.07 / our spatial model)")


if __name__ == "__main__":
    main()
