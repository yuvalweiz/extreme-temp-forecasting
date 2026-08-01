"""
Strong baseline B1: TimesFM 2.5 (200M, zero-shot) -> the foundational forecaster
R4 + meta-review demanded. Non-spatial: sees only the region's daily cluster
series, forecasts the next 30 days, and reduces to p1/p3/p7/p15 with the SAME
sort-descending [0,2,6,14] protocol as every other model (docs/BASELINE_PROTOCOL).
No leakage: context is strictly the past up to each prediction day.

Run (env python):  python run_timesfm.py  (REGION env, default Center)
Writes: repo/results/timesfm_<REGION>.csv + prints MAE vs our ConvNeXt-TFT.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))                    # this dir
import repo_paths as RP
import series as S

REGION = os.environ.get("REGION", "Center")
TARGET = os.environ.get("TARGET_COL", "max_dry_temp")
HOTTEST = TARGET == "max_dry_temp"
MAX_CONTEXT = 1024
OUT = RP.results_dir()


def order_stats(vals, hottest=True):
    s = np.sort(np.asarray(vals, float))[::-1] if hottest else np.sort(np.asarray(vals, float))
    return s[0], s[2], s[6], s[14]


def main():
    import timesfm
    cd = S.build_cluster_daily(REGION, cols=[TARGET])
    y = cd[TARGET].interpolate(limit_direction="both")  # continuous daily series for context
    df = S.build_targets(cd[TARGET], hottest=HOTTEST)
    test = df[df.split == "test"].reset_index(drop=True)
    train = df[df.split == "train"]
    thr = float(np.quantile(train.true_m1_hot, 0.90 if HOTTEST else 0.10))

    # build contexts (strictly past) for each test pred_point
    contexts, ppoints = [], []
    yi = y.dropna()
    for _, r in test.iterrows():
        pp = pd.Timestamp(r["pred_point"])
        ctx = yi.loc[:pp].to_numpy(float)[-MAX_CONTEXT:]
        if len(ctx) < 64:
            continue
        contexts.append(ctx); ppoints.append(r["tag"])

    print(f"[TimesFM/{REGION}] {len(contexts)} test contexts | extreme thr={thr:.2f}")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=MAX_CONTEXT, max_horizon=32,
                                         normalize_inputs=True, use_continuous_quantile_head=True,
                                         force_flip_invariance=True, infer_is_positive=True,
                                         fix_quantile_crossing=True))
    # forecast in batches
    preds = []
    B = 64
    for i in range(0, len(contexts), B):
        pf, _ = model.forecast(horizon=30, inputs=contexts[i:i + B])
        preds.append(np.asarray(pf))
    pf = np.concatenate(preds, 0)  # (N,30)

    rows = []
    for tag, fc in zip(ppoints, pf):
        p1, p3, p7, p15 = order_stats(fc, HOTTEST)
        row = {"tag": tag, "pred_m1_hot": p1, "pred_m1_p3": p3, "pred_m1_p7": p7, "pred_m1_p15": p15}
        row.update({f"fc_d{i+1}": float(v) for i, v in enumerate(np.asarray(fc, float)[:30])})
        rows.append(row)
    pred = pd.DataFrame(rows).merge(test, on="tag")
    os.makedirs(OUT, exist_ok=True)
    suffix = "" if HOTTEST else "_min"
    pred.to_csv(os.path.join(OUT, f"timesfm_{REGION}{suffix}.csv"), index=False)

    ext = pred.true_m1_hot >= thr
    def mae(col_t, col_p, mask=None):
        e = (pred[col_p] - pred[col_t]).abs()
        return float(e[mask].mean()) if mask is not None else float(e.mean())
    print("\n=== TimesFM 2.5 zero-shot (non-spatial) — MAE C ===")
    print(f"  p1(hot): all={mae('true_m1_hot','pred_m1_hot'):.3f}  ext={mae('true_m1_hot','pred_m1_hot',ext):.3f}")
    for t in ["p3", "p7", "p15"]:
        print(f"  {t:3s}    : all={mae(f'true_m1_{t}',f'pred_m1_{t}'):.3f}")
    soft_t = pred[["true_m1_p3","true_m1_p7","true_m1_p15"]].mean(1)
    soft_p = pred[["pred_m1_p3","pred_m1_p7","pred_m1_p15"]].mean(1)
    print(f"  soft(avg p3/7/15): all={float((soft_p-soft_t).abs().mean()):.3f}")
    print(f"\n  (bar to beat: ConvNeXt-TFT Center hot all~2.16 / our spatial model)")


if __name__ == "__main__":
    main()
