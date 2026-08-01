"""FOLD-IN evaluation: row-level actual-vs-predicted CSVs + metrics for the corrected
5-seed models (ours = frozen picks; ConvNeXtTiny-LSTM = hlst/clst; Tab-LSTM = atl/catl),
written in the same AVP format as article /results/actual_vs_predicted/.

Ensembling conventions (match foldin_numbers.py / run protocol):
  HOT : per-run top-5-epoch preds (preds_test_topk.csv) where the configuration uses
        them (all except ours-Negev-soft ng3_opt -> plain), averaged over seeds/runs.
  COLD: plain best-epoch preds (preds_test.csv), averaged over seeds (parity with csp).

Verifies OUR numbers against ~/expreal/foldin_package.txt values (hard assert, 2e-3).
Writes: article /results/actual_vs_predicted_corrected/{summer,winter}/{Region}/<Model>__<target>.csv
        + metrics_summary_corrected.csv (includes 3-region means).
Run:  python ~/expreal/foldin_avp.py
"""
import glob
import os

import numpy as np
import pandas as pd

OUT = "/home/weizyuv/article /results/actual_vs_predicted_corrected"
COLS = ["tag", "true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15",
        "pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15"]
REGIONS = [("Center", "Ce"), ("Negev", "Ne"), ("Northwest", "No")]
SOFT = ["p3", "p7", "p15"]

# (model label, season, region) -> (glob pattern, use_topk)
def spec(reg, S):
    E = "/home/weizyuv/expreal"
    # FINAL picks incl. the 2026-07-23 guarded upgrades (LOCKED; blends = 2-tuple patterns)
    return {
        ("ConvNeXtTiny-TFT", "summer", "anchor"): {
            "Center": (f"{E}/Center_stable/opt3__*", 1),
            "Negev": (f"{E}/Negev_stable/ng_keda3__*", 1),
            "Northwest": ((f"{E}/Northwest_stable/No_os3__*",
                           f"{E}/Northwest_stable/nw3_optm__*"), 1)}[reg],
        ("ConvNeXtTiny-TFT", "summer", "soft"): {
            "Center": (f"{E}/Center_stable/opa3_opt__*", 1),
            "Negev": ((f"{E}/Negev_stable/ng_optm__*",
                       f"{E}/Negev_stable/ng3_optm__*"), 1),
            "Northwest": (f"{E}/Northwest_stable/nw_ked__*", 1)}[reg],
        ("ConvNeXtTiny-TFT", "winter", "anchor"): (f"{E}/{reg}_cold/{S}_csp__*", 0),
        ("ConvNeXtTiny-TFT", "winter", "soft"): {
            # blends locked AS ADOPTED by the guarded one-shot = topk-preferred ensembles
            "Center": ((f"{E}/Center_cold/Ce_cked__*", f"{E}/Center_cold/Ce_ctpk__*"), 1),
            "Negev": ((f"{E}/Negev_cold/Ne_cselext__*", f"{E}/Negev_cold/Ne_cked__*"), 1),
            "Northwest": (f"{E}/Northwest_cold/No_csp__*", 0)}[reg],
        ("ConvNeXtTiny-LSTM", "summer", "both"): (f"{E}/{reg}_stable/hlst__*", 1),
        ("ConvNeXtTiny-LSTM", "winter", "both"): (f"{E}/{reg}_cold/{S}_clst__*", 0),
        # FEATURED ablation per author 2026-07-23: per-station 612-dim Tab-LSTM
        ("Tab-LSTM", "summer", "both"): (f"{E}/{reg}_stable/stl__*", 1),
        ("Tab-LSTM", "winter", "both"): (f"{E}/{reg}_cold/{S}_cstl__*", 0),
    }


def _ens1(pattern, topk):
    fs = []
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, f"preds_test{'_topk' if topk else ''}.csv")
        if not os.path.exists(f):
            f = os.path.join(d, "preds_test.csv")
        if os.path.exists(f):
            fs.append(f)
    if not fs:
        return None, 0
    d = pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag", as_index=False).mean()
    return d.sort_values("tag").reset_index(drop=True), len(fs)


def ens(pattern, topk):
    """pattern: glob (run dirs) or 2-tuple of globs -> 50/50 blend of the two ensembles."""
    if isinstance(pattern, tuple):
        d1, n1 = _ens1(pattern[0], topk); d2, n2 = _ens1(pattern[1], topk)
        if d1 is None or d2 is None:
            return None, 0
        m = d1.merge(d2, on="tag", suffixes=("_1", "_2"))
        d = pd.DataFrame({"tag": m["tag"]})
        for c in COLS[1:]:
            d[c] = (m[c + "_1"] + m[c + "_2"]) / 2 if c.startswith("pred") else m[c + "_1"]
        return d.sort_values("tag").reset_index(drop=True), n1 + n2
    return _ens1(pattern, topk)


def rows_anchor(d, season, region, model):
    t = d["true_m1_hot"].astype(float); p = d["pred_m1_hot"].astype(float)
    res = p - t
    return pd.DataFrame({
        "sample_id": [f"{region}_{season}_{x}" for x in d["tag"]],
        "date": d["tag"], "region": region, "season": season, "target": "anchor",
        "model": model, "split": "test", "actual": t, "predicted": p,
        "residual": res, "abs_error": res.abs(), "sq_error": res ** 2})


def rows_soft(d, season, region, model):
    t = d[[f"true_m1_{s}" for s in SOFT]].astype(float).to_numpy()
    p = d[[f"pred_m1_{s}" for s in SOFT]].astype(float).to_numpy()
    err = p - t
    out = pd.DataFrame({
        "sample_id": [f"{region}_{season}_{x}" for x in d["tag"]],
        "date": d["tag"], "region": region, "season": season, "target": "soft",
        "model": model, "split": "test",
        "actual": t.mean(1), "predicted": p.mean(1), "residual": err.mean(1),
        "abs_error": np.abs(err).mean(1), "sq_error": (err ** 2).mean(1)})
    for k, s in enumerate(SOFT):
        out[f"actual_{s}"] = t[:, k]; out[f"predicted_{s}"] = p[:, k]
    return out


def metrics(rows):
    a = rows["abs_error"].to_numpy(float)
    return dict(n=len(a), mae=float(a.mean()),
                rmse=float(np.sqrt(rows["sq_error"].to_numpy(float).mean())))


# frozen reference values from foldin_package.txt (OUR model; MAE/RMSE full-set)
REF = {("summer", "Center", "anchor"): (2.113, 2.713), ("summer", "Center", "soft"): (1.267, 1.551),
       ("summer", "Negev", "anchor"): (2.007, 2.511), ("summer", "Negev", "soft"): (1.537, 1.807),
       ("summer", "Northwest", "anchor"): (2.006, 2.547), ("summer", "Northwest", "soft"): (1.280, 1.607),
       ("winter", "Center", "anchor"): (1.275, 1.645), ("winter", "Center", "soft"): (0.986, 1.221),
       ("winter", "Negev", "anchor"): (1.331, 1.730), ("winter", "Negev", "soft"): (1.066, 1.362),
       ("winter", "Northwest", "anchor"): (1.440, 1.837), ("winter", "Northwest", "soft"): (1.006, 1.281)}


def main():
    summary, missing = [], []
    for reg, S in REGIONS:
        sp = spec(reg, S)
        jobs = [("ConvNeXtTiny-TFT", "summer", "anchor", *sp[("ConvNeXtTiny-TFT", "summer", "anchor")]),
                ("ConvNeXtTiny-TFT", "summer", "soft", *sp[("ConvNeXtTiny-TFT", "summer", "soft")]),
                ("ConvNeXtTiny-TFT", "winter", "anchor", *sp[("ConvNeXtTiny-TFT", "winter", "anchor")]),
                ("ConvNeXtTiny-TFT", "winter", "soft", *sp[("ConvNeXtTiny-TFT", "winter", "soft")]),
                ("ConvNeXtTiny-LSTM", "summer", "both", *sp[("ConvNeXtTiny-LSTM", "summer", "both")]),
                ("ConvNeXtTiny-LSTM", "winter", "both", *sp[("ConvNeXtTiny-LSTM", "winter", "both")]),
                ("Tab-LSTM", "summer", "both", *sp[("Tab-LSTM", "summer", "both")]),
                ("Tab-LSTM", "winter", "both", *sp[("Tab-LSTM", "winter", "both")])]
        for model, season, tgt, pattern, topk in jobs:
            d, n = ens(pattern, topk)
            if d is None:
                missing.append((model, season, reg, pattern)); continue
            odir = os.path.join(OUT, season, reg); os.makedirs(odir, exist_ok=True)
            targets = ["anchor", "soft"] if tgt == "both" else [tgt]
            for target in targets:
                rows = (rows_anchor if target == "anchor" else rows_soft)(d, season, reg, model)
                rows.to_csv(os.path.join(odir, f"{model}__{target}.csv"), index=False, float_format="%.6f")
                m = metrics(rows)
                summary.append(dict(season=season, region=reg, target=target, model=model,
                                    n_runs=n, **m))
                if model == "ConvNeXtTiny-TFT":
                    # REF asserts only for UNCHANGED picks, MAE only (foldin_package's
                    # soft "RMSE" used the legacy RMSE-of-mean convention; ours is the
                    # paper's per-rank RMSE_3 — not comparable). Upgraded cells (NW hot
                    # anchor, Negev hot soft, Ce+Ne winter soft) are report-only:
                    # pre-committed full-seed finals of the locked blend configs.
                    upgraded = (season, reg, target) in {
                        ("summer", "Northwest", "anchor"), ("summer", "Negev", "soft"),
                        ("winter", "Center", "soft"), ("winter", "Negev", "soft")}
                    ref = REF[(season, reg, target)]
                    if upgraded:
                        print(f"[OURS-UPGRADED] {season} {reg:10s} {target:6s} "
                              f"mae {m['mae']:.3f} rmse {m['rmse']:.3f} ({n} runs; "
                              f"pre-upgrade ref {ref[0]:.3f})")
                    else:
                        d_mae = abs(m["mae"] - ref[0])
                        tagx = "OK" if d_mae < 2e-3 else "MISMATCH!"
                        print(f"[verify OURS] {season} {reg:10s} {target:6s} "
                              f"mae {m['mae']:.3f} vs {ref[0]:.3f} {tagx}")
                        assert d_mae < 2e-3, f"frozen-number mismatch {season}/{reg}/{target}"
    sm = pd.DataFrame(summary)
    agg = (sm.groupby(["season", "target", "model"])
             .agg(n_regions=("region", "nunique"), n=("n", "sum"),
                  mae=("mae", "mean"), rmse=("rmse", "mean")).reset_index())
    agg.insert(1, "region", "MEAN-3REGIONS"); agg["n_runs"] = np.nan
    sm = pd.concat([sm, agg], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    sm.to_csv(os.path.join(OUT, "metrics_summary_corrected.csv"), index=False, float_format="%.6f")
    print(f"\nwrote {len(sm)} rows -> metrics_summary_corrected.csv")
    with pd.option_context("display.width", 200):
        print(sm[sm.region == "MEAN-3REGIONS"].round(3).to_string(index=False))
    for m in missing:
        print("[MISSING RUNS]", m)


if __name__ == "__main__":
    main()
