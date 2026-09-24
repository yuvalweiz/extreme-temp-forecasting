"""Per-seed anchor-MAE spread for EVERY five-seed deep ensemble in the result tables
(featured learned pipeline, kernel variant, LSTM head, and both per-station tabular
ablations), both seasons, all regions. Per-seed predictions are read from the run
directories the released ensembles were averaged from (foldin_avp / gather_framingA
glob specs); truth is joined from the released AVP actuals so every value reconciles
with the printed tables. Blended deployments pool the seeds of both members.
"""
import glob, os, json, json
import numpy as np
import pandas as pd

E = "/home/weizyuv/expreal"
R = "/home/weizyuv/article /repo/results/actual_vs_predicted_corrected"
S = {"Center": "Ce", "Negev": "Ne", "Northwest": "No"}

def pats(model, season, reg):
    s = S[reg]
    if model == "ConvNeXtTiny-TFT-NN":
        # anchor-target deployments per guarded_nn_featured.py (summer) / single nnfwin (winter)
        if season == "winter":
            return [f"{E}/{reg}_cold/nnfwin__*"]
        return {"Center": [f"{E}/Center_tune/nnfs2std_a1b0__*", f"{E}/Center_stable/nnfb__*"],
                "Northwest": [f"{E}/Northwest_stable/nnfFIN_std__*"],
                "Negev": [f"{E}/Negev_stable/nnfP2FIN__*"]}[reg]
    if model == "ConvNeXtTiny-TFT":
        if season == "summer":
            return {"Center":[f"{E}/Center_stable/opt3__*"],
                    "Negev":[f"{E}/Negev_stable/ng_keda3__*"],
                    "Northwest":[f"{E}/Northwest_stable/No_os3__*", f"{E}/Northwest_stable/nw3_optm__*"]}[reg]
        return [f"{E}/{reg}_cold/{s}_csp__*"]
    if model == "ConvNeXtTiny-LSTM":
        return [f"{E}/{reg}_stable/hlst__*"] if season=="summer" else [f"{E}/{reg}_cold/{s}_clst__*"]
    if model == "Tab-LSTM":
        if season == "winter":
            return [f"{E}/{reg}_cold/{s}_cstl__*"]
        return [f"{E}/Center_tune/stls2std_a1b0__*"] if reg=="Center" else [f"{E}/{reg}_stable/stlFIN_std__*"]
    if model == "Tab-TFT":
        if season == "summer":
            return [f"{E}/Center_tune/stfs2std_a1b0__*"] if reg=="Center" else [f"{E}/{reg}_stable/stfSTDFIN__*"]
        return [f"{E}/{reg}_cold/stfwin__*"]
    raise ValueError(model)

rows = []
for model in ["ConvNeXtTiny-TFT-NN", "ConvNeXtTiny-TFT", "ConvNeXtTiny-LSTM", "Tab-TFT", "Tab-LSTM"]:
    for season in ["summer", "winter"]:
        for reg in ["Center", "Northwest", "Negev"]:
            avp = pd.read_csv(f"{R}/{season}/{reg}/{model}__anchor.csv", parse_dates=["date"])
            truth = avp.set_index(avp["date"].dt.strftime("%Y-%m-%d"))["actual"]
            ens = avp["abs_error"].mean()
            maes = []
            for g in pats(model, season, reg):
                for d in sorted(glob.glob(g)):
                    f = os.path.join(d, "preds_test_topk.csv")
                    if not os.path.exists(f):
                        f = os.path.join(d, "preds_test.csv")
                    if not os.path.exists(f):
                        continue
                    p = pd.read_csv(f)
                    p["tag"] = p["tag"].astype(str).str[:10]
                    p = p.set_index("tag")
                    if "pred_m1_hot" not in p.columns:
                        continue
                    j = truth.to_frame("y").join(p["pred_m1_hot"].rename("p"), how="inner").dropna()
                    if len(j) > 100:
                        maes.append((j["p"] - j["y"]).abs().mean())
            sm = np.array(maes)
            # reproduction check: equal-weight average of member ensembles (one member per glob)
            mem = []
            for g in pats(model, season, reg):
                fs = []
                for d in sorted(glob.glob(g)):
                    f = os.path.join(d, "preds_test_topk.csv")
                    if not os.path.exists(f):
                        f = os.path.join(d, "preds_test.csv")
                    if os.path.exists(f):
                        p = pd.read_csv(f); p["tag"] = p["tag"].astype(str).str[:10]
                        fs.append(p.set_index("tag")["pred_m1_hot"])
                if fs:
                    mem.append(pd.concat(fs, axis=1).mean(axis=1))
            rep = np.nan
            if mem:
                j = truth.to_frame("y").join(pd.concat(mem, axis=1).mean(axis=1).rename("p"), how="inner").dropna()
                rep = (j["p"] - j["y"]).abs().mean()
            be, tk = [], []
            key = "mae_hot_all" if season == "summer" else "mae_cold_all"
            for g in pats(model, season, reg):
                for d in sorted(glob.glob(g)):
                    f = os.path.join(d, "meta.json")
                    if os.path.exists(f):
                        m = json.load(open(f))
                        if "test" in m and "test_topk" in m and key in m["test"] and key in m["test_topk"]:
                            be.append(m["test"][key]); tk.append(m["test_topk"][key])
            rows.append(dict(model=model, season=season, region=reg, n=len(sm),
                             ens=ens, reproduced=rep, seed_mean=sm.mean() if len(sm) else np.nan,
                             seed_std=sm.std(ddof=1) if len(sm) > 1 else np.nan,
                             n_cmp=len(be), top5_minus_best=(np.mean(tk)-np.mean(be)) if be else np.nan,
                             runs_top5_better=int((np.array(tk) < np.array(be)).sum()) if be else 0))
df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(df.round(3).to_string(index=False))
print("\nseed_std: min %.3f max %.3f median %.3f (n cells with >=2 seeds: %d)" %
      (df.seed_std.min(), df.seed_std.max(), df.seed_std.median(), df.seed_std.notna().sum()))
piv = df.pivot_table(index=["model","season"], columns="region", values="seed_std")
print("\nper-cell seed std:\n", piv.round(3).to_string())
df.to_csv(f"{E}/seed_variance_all.csv", index=False)
