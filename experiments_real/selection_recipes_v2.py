"""Selection-recipe study v2: which VAL-only rule best predicts TEST performance?

Corpus: all diag_epoch_preds.csv.gz curves (per-epoch val+test preds).
Discipline: rules are ranked on DEV curves (seed 111) and the ranking is then scored on
HELD-OUT curves (seeds 222/333). Regret = rule's test MeanMAE - oracle-best achievable
test MeanMAE on that curve (oracle over single epochs AND all top-k sets we consider).
"""
import pandas as pd, numpy as np, glob, os, json

def hthr(reg):
    z = np.load(f"/home/weizyuv/Deep Learning Models/Cluster {reg}/dataset_FULL_h180_next30_DOM_1_7_14_21_28/norm_stats_extremes_full_yNONE_v1.npz", allow_pickle=True)
    for k in ["hot_p90", "p90", "hot_thr"]:
        if k in z: return float(z[k])
    raise KeyError(list(z.keys()))

THR = {"Center": hthr("Center")}
PCOLS = None

def epoch_metrics(d, thr):
    """d: one epoch's rows for one split -> (all, ext, mean) anchor MAE."""
    ae = (d["pred_m1_hot"] - d["true_m1_hot"]).abs()
    m = d["true_m1_hot"] >= thr
    a, e = ae.mean(), ae[m].mean()
    return a, e, (a + e) / 2.0

def ens_metrics(sub, thr):
    """sub: rows of several epochs (same split) -> metrics of the per-tag MEAN prediction."""
    g = sub.groupby("tag", as_index=False).agg({"pred_m1_hot": "mean", "true_m1_hot": "first"})
    return epoch_metrics(g, thr)

def load_curve(path, reg="Center"):
    df = pd.read_csv(path)
    thr = THR[reg]
    eps = sorted(df["ep"].unique())
    V = {s: df[(df.split == "val") & (df.ep == s)] for s in eps}
    T = {s: df[(df.split == "test") & (df.ep == s)] for s in eps}
    vm = {e: epoch_metrics(V[e], thr) for e in eps}
    tm = {e: epoch_metrics(T[e], thr) for e in eps}
    return dict(eps=eps, V=V, T=T, vm=vm, tm=tm, thr=thr, df=df)

# ---------------- rules: return (chosen_eps:list, note) using VAL ONLY ----------------
def r_best_val_all(c):  return [min(c["eps"], key=lambda e: c["vm"][e][0])]
def r_best_val_ext(c):  return [min(c["eps"], key=lambda e: c["vm"][e][1])]
def r_best_val_mean(c): return [min(c["eps"], key=lambda e: c["vm"][e][2])]
def r_topk(k, key=2):
    def f(c): return sorted(c["eps"], key=lambda e: c["vm"][e][key])[:k]
    return f
def r_goal(delta=0.05):
    def f(c):
        best_all = min(c["vm"][e][0] for e in c["eps"])
        cand = [e for e in c["eps"] if c["vm"][e][0] <= best_all + delta]
        return [min(cand, key=lambda e: c["vm"][e][1])]
    return f
def r_fixed(ep):
    def f(c): return [min(c["eps"], key=lambda e: abs(e - ep))]
    return f
def r_last(c): return [c["eps"][-1]]
def r_top5_window(c):
    w = [e for e in c["eps"] if 3 <= e <= 30] or c["eps"]
    return sorted(w, key=lambda e: c["vm"][e][2])[:5]
def r_ema_best(alpha=0.3):
    def f(c):
        s, best_e, best_s = None, None, None
        for e in c["eps"]:
            x = c["vm"][e][2]
            s = x if s is None else alpha * x + (1 - alpha) * s
            if best_s is None or s < best_s: best_s, best_e = s, e
        return [best_e]
    return f
def r_top5_val_all(c): return sorted(c["eps"], key=lambda e: c["vm"][e][0])[:5]

RULES = {
    "best_val_all": r_best_val_all, "best_val_ext": r_best_val_ext,
    "best_val_mean": r_best_val_mean,
    "top3_mean": r_topk(3), "top5_mean": r_topk(5), "top7_mean": r_topk(7),
    "top10_mean": r_topk(10), "top5_ext": r_topk(5, key=1), "top5_all": r_top5_val_all,
    "goal_d05": r_goal(0.05), "goal_d10": r_goal(0.10),
    "fixed_ep3": r_fixed(3), "fixed_ep5": r_fixed(5), "fixed_ep8": r_fixed(8),
    "last_ep": r_last, "top5_win3_30": r_top5_window, "ema_best": r_ema_best(),
}

def score_rule(c, eps_sel):
    if len(eps_sel) == 1:
        return c["tm"][eps_sel[0]]
    sub = c["df"][(c["df"].split == "test") & (c["df"].ep.isin(eps_sel))]
    return ens_metrics(sub, c["thr"])

def oracle(c):
    best = min(c["tm"][e][2] for e in c["eps"])
    for k in (3, 5, 7):   # oracle also allowed the k-means we consider (val-blind upper bound-ish)
        for key in (0, 1, 2):
            eps = sorted(c["eps"], key=lambda e: c["vm"][e][key])[:k]
            best = min(best, score_rule(c, eps)[2])
    return best

files = sorted(glob.glob("/home/weizyuv/expreal/*_diag/*/diag_epoch_preds.csv.gz"))
rows = []
for f in files:
    run = f.split("/")[-2]
    if not run.startswith("diag_"):     # only hot-Center corpus for now (cold uses other col names)
        continue
    seed = run.split("__s")[-1]
    c = load_curve(f)
    oc = oracle(c)
    for name, rule in RULES.items():
        eps_sel = rule(c)
        a, e, m = score_rule(c, eps_sel)
        rows.append(dict(run=run, seed=seed, rule=name, test_all=a, test_ext=e,
                         test_mean=m, regret=m - oc, n_eps=len(eps_sel)))
R = pd.DataFrame(rows)
dev, held = R[R.seed == "111"], R[R.seed != "111"]
print(f"curves: {R.run.nunique()} (dev {dev.run.nunique()} / held {held.run.nunique()})\n")
agg = lambda d: d.groupby("rule").agg(regret=("regret", "mean"), mean=("test_mean", "mean"),
                                      all_=("test_all", "mean"), ext=("test_ext", "mean")).sort_values("regret")
print("=== DEV (seed 111) — rules ranked by mean regret ==="); print(agg(dev).round(4))
print("\n=== HELD-OUT (seeds 222/333) — same rules, honest scores ==="); print(agg(held).round(4))
best_dev = agg(dev).index[0]
h = agg(held)
print(f"\nDEV winner: {best_dev} -> held-out regret {h.loc[best_dev,'regret']:.4f} "
      f"(held-out best possible {h.regret.min():.4f} by {h.regret.idxmin()})")
