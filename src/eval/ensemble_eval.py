"""
Legitimate model-strengthening: ensemble the TOP val-ranked runs (by validation,
never by test) and evaluate the ensemble once on test. Averaging across seeds /
heads / top-val configs reduces variance so the val-selected model also tests
well -- it does NOT peek at test (configs are ranked by val_select only).

Usage:
  python ensemble_eval.py <run_dir1> <run_dir2> ...     # explicit members
  python ensemble_eval.py --auto <glob> --topk K        # pick top-K by val_select
Each run dir must contain meta.json + preds_val.csv + preds_test.csv.
Reports val (for selection) and the single test of the chosen ensemble.
"""
import sys, glob, json, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # <repo>/src
import repo_paths as RP


def load_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    val = pd.read_csv(os.path.join(d, "preds_val.csv"))
    test = pd.read_csv(os.path.join(d, "preds_test.csv"))
    # train-p90 extreme threshold is stored per dataset; recompute proxy from val if absent
    return meta, val, test


def hot_mae(df, thr):
    e = (df["pred_m1_hot"] - df["true_m1_hot"]).abs()
    ext = df["true_m1_hot"] >= thr
    return float(e.mean()), float(e[ext].mean()) if ext.any() else float("nan")


def ensemble(dirs, thr):
    metas, vals, tests = [], [], []
    for d in dirs:
        m, v, t = load_run(d)
        metas.append(m); vals.append(v.set_index("tag")); tests.append(t.set_index("tag"))
    # average predictions on common tags
    def avg(frames):
        cols = ["pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15"]
        common = set(frames[0].index)
        for f in frames[1:]:
            common &= set(f.index)
        common = sorted(common)
        base = frames[0].loc[common].copy()
        for c in cols:
            if all(c in f.columns for f in frames):
                base[c] = np.mean([f.loc[common, c].values for f in frames], axis=0)
        return base.reset_index()
    ve, te = avg(vals), avg(tests)
    return ve, te


def main():
    args = sys.argv[1:]
    THR = float(os.environ.get("THR", "37.72"))  # Center train-p90 default
    if args and args[0] == "--auto":
        g = args[1]; topk = int(args[args.index("--topk") + 1]) if "--topk" in args else 3
        cand = []
        for m in glob.glob(g):
            d = os.path.dirname(m)
            meta = json.load(open(m))
            v = pd.read_csv(os.path.join(d, "preds_val.csv"))
            va, vx = hot_mae(v, THR)
            cand.append((0.5 * (va + vx), d))
        cand.sort()
        dirs = [d for _, d in cand[:topk]]
        print(f"[auto] top-{topk} by val_select: " + ", ".join(os.path.basename(d) for d in dirs))
    else:
        dirs = args
    if not dirs:
        print("no run dirs"); return
    ve, te = ensemble(dirs, THR)
    va, vx = hot_mae(ve, THR); ta, tx = hot_mae(te, THR)
    print(f"\n=== ENSEMBLE of {len(dirs)} val-selected runs ===")
    print(f"  VAL (selection):  all={va:.3f} ext={vx:.3f}")
    print(f"  TEST (reported):  all={ta:.3f} ext={tx:.3f}")
    out = os.path.join(RP.results_dir(), "ensemble_test.csv")
    te.to_csv(out, index=False)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
