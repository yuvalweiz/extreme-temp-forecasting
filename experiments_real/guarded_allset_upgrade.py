"""PRE-REGISTERED one-shot: can OUR model legitimately improve its full-set MAE in the
two cells where the matched 5-seed Tab-LSTM leads (NW summer anchor, Negev summer soft),
using ONLY existing runs and validation-only selection?

PROCEDURE (fixed before any test evaluation; same guarded discipline as the project's
selection studies):
 1. Candidate pool per cell = every existing run-tag ensemble in the region's _stable dir
    (topk preds where available, else plain), plus 2-way blends (50/50) of the top-5
    tag-ensembles by validation criterion. Incumbent = the frozen pick.
 2. Validation criterion = full-set MAE on the VALIDATION split (the target metric of the
    revised protocol), computed on the run's saved val predictions.
 3. Guard: split validation chronologically into halves A and B. A challenger is
    admissible ONLY if it beats the incumbent's val MAE on BOTH halves (protects against
    val overfitting on 143 samples).
 4. Adopt the single best admissible challenger (by whole-val MAE); evaluate it on test
    EXACTLY ONCE and report. If none admissible -> report 'no legitimate upgrade'.
NO test values are read before step 4. Output: appended verdict to this file's log print.
"""
import glob
import os

import numpy as np
import pandas as pd

COLS = ["tag", "true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15",
        "pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15"]

CELLS = [
    dict(name="NW summer anchor", region="Northwest", target="anchor", kind="stable",
         incumbent="No_pbo", inc_tk=1, rival=("Tab-LSTM 5s", 1.886)),
    dict(name="Negev summer soft", region="Negev", target="soft", kind="stable",
         incumbent="ng3_opt", inc_tk=0, rival=("Tab-LSTM 5s", 1.492)),
    dict(name="Center winter soft", region="Center", target="soft", kind="cold",
         incumbent="Ce_csp", inc_tk=0, rival=("Tab-TFT catft (proxy)", 0.926)),
    dict(name="Negev winter soft", region="Negev", target="soft", kind="cold",
         incumbent="Ne_csp", inc_tk=0, rival=("Tab-TFT catft (proxy)", 1.000)),
    dict(name="NW winter soft", region="Northwest", target="soft", kind="cold",
         incumbent="No_csp", inc_tk=0, rival=("Tab-TFT catft (proxy)", 0.954)),
]


def load_ens(dirs, split, tk):
    fs = []
    for d in dirs:
        f = os.path.join(d, f"preds_{split}_topk.csv")
        if not (tk and os.path.exists(f)):
            f = os.path.join(d, f"preds_{split}.csv")
        if os.path.exists(f):
            fs.append(f)
    if not fs:
        return None
    frames = []
    for f in fs:
        try:
            x = pd.read_csv(f)
            if all(c in x.columns for c in COLS):
                frames.append(x[COLS])
        except Exception:
            pass
    if not frames:
        return None
    return pd.concat(frames).groupby("tag", as_index=False).mean().sort_values("tag")


def err(d, target):
    if target == "anchor":
        return (d["pred_m1_hot"] - d["true_m1_hot"]).abs().to_numpy(float)
    e = (d[[f"pred_m1_{s}" for s in ["p3", "p7", "p15"]]].to_numpy(float)
         - d[[f"true_m1_{s}" for s in ["p3", "p7", "p15"]]].to_numpy(float))
    return np.abs(e).mean(axis=1)


EXCLUDE_PREFIXES = ("ftv_", "ftv", "fta", "hlst", "atl", "atft",
                    "Ce_cft", "Ne_cft", "No_cft", "cft", "Ce_catl", "Ne_catl", "No_catl",
                    "Ce_catft", "Ne_catft", "No_catft", "Ce_clst", "Ne_clst", "No_clst",
                    "Ce_ctv", "Ne_ctv", "No_ctv")
# ftv_* = TRAINVAL+FIXED_STOP runs: trained on train+val, so their val preds are
# IN-SAMPLE -> invalid as a selection signal (leakage). fta*/atft/atl/hlst = tabular /
# comparison-model runs, not OUR-model candidates.


def tags_of(root):
    import json
    tags = {}
    for d in sorted(glob.glob(os.path.join(root, "*__temporalfusion__*")) +
                    glob.glob(os.path.join(root, "*__lstm__*"))):
        tag = os.path.basename(d).split("__")[0]
        if tag.startswith(EXCLUDE_PREFIXES) or "__lstm__" in os.path.basename(d):
            continue
        # ELIGIBILITY (corrected-frames paper): members must be trained on the corrected
        # stationvec/PAPERW synthesis, not the defective published frame bank
        # (criterion motivated by the erratum; discovered via nw_selsp, 2026-07-23)
        mfile = os.path.join(d, "meta.json")
        try:
            rd = json.load(open(mfile)).get("region_dir", "")
        except Exception:
            rd = ""
        if "stationvec" not in rd:
            if tag not in tags:
                print(f"  [ineligible: non-corrected frames] {tag} ({rd.split('/')[-1]})")
                tags[tag] = "SKIP"
            continue
        if tags.get(tag) == "SKIP":
            continue
        tags.setdefault(tag, []).append(d)
    return {t: ds for t, ds in tags.items() if ds != "SKIP"}


def main():
    for cell in CELLS:
        root = f"/home/weizyuv/expreal/{cell['region']}_{cell.get('kind','stable')}"
        pool = tags_of(root)
        # candidate val scores
        cand = {}
        inc_v = load_ens(pool[cell["incumbent"]], "val", 1)
        for tag, dirs in pool.items():
            v = load_ens(dirs, "val", 1)
            if v is None or len(v) < 100:
                continue
            # ELIGIBILITY: identical target definitions as the incumbent (guards against
            # target-set-ablation runs, e.g. TARGETS=1,5,10,15, whose p3/p7 columns are
            # different order statistics — discovered via ng_v2cw, 2026-07-23)
            mv = v.merge(inc_v, on="tag", suffixes=("", "_i"))
            tgt_ok = all((mv[f"true_m1_{s}"] - mv[f"true_m1_{s}_i"]).abs().max() < 1e-3
                         for s in ["hot", "p3", "p7", "p15"])
            if not tgt_ok:
                print(f"  [ineligible: different targets] {tag}")
                continue
            e = err(v, cell["target"])
            half = len(e) // 2
            cand[tag] = dict(val=float(e.mean()), a=float(e[:half].mean()),
                             b=float(e[half:].mean()), dirs=dirs, n=len(dirs))
        if cell["incumbent"] not in cand:
            print(f"[{cell['name']}] incumbent val preds missing — abort cell"); continue
        inc = cand[cell["incumbent"]]
        # blends of top-5 by whole-val
        top5 = sorted(cand, key=lambda t: cand[t]["val"])[:5]
        for i in range(len(top5)):
            for j in range(i + 1, len(top5)):
                t1, t2 = top5[i], top5[j]
                v1 = load_ens(cand[t1]["dirs"], "val", 1); v2 = load_ens(cand[t2]["dirs"], "val", 1)
                m = v1.merge(v2, on="tag", suffixes=("_1", "_2"))
                d = pd.DataFrame({"tag": m["tag"]})
                for c in COLS[1:]:
                    d[c] = (m[c + "_1"] + m[c + "_2"]) / 2 if c.startswith("pred") else m[c + "_1"]
                e = err(d, cell["target"]); half = len(e) // 2
                cand[f"BLEND:{t1}+{t2}"] = dict(val=float(e.mean()), a=float(e[:half].mean()),
                                               b=float(e[half:].mean()),
                                               dirs=(cand[t1]["dirs"], cand[t2]["dirs"]), n=0)
        admissible = {t: c for t, c in cand.items()
                      if t != cell["incumbent"] and c["a"] < inc["a"] and c["b"] < inc["b"]}
        print(f"\n=== {cell['name']} — incumbent {cell['incumbent']} "
              f"val={inc['val']:.3f} (A {inc['a']:.3f} / B {inc['b']:.3f}) | "
              f"rival {cell['rival'][0]} test={cell['rival'][1]:.3f}")
        print(f"pool={len(cand)} candidates, admissible={len(admissible)}")
        if not admissible:
            print("VERDICT: no legitimate upgrade — no candidate beats the incumbent on both val halves.")
            continue
        best = min(admissible, key=lambda t: admissible[t]["val"])
        c = admissible[best]
        print(f"ADOPTED (pre-test): {best} val={c['val']:.3f} (A {c['a']:.3f} / B {c['b']:.3f})")
        # ---- single test evaluation ----
        if best.startswith("BLEND:"):
            d1 = load_ens(c["dirs"][0], "test", 1); d2 = load_ens(c["dirs"][1], "test", 1)
            m = d1.merge(d2, on="tag", suffixes=("_1", "_2"))
            t = pd.DataFrame({"tag": m["tag"]})
            for col in COLS[1:]:
                t[col] = (m[col + "_1"] + m[col + "_2"]) / 2 if col.startswith("pred") else m[col + "_1"]
        else:
            t = load_ens(c["dirs"], "test", 1)
        e = err(t, cell["target"])
        inc_t = load_ens(cand[cell["incumbent"]]["dirs"], "test", 1 if cell["inc_tk"] else 0)
        einc = err(inc_t, cell["target"])
        print(f"ONE-SHOT TEST: challenger={e.mean():.3f}  incumbent={einc.mean():.3f}  "
              f"rival={cell['rival'][1]:.3f}  -> "
              f"{'BEATS RIVAL' if e.mean() < cell['rival'][1] else 'does NOT beat rival'}")


if __name__ == "__main__":
    main()
