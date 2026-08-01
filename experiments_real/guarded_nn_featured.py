"""PRE-REGISTERED featured-row selection for the NN pipeline (Framing A), using the
paper's guarded procedure verbatim (guarded_allset_upgrade.py discipline):
  Per summer region x target (anchor = m1_hot; soft = mean p3/p7/p15):
  1. Pool = every NN-frames std run-tag ensemble (all seeds of a tag, val preds
     averaged) + 50/50 blends of the top-5 tags by val criterion.
     Tags: nnfb (deployed recipe = INCUMBENT), nnfFIN_std / nnfs2std_* / nnfp2std_*
     (tuned variants), per region.
  2. Criterion = full-set VAL MAE of the target.
  3. Guard: chronological val halves A/B; challenger admissible iff it beats the
     incumbent on BOTH halves.
  4. Adopt best admissible by whole-val MAE; evaluate on TEST once; report.
  Winter: single tag (nnfwin) exists -> incumbent stands by construction.
No test values are read before step 4 (test preds loaded only for the adopted pick
and the incumbent)."""
import glob
import os
import numpy as np
import pandas as pd

E = "/home/weizyuv/expreal"
VC = ["tag", "pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15",
      "true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15"]

def tag_dirs(region):
    pats = [f"{E}/{region}_stable/nnfb__*", f"{E}/{region}_stable/nnfFIN_std__*",
            f"{E}/{region}_stable/nnfP2FIN__*"]
    if region == "Center":
        pats += [f"{E}/Center_tune/nnfs2std_*__*", f"{E}/Center_tune/nnfp2std_*__*"]
    groups = {}
    for p in pats:
        for d in glob.glob(p):
            key = os.path.basename(d).split("__")[0]
            if region == "Center" and "_tune/" in d:
                key = "tune:" + key
            groups.setdefault(key, []).append(d)
    return {k: v for k, v in groups.items() if len(v) >= 3}

def ens(dirs, which):
    fs = [os.path.join(d, f"preds_{which}_topk.csv") for d in dirs]
    fs = [f if os.path.exists(f) else f.replace("_topk", "") for f in fs]
    fs = [f for f in fs if os.path.exists(f)]
    if not fs:
        return None
    return pd.concat([pd.read_csv(f)[VC] for f in fs]).groupby("tag", as_index=False).mean().sort_values("tag")

def crit(df, target):
    if target == "anchor":
        return np.abs(df.pred_m1_hot - df.true_m1_hot).to_numpy()
    return np.mean([np.abs(df[f"pred_m1_{t}"] - df[f"true_m1_{t}"]).to_numpy() for t in ["p3", "p7", "p15"]], 0)

def blend(a, b):
    m = a.merge(b, on="tag", suffixes=("_x", "_y"))
    out = pd.DataFrame({"tag": m.tag})
    for c in VC[1:]:
        out[c] = (m[c + "_x"] + m[c + "_y"]) / 2
    return out

print("GUARDED NN FEATURED-ROW SELECTION (summer)")
for region in ["Center", "Negev", "Northwest"]:
    groups = tag_dirs(region)
    vals = {k: ens(v, "val") for k, v in groups.items()}
    vals = {k: v for k, v in vals.items() if v is not None}
    inc = "nnfb"
    if inc not in vals:
        print(f"{region}: incumbent nnfb missing"); continue
    for target in ["anchor", "soft"]:
        scored = {k: crit(v, target) for k, v in vals.items()}
        whole = {k: s.mean() for k, s in scored.items()}
        top5 = sorted(whole, key=whole.get)[:5]
        cands = dict(vals)
        for i in range(len(top5)):
            for j in range(i + 1, len(top5)):
                cands[f"blend:{top5[i]}+{top5[j]}"] = blend(vals[top5[i]], vals[top5[j]])
        cscored = {k: crit(v, target) for k, v in cands.items()}
        cwhole = {k: s.mean() for k, s in cscored.items()}
        n = len(cscored[inc]); half = n // 2
        incA, incB = cscored[inc][:half].mean(), cscored[inc][half:].mean()
        admissible = [k for k, s in cscored.items() if k != inc
                      and s[:half].mean() < incA and s[half:].mean() < incB]
        pick = min(admissible, key=lambda k: cwhole[k]) if admissible else inc
        # single test evaluation of pick (+ incumbent reference)
        def test_of(key):
            if key.startswith("blend:"):
                a, b = key[6:].split("+")
                ta, tb = ens(groups.get(a.replace("tune:",""), []) if not a.startswith("tune:") else groups[a], "test"), None
                # rebuild via same group lookup
                ta = ens(groups[a], "test"); tb = ens(groups[b], "test")
                return crit(blend(ta, tb), target).mean()
            return crit(ens(groups[key], "test"), target).mean()
        t_pick, t_inc = test_of(pick), test_of(inc)
        print(f"  {region:10s} {target:6s}: pick={pick:34s} val {cwhole[pick]:.3f} -> TEST {t_pick:.3f}"
              f"  (incumbent nnfb: val {cwhole[inc]:.3f}, test {t_inc:.3f}; admissible={len(admissible)})")
