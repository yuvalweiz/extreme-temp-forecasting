"""ONE-COMMAND VERDICT ASSEMBLER — robust to partially-landed runs.
Usage:  python ~/expreal/verdict.py [> report]
Prints, for each output mode (std = anchor+soft current, mk4, mkt):
  1. per-family candidate tables (3-seed-mean plain-val + test) and the val PICK
  2. the matched-defaults CONTROL row
  3. seed-ensembled finals per region where runs exist (all-seeds + top3-by-val views)
Missing candidates print as 'pending'. Selection currency: mean plain validation
avg-over-outputs MAE (locked 2026-07-25). Deterministic; safe to re-run any time.
"""
import json
import glob
import os
import numpy as np
import pandas as pd

E = "/home/weizyuv/expreal"
C = f"{E}/Center_stable"
T = f"{E}/Center_tune"
REGIONS = ["Center", "Negev", "Northwest"]
COLS = ["tag", "pred_m1_hot", "pred_m1_p3", "pred_m1_p7", "pred_m1_p15",
        "true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15"]
PRED = COLS[1:5]; TRUE = COLS[5:9]

GRID_WIN = {}  # (fam, mode) -> grid-winner tag, discovered from Center_tune

def _runs(pats):
    out = []
    for pat in pats:
        for d in sorted(glob.glob(pat)):
            try:
                m = json.load(open(os.path.join(d, "meta.json")))
            except Exception:
                continue
            t = m.get("test_topk") or m.get("test")
            out.append(dict(dir=d, val=m["val"]["mae_mean"], test=t["mae_mean"],
                            sel=min(m["topk_scores"].values()) if m.get("topk_scores") else m["val"]["mae_mean"]))
    return out

def _stat(rs):
    return (len(rs), float(np.mean([r["val"] for r in rs])), float(np.mean([r["test"] for r in rs]))) if rs else None

def grid_winner(fam, mode):
    if (fam, mode) in GRID_WIN:
        return GRID_WIN[(fam, mode)]
    best = None
    for d in glob.glob(f"{T}/{fam}tg{mode}L*__*"):
        try:
            m = json.load(open(d + "/meta.json"))
        except Exception:
            continue
        v = min(m["topk_scores"].values())
        tag = d.split("/")[-1].split("__")[0]
        if best is None or v < best[0]:
            best = (v, tag)
    GRID_WIN[(fam, mode)] = best[1] if best else None
    return GRID_WIN[(fam, mode)]

def candidates(fam, mode):
    suf = "" if mode == "std" else f"_{mode}"
    wave1 = f"{C}/{fam}{suf}__*" if mode != "std" else None   # std wave-1 = published/deployed, handled per family
    cands = {}
    if mode != "std":
        cands["def+custom"] = _runs([f"{C}/{fam}_{mode}__*"])
        cands["def+plain"] = _runs([f"{C}/{fam}pm_{mode}__*"])
    gw = grid_winner(fam, mode)
    if gw:
        cands["tun+custom"] = _runs([f"{T}/{gw}__*", f"{T}/{fam}s2{mode}_a2b1__*"])
        cands["tun+a3b1.5"] = _runs([f"{T}/{fam}s2{mode}_a3b15__*"])
        cands["tun+plain"] = _runs([f"{T}/{fam}s2{mode}_a1b0__*"])
    if fam == "ours" and mode == "mkt":     # our-side pool + NN candidates (same family freedom)
        for k, pat in [("pool:KED", f"{E}/*_stable/pkked_mkt__*"), ("pool:ktuned", f"{E}/*_stable/pkktn_mkt__*"),
                       ("pool:masked", f"{E}/*_stable/pkmsk_mkt__*"), ("NN-frames", f"{C}/nnmkt__*")]:
            r = _runs([pat.replace("*_stable", "Center_stable")])
            if r:
                cands[k] = r
    return cands

def pick(fam, mode):
    cands = {k: v for k, v in candidates(fam, mode).items() if v}
    if not cands:
        return None, None, {}
    stats = {k: _stat(v) for k, v in cands.items()}
    best = min(stats, key=lambda k: stats[k][1])
    return best, stats[best], stats

def ens_metric(files, top3_by_val=None):
    fs = []
    for d in files:
        f = os.path.join(d, "preds_test_topk.csv")
        if not os.path.exists(f):
            f = os.path.join(d, "preds_test.csv")
        if os.path.exists(f):
            fs.append(f)
    if not fs:
        return None
    ens = pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag", as_index=False).mean()
    Er = np.abs(ens[PRED].to_numpy() - ens[TRUE].to_numpy())
    return float(Er.mean()), len(fs)

def main():
    print("=" * 78)
    print("FULL-PICTURE VERDICT (generated from whatever has landed; re-run any time)")
    print("=" * 78)
    fams_by_mode = {"mkt": ["ours", "stf", "atft", "stl"], "mk4": ["ours", "stf", "atft", "stl"],
                    "std": ["ours", "stf", "atft", "stl", "nnf"]}
    for mode in ["mkt", "mk4", "std"]:
        label = {"mkt": "TOP-K rank-free (t3,t5,t10,t15)", "mk4": "TOP-K with worst day (1,t5,t10,t15)",
                 "std": "CURRENT outputs (anchor+soft)"}[mode]
        print(f"\n### MODE {mode} — {label}")
        for fam in fams_by_mode[mode] + (["nnf"] if mode != "std" else []):
            best, bstat, stats = pick(fam, mode)
            if not best:
                print(f"  {fam:5s}: pending (no candidates landed)")
                continue
            print(f"  {fam:5s} PICK {best:12s} val={bstat[1]:.3f} -> TEST {bstat[2]:.3f}")
            for k, s in sorted(stats.items(), key=lambda kv: kv[1][1]):
                print(f"         {k:12s} n={s[0]}  val={s[1]:.3f}  test={s[2]:.3f}")
        if mode == "std":
            print("  NOTE std: 'ours' deployed reference (validated pool, published protocol):")
            print("        summer anchor Ce/NW/Ng = 2.113/1.809/2.007 | winter 1.275/1.440/1.331")
    print("\n### Seed-ensembled finals per region (family_mode tag families with >=3 runs)")
    for tagfam in ["ours_mkt", "stf_mkt", "atft_mkt", "stl_mkt", "stfFIN_mkt", "nnmkt", "nnfb"]:
        row = []
        for reg in REGIONS:
            r = ens_metric(sorted(glob.glob(f"{E}/{reg}_stable/{tagfam}__*")))
            row.append(f"{r[0]:.3f}(n{r[1]})" if r else "—")
        print(f"  {tagfam:12s} " + "  ".join(f"{x:>12s}" for x in row))
    print("\nDone. Selection currency: 3-seed-mean plain validation MAE (locked, pre-registered).")

if __name__ == "__main__":
    main()
