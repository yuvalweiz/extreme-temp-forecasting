"""Guarded pick upgrades: challenger replaces the frozen incumbent ONLY if it beats it on
BOTH chronological halves of the validation set (family Mean-MAE). Test evaluated once,
only for admitted challengers. This is the pre-registered anti-val-overfit rule.
"""
import pandas as pd, numpy as np, glob

COLS = ["tag","true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15",
        "pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]
def hthr(reg):
    z=np.load(f"/home/weizyuv/Deep Learning Models/Cluster {reg}/dataset_FULL_h180_next30_DOM_1_7_14_21_28/norm_stats_extremes_full_yNONE_v1.npz",allow_pickle=True)
    for k in ["hot_p90","p90","hot_thr"]:
        if k in z: return float(z[k])
def cthr(reg):
    d=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/Winter Models/dataset_FULL_MIN_h180_next30_DOM_1_7_14_21_28"
    return float(np.load(f"{d}/norm_stats_extremes_full_MIN.npz",allow_pickle=True)["cold_p10"])
def ens(pat, split, tk):
    fs = glob.glob(f"{pat}/preds_{split}{'_topk' if tk else ''}.csv")
    if not fs: return None
    return pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag",as_index=False).mean()
def fam_score(d, thr, low, fam):
    m = (d["true_m1_hot"]<=thr) if low else (d["true_m1_hot"]>=thr)
    if fam == "anc":
        ae = (d["pred_m1_hot"]-d["true_m1_hot"]).abs(); return (ae.mean()+ae[m].mean())/2
    se = [(d[f"pred_m1_{p}"]-d[f"true_m1_{p}"]).abs() for p in ["p3","p7","p15"]]
    return (np.mean([e.mean() for e in se]) + np.mean([e[m].mean() for e in se]))/2
def halves(d):
    d = d.sort_values("tag").reset_index(drop=True); h = len(d)//2
    return d.iloc[:h], d.iloc[h:]

INCUMBENT = {  # frozen picks
 ("HOT","Center","anc"):("opt3",1), ("HOT","Center","soft"):("opa3_opt",1),
 ("HOT","Negev","anc"):("ng_keda3",1), ("HOT","Negev","soft"):("ng3_opt",0),
 ("HOT","Northwest","anc"):("nw_opt",0), ("HOT","Northwest","soft"):("nw_ked",1),
 ("COLD","Center","anc"):("Ce_csp",0), ("COLD","Center","soft"):("Ce_csp",0),
 ("COLD","Negev","anc"):("Ne_csp",0), ("COLD","Negev","soft"):("Ne_csp",0),
 ("COLD","Northwest","anc"):("No_csp",0), ("COLD","Northwest","soft"):("No_csp",0),
}
CHALLENGERS = {
 ("HOT","Center"):   [("Ce_pb07",1),("Ce_pbo",1),("Ce_os3",1),("Ce_kt",1)],
 ("HOT","Negev"):    [("Ne_pb07",1),("Ne_pbo",1),("Ne_os3",1),("Ne_kt",1)],
 ("HOT","Northwest"):[("No_pb07",1),("No_pbo",1),("No_os3",1),("No_kt",1)],
 ("COLD","Center"):  [("Ce_cked",1),("Ce_ckeda3",1),("Ce_ca3",1),("Ce_cpb03",1),("Ce_ckt",1),("Ce_cos3",1)],
 ("COLD","Negev"):   [("Ne_cked",1),("Ne_ckeda3",1),("Ne_ca3",1),("Ne_cpb03",1),("Ne_ckt",1),("Ne_cos3",1)],
 ("COLD","Northwest"):[("No_cked",1),("No_ckeda3",1),("No_ca3",1),("No_cpb03",1),("No_ckt",1),("No_cos3",1)],
}
for season, low, root in [("HOT", False, "/home/weizyuv/expreal/{reg}_stable"),
                          ("COLD", True, "/home/weizyuv/expreal/{reg}_cold")]:
    for reg in ["Center","Negev","Northwest"]:
        thr = (cthr if low else hthr)(reg); R = root.format(reg=reg)
        for fam in ["anc","soft"]:
            inc_tag, inc_tk = INCUMBENT[(season,reg,fam)]
            vi = ens(f"{R}/{inc_tag}__*", "val", inc_tk)
            vi1, vi2 = halves(vi)
            s_inc = (fam_score(vi1,thr,low,fam), fam_score(vi2,thr,low,fam))
            admitted = []
            for tag, tk in CHALLENGERS[(season,reg)]:
                vc = ens(f"{R}/{tag}__*", "val", tk)
                if vc is None: continue
                v1, v2 = halves(vc)
                s = (fam_score(v1,thr,low,fam), fam_score(v2,thr,low,fam))
                if s[0] < s_inc[0] and s[1] < s_inc[1]:
                    admitted.append((tag, tk, s))
            if not admitted:
                print(f"{season} {reg:10} {fam}: incumbent {inc_tag} HOLDS (no challenger wins both halves)")
                continue
            # among admitted: best full-val score
            best = min(admitted, key=lambda a: fam_score(ens(f"{R}/{a[0]}__*","val",a[1]),thr,low,fam))
            ti = ens(f"{R}/{inc_tag}__*", "test", inc_tk); tc = ens(f"{R}/{best[0]}__*", "test", best[1])
            print(f"{season} {reg:10} {fam}: UPGRADE {inc_tag} -> {best[0]} | "
                  f"test {fam}-MeanMAE {fam_score(ti,thr,low,fam):.3f} -> {fam_score(tc,thr,low,fam):.3f}")
