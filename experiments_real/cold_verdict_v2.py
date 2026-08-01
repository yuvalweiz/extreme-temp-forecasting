"""COLD verdict v2 — protocol-symmetric with hot: member ensembles + top-k averaging,
selection on VALIDATION Mean-MAE only, test evaluated once per chosen member.

Members (pre-declared): csp5 (plain 5-seed), pool10 (csp+ctpk plain), ctpk_topk5,
cked_topk3 (KED frames), ckeda3_topk3 (KED+alpha3), blend(ctpk_topk,cked_topk).
Fair-tab members: cft5 (plain), cftpk_topk5. Article winter Tab-LSTM = published preds.
"""
import pandas as pd, numpy as np, glob

COLS = ["tag","true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15",
        "pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]
PC = [c for c in COLS if c.startswith("pred_")]

def cthr(reg):
    d=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/Winter Models/dataset_FULL_MIN_h180_next30_DOM_1_7_14_21_28"
    return float(np.load(f"{d}/norm_stats_extremes_full_MIN.npz",allow_pickle=True)["cold_p10"])

def ens(pats, split, topk=False):
    suf = "_topk" if topk else ""
    fs = []
    for p in pats: fs += glob.glob(f"{p}/preds_{split}{suf}.csv")
    if not fs: return None
    return pd.concat([pd.read_csv(f)[COLS] for f in fs]).groupby("tag",as_index=False).mean(), len(fs)

def blend(a, b):
    m = a.merge(b, on="tag", suffixes=("", "_b"))
    out = m[COLS].copy()
    for c in PC: out[c] = 0.5*m[c] + 0.5*m[f"{c}_b"]
    for c in ["true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15"]: out[c] = m[c]
    return out

def cells(d, thr):
    m = d["true_m1_hot"] <= thr
    ae = (d["pred_m1_hot"]-d["true_m1_hot"]).abs()
    se = [(d[f"pred_m1_{p}"]-d[f"true_m1_{p}"]).abs() for p in ["p3","p7","p15"]]
    return dict(anc_all=ae.mean(), anc_ext=ae[m].mean(),
                soft_all=np.mean([e.mean() for e in se]),
                soft_ext=np.mean([e[m].mean() for e in se]))

def article(reg):
    B=f"/home/weizyuv/Deep Learning Models/Cluster {reg}/Final Results Cluster {reg} Winter/TAB_LSTM_ISRAEL_WIDE_WINTER"
    am=pd.read_csv(glob.glob(f"{B}/preds_tab_lstm_main_israel_wide_WINTER_{reg}.csv")[0]).rename(
        columns={"true_m1_main":"true_m1_hot","pred_m1_main":"pred_m1_hot"})
    asf=pd.read_csv(glob.glob(f"{B}/preds_tab_lstm_soft_israel_wide_WINTER_{reg}.csv")[0]).drop(columns=["true_m1_main"])
    return am.merge(asf,on="tag")

total_art, total_fair = 0, 0
for reg, S in [("Center","Ce"),("Negev","Ne"),("Northwest","No")]:
    thr = cthr(reg); R=f"/home/weizyuv/expreal/{reg}_cold"
    members = {
        "csp5":       (lambda s: ens([f"{R}/{S}_csp__*"], s)),
        "pool10":     (lambda s: ens([f"{R}/{S}_csp__*", f"{R}/{S}_ctpk__*"], s)),
        "ctpk_topk":  (lambda s: ens([f"{R}/{S}_ctpk__*"], s, topk=True)),
        "cked_topk":  (lambda s: ens([f"{R}/{S}_cked__*"], s, topk=True)),
        "ckeda3_topk":(lambda s: ens([f"{R}/{S}_ckeda3__*"], s, topk=True)),
    }
    V, T = {}, {}
    for name, f in members.items():
        v, t = f("val"), f("test")
        if v and t: V[name], T[name] = v[0], t[0]
    if "ctpk_topk" in V and "cked_topk" in V:
        V["blend_tk"] = blend(V["ctpk_topk"], V["cked_topk"])
        T["blend_tk"] = blend(T["ctpk_topk"], T["cked_topk"])
    fair = {}
    for name, pats, tk in [("cft5",[f"{R}/{S}_cft__*"],False), ("cftpk_topk",[f"{R}/{S}_cftpk__*"],True)]:
        v, t = ens(pats,"val",tk), ens(pats,"test",tk)
        if v and t: fair[name] = (v[0], t[0])
    def val_score(d, fam):
        c = cells(d, thr)
        return (c["anc_all"]+c["anc_ext"])/2 if fam=="anc" else (c["soft_all"]+c["soft_ext"])/2
    pick = {fam: min(V, key=lambda n: val_score(V[n], fam)) for fam in ["anc","soft"]}
    fpick = {fam: min(fair, key=lambda n: val_score(fair[n][0], fam)) for fam in ["anc","soft"]}
    o_anc, o_soft = cells(T[pick["anc"]], thr), cells(T[pick["soft"]], thr)
    f_anc, f_soft = cells(fair[fpick["anc"]][1], thr), cells(fair[fpick["soft"]][1], thr)
    a = cells(article(reg), thr)
    print(f"=== {reg} (thr={thr:.2f})  picks: anc={pick['anc']} soft={pick['soft']} | fair: {fpick['anc']}/{fpick['soft']} ===")
    for cell, ours, fv in [("anc_all",o_anc,f_anc),("anc_ext",o_anc,f_anc),
                            ("soft_all",o_soft,f_soft),("soft_ext",o_soft,f_soft)]:
        vo, va, vf = ours[cell], a[cell], fv[cell]
        wa, wf = vo<va, vo<vf; total_art+=wa; total_fair+=wf
        print(f"  {cell:9} OURS {vo:.3f} | ART {va:.3f} {'WIN ' if wa else 'LOSS'} | FAIR {vf:.3f} {'WIN' if wf else 'LOSS'}")
print(f"\nTOTAL: {total_art}/12 vs ARTICLE winter tab | {total_fair}/12 vs fair tab (both sides val-selected)")
