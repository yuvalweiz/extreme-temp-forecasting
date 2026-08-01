"""Verify + tune the per-feature elevation-impact coefficients k_f jointly with gamma.

For each of the 9 channels: LOSO over 68 stations (train period ONLY, cutoff 2019-01-28),
grid k_scale x gamma; objective = mean per-station LOSO MAE. k_scale=1 recovering the
optimum = published k_f functionally verified; otherwise k*_f = k_f * best_scale.
"""
import os, sys, numpy as np, pandas as pd
os.environ["LOSO_END"] = "2019-01-28"          # leakage-free: train period only
sys.path.insert(0, "/home/weizyuv/article /repo/src/eval")
sys.path.insert(0, "/home/weizyuv/article /repo/src")
import loso_interpolation as L
from data.interpolation import weights_paper

K_SCALES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
GAMMAS   = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]

g = np.load(L.GRID, allow_pickle=True)
names = [str(x) for x in g["station_names"]]
slat, slon, sh = (g[k].astype(np.float64) for k in ("station_lat", "station_lon", "station_elev"))
kdf = pd.read_csv(L.KVALS)
features = [l.strip() for l in open(
    "/home/weizyuv/Deep Learning Models/Cluster Center/dataset_FULL_h180_next30_DOM_1_7_14_21_28/features_order.txt") if l.strip()]
agg = lambda f: "min" if f.startswith("min_") else ("max" if f.startswith("max_") else "mean")

print(f"k-tune LOSO | train-only (<= {os.environ['LOSO_END']}) | grid {len(K_SCALES)}x{len(GAMMAS)}")
summary = []
for feat in features:
    k_f = float(kdf.loc[kdf["Feature"] == feat].iloc[0]["k"])
    dates, V = L.load_station_series(names, feat, agg(feat))
    res = {}
    for ks in K_SCALES:
        for gm in GAMMAS:
            maes = []
            for j in range(len(names)):
                donors = np.arange(len(names)) != j
                has = np.isfinite(V[:, j])
                if has.sum() < 365: continue
                pix = np.array([[slon[j], slat[j], sh[j]]])
                w = weights_paper(pix, slat[donors], slon[donors], sh[donors],
                                  k_f * ks if ks > 0 else 0.0, gamma=gm, elev=(ks > 0))[0]
                pred = L.loso_pred(w, V[:, donors])
                m = has & np.isfinite(pred)
                if m.sum() >= 365:
                    maes.append(np.abs(pred[m] - V[m, j]).mean())
            res[(ks, gm)] = float(np.mean(maes))
    (bks, bgm), best = min(res.items(), key=lambda kv: kv[1])
    # published-k best gamma for comparison
    pub = {gm: res[(1.0, gm)] for gm in GAMMAS}
    pgm = min(pub, key=pub.get)
    noelev = min(res[(0.0, gm)] for gm in GAMMAS)
    print(f"{feat:16} pub-k best: g{pgm:<4} {pub[pgm]:.4f} | TUNED k x{bks:<4} g{bgm:<4} {best:.4f} "
          f"(d={pub[pgm]-best:+.4f}) | no-elev {noelev:.4f}")
    summary.append(dict(feature=feat, k_pub=k_f, best_scale=bks, k_tuned=k_f*bks,
                        gamma=bgm, mae_tuned=best, mae_pub=pub[pgm], mae_noelev=noelev))
pd.DataFrame(summary).to_csv("/home/weizyuv/expreal/ktune_results.csv", index=False)
print("\nsaved -> ~/expreal/ktune_results.csv")
