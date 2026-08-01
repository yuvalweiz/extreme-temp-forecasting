"""
Non-spatial DL baselines (answers R4 "no transformer-based / recursive methods"):
feed the region's daily cluster-mean multivariate series (9 features x 180 days)
DIRECTLY to a temporal head (PatchTST / iTransformer / LSTM / TFT) and predict the
4 order statistics. Same targets, splits, loss, and metrics as the spatial model,
but NO ConvNeXt / NO spatial structure -> the clean non-spatial control.

env: REGION, HEAD (patchtst|itransformer|lstm|temporalfusion), K, ALPHA, BETA,
     EPOCHS, PATIENCE, SEED, CKPT_ROOT
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                                          # <repo>/src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "legacy", "train_reimpl"))                          # models.py
sys.path.insert(0, HERE)
import repo_paths as RP
import models as M
import series as S

REGION = os.environ.get("REGION", "Center")
HEAD = os.environ.get("HEAD", "patchtst")
K = float(os.environ.get("K", "1")); ALPHA = float(os.environ.get("ALPHA", "3")); BETA = float(os.environ.get("BETA", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "150")); PATIENCE = int(os.environ.get("PATIENCE", "30")); SEED = int(os.environ.get("SEED", "333"))
CKPT = os.environ.get("CKPT_ROOT", f"./tab_{REGION}_{HEAD}")
HISTORY = 180; OUT_W = (1.0, 0.5, 0.25, 0.125); EXTREME_Q = 0.90
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FEATS = open(RP.features_order_txt()).read().split()


class SeqDS(Dataset):
    def __init__(self, items, series, xm, xs, ym, yi):
        self.items, self.series = items, series
        self.xm, self.xs, self.ym, self.yi = xm, xs, ym, yi

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        days = pd.date_range(it["pred_point"] - pd.Timedelta(days=HISTORY - 1), it["pred_point"])
        X = self.series.reindex(days).to_numpy(np.float32)            # (180, 9)
        X = np.nan_to_num((X - self.xm) / (self.xs + 1e-8), nan=0.0)
        yr = np.asarray(it["y"], np.float32)
        yn = (yr - self.ym) / np.maximum(self.yi, 1e-6)
        return torch.from_numpy(X), torch.from_numpy(yn), torch.from_numpy(yr), it["tag"]


def main():
    os.makedirs(CKPT, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cd = S.build_cluster_daily(REGION)
    series = cd[FEATS].interpolate(limit_direction="both")
    df = S.build_targets(cd["max_dry_temp"], hottest=True)
    df["pred_point"] = pd.to_datetime(df["pred_point"])
    df["y"] = df[["true_m1_hot", "true_m1_p3", "true_m1_p7", "true_m1_p15"]].values.tolist()
    items = {s: df[df.split == s].to_dict("records") for s in ["train", "val", "test"]}
    tr_dates = [pd.date_range(it["pred_point"] - pd.Timedelta(days=HISTORY - 1), it["pred_point"]) for it in items["train"]]
    tr_idx = series.index.isin(pd.DatetimeIndex(np.concatenate([d.values for d in tr_dates])))
    xm, xs = series[tr_idx].mean().to_numpy(np.float32), series[tr_idx].std().to_numpy(np.float32)
    ys = np.array([it["y"] for it in items["train"]], np.float32)
    ym, yi = np.median(ys, 0).astype(np.float32), (np.quantile(ys, .75, 0) - np.quantile(ys, .25, 0)).astype(np.float32)
    train_hot = np.sort(ys[:, 0]); thr = float(np.quantile(ys[:, 0], EXTREME_Q))
    print(f"[tab/{REGION}/{HEAD}] train/val/test={len(items['train'])}/{len(items['val'])}/{len(items['test'])} p90={thr:.2f}")

    def dl(s, sh): return DataLoader(SeqDS(items[s], series, xm, xs, ym, yi), batch_size=16, shuffle=sh)
    tr, va, te = dl("train", True), dl("val", False), dl("test", False)

    head = {"patchtst": M.PatchTSTHead(len(FEATS)), "itransformer": M.iTransformerHead(len(FEATS), HISTORY),
            "lstm": M.LSTMHead(len(FEATS)), "temporalfusion": M.TemporalFusionLiteHead(len(FEATS))}[HEAD]
    model = torch.nn.Sequential(); model.head = head; model.aff = M.OutputAffine(4)

    class Net(torch.nn.Module):
        def __init__(s): super().__init__(); s.head = head; s.aff = M.OutputAffine(4)
        def forward(s, x): return s.aff(s.head(x))
    net = Net().to(DEV)
    ymed = torch.tensor(ym, device=DEV); yiqr = torch.tensor(yi, device=DEV)
    base = M.PercentileWeightedMAE(ym, yi, "none", OUT_W, train_hot, k=K, alpha=ALPHA).to(DEV)
    loss_fn = M.PercentileWeightedHinge(base, beta_hot=BETA).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    def preds(loader):
        net.eval(); rows = []
        with torch.no_grad():
            for X, yn, yr, tags in loader:
                pc = M.denorm_y_torch(net(X.to(DEV)).float(), ymed, yiqr).cpu().numpy()
                for i in range(pc.shape[0]):
                    rows.append({"tag": tags[i], **{f"true_m1_{k}": float(yr[i, j]) for j, k in enumerate(["hot", "p3", "p7", "p15"])},
                                 **{f"pred_m1_{k}": float(pc[i, j]) for j, k in enumerate(["hot", "p3", "p7", "p15"])}})
        return pd.DataFrame(rows)

    def met(d):
        e = (d.pred_m1_hot - d.true_m1_hot).abs(); ext = d.true_m1_hot >= thr
        return float(e.mean()), float(e[ext].mean()) if ext.any() else float("nan")
    best, bep, bstate, bad, ema = 1e9, -1, None, 0, None
    for ep in range(1, EPOCHS + 1):
        net.train()
        for X, yn, yr, tg in tr:
            opt.zero_grad(); loss = loss_fn(net(X.to(DEV)).float(), yr.to(DEV).float()); loss.backward(); opt.step()
        sch.step()
        va_a, va_e = met(preds(va)); raw = va_a if np.isnan(va_e) else 0.5 * (va_a + va_e)
        ema = raw if ema is None else 0.35 * raw + 0.65 * ema
        if ema < best - 1e-6: best, bep, bstate, bad = ema, ep, {k: v.cpu() for k, v in net.state_dict().items()}, 0
        else: bad += 1
        if bad >= PATIENCE: break
    if bstate: net.load_state_dict(bstate)
    dft, dfv = preds(te), preds(va); ta, tee = met(dft); va_a, va_e = met(dfv)
    dft.to_csv(os.path.join(CKPT, "preds_test.csv"), index=False)
    json.dump({"region": REGION, "head": HEAD, "k": K, "alpha": ALPHA, "beta": BETA, "best_ep": bep,
               "val_mae_all": va_a, "val_mae_ext": va_e, "test_mae_all": ta, "test_mae_ext": tee},
              open(os.path.join(CKPT, "meta.json"), "w"), indent=2)
    print(f"[DONE tab {HEAD}] TEST all={ta:.3f} ext={tee:.3f} (val {va_a:.3f}/{va_e:.3f})")


if __name__ == "__main__":
    main()
