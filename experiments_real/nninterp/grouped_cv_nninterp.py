"""Station-grouped cross-validation for the learned interpolator: the honest
held-out LOSO. The deployed nninterp saw every station as a masked training
target, so its 1.044 LOSO is in-network. Here we split the 68 stations into
K=5 folds (seeded permutation); for fold k we train a fresh network with the
fold's stations FULLY EXCLUDED from training (never context, never query),
then reconstruct each fold station on test-period days from the full remaining
67-station context, exactly matching the kernel/KED LOSO protocol. Aggregate
MAE over all 68 stations = generalization-to-unseen-locations fidelity.
Identical hyperparameters, steps, eval subsampling (every 3rd test day) to
train_nninterp.py so the number is directly comparable.
"""
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train_nninterp import NNInterp, ST, M, dates, MU, SD, DEV

torch.manual_seed(111)
rng = np.random.default_rng(111)

train_idx = np.where(dates <= "2019-01-28")[0]
test_idx = np.where(dates > "2021-01-01")[0]

feats = [l.strip() for l in open("/home/weizyuv/Deep Learning Models/Cluster Center/"
         "dataset_FULL_h180_next30_DOM_1_7_14_21_28/features_order.txt") if l.strip()]
fmax = feats.index("max_dry_temp")

K = 5
perm = rng.permutation(68)
folds = np.array_split(perm, K)

all_errs = []
per_fold = []
for k, fold in enumerate(folds):
    keep = np.setdiff1d(np.arange(68), fold)        # training stations only
    STk = ST[keep]
    nk = len(keep)
    model = NNInterp().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    B, STEPS = 64, 6000
    for step in range(STEPS):
        di = np.random.choice(train_idx, B)
        fi = np.random.randint(0, 9, B)
        vals = (M[di, :, fi][:, keep] - MU[fi, None]) / SD[fi, None]
        nq = np.random.randint(8, 21)
        qsel = np.stack([np.random.permutation(nk)[:nq] for _ in range(B)])
        mask = np.zeros((B, nk), bool)
        np.put_along_axis(mask, qsel, True, 1)
        ctx = np.concatenate([np.broadcast_to(STk, (B, nk, 3)).copy(), vals[:, :, None]], -1)
        pred = model(torch.tensor(ctx, device=DEV),
                     torch.tensor(np.take_along_axis(np.broadcast_to(STk, (B, nk, 3)).copy(), qsel[:, :, None], 1), device=DEV),
                     torch.tensor(fi, device=DEV),
                     ctx_mask=torch.tensor(mask, device=DEV))
        tgt = torch.tensor(np.take_along_axis(vals, qsel, 1), device=DEV)
        loss = (pred - tgt).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"fold {k} step {step}: masked-MAE(norm) {loss.item():.4f}", flush=True)
    # eval: each fold station from the FULL remaining 67 (kernel-LOSO protocol)
    model.eval()
    ferrs = []
    with torch.no_grad():
        for di in test_idx[::3]:
            vals = (M[di, :, fmax] - MU[fmax]) / SD[fmax]
            for s in fold:
                keep67 = np.setdiff1d(np.arange(68), [s])
                ctx = np.concatenate([ST[keep67], vals[keep67, None]], -1)[None]
                q = ST[s][None, None, :]
                p = model(torch.tensor(ctx.astype(np.float32), device=DEV),
                          torch.tensor(q.astype(np.float32), device=DEV),
                          torch.tensor([fmax], device=DEV)).item()
                ferrs.append(abs(p - vals[s]) * SD[fmax])
    per_fold.append(np.mean(ferrs))
    all_errs.extend(ferrs)
    print(f"fold {k} ({len(fold)} stations): held-out LOSO MAE {np.mean(ferrs):.3f} C", flush=True)

print(f"\nSTATION-GROUPED held-out LOSO MAE (max_dry_temp, test period): {np.mean(all_errs):.3f} C")
print(f"per-fold: {np.round(per_fold,3)}")
print("references: in-network learned 1.044 | kernel 1.183 | KED 0.909 | IDW 1.693")
np.savez(os.path.join(os.path.dirname(__file__), "grouped_cv_results.npz"),
         errs=np.array(all_errs), per_fold=np.array(per_fold), folds=np.array(folds, dtype=object))
