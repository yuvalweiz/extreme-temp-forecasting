"""Learned interpolation model (author's idea): a light attention set-regressor that maps
{(coord, elev, value) of the day's stations} + a feature embedding to the value at any
query (coord, elev). One shared model for all 9 features.

Training: days <= 2019-01-28 ONLY (the forecaster's train cutoff — no leakage). Each
step samples (day, feature), masks a random subset of stations as queries, and regresses
their values from the remaining context stations (masked-station objective = LOSO-style).

Eval: leave-one-station-out on TEST-period days for max_dry_temp — directly comparable
to the paper's kernel LOSO numbers (exp kernel 1.183 °C, KED 0.909 °C).

Output: nninterp.pt + printed LOSO metrics. Frame-bank generation is a separate script.
"""
import os
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(111)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = np.load("/home/weizyuv/expreal/nninterp/daily_sv.npz", allow_pickle=True)
M, dates = D["M"], D["dates"].astype(str)          # (T,68,9)
MU, SD = D["feat_mu"], D["feat_sd"]
G = np.load("/home/weizyuv/article /repo/data/grid_metadata.npz", allow_pickle=True)
st_lat, st_lon, st_h = G["station_lat"], G["station_lon"], G["station_elev"]

LAT0, LAT1 = float(G["lat_grid"].min()), float(G["lat_grid"].max())
LON0, LON1 = float(G["lon_grid"].min()), float(G["lon_grid"].max())
def coords_feats(lat, lon, h):
    return np.stack([(lat - LAT0) / (LAT1 - LAT0), (lon - LON0) / (LON1 - LON0),
                     np.nan_to_num(h, nan=0.0) / 1000.0], -1).astype(np.float32)
ST = coords_feats(st_lat, st_lon, st_h)            # (68,3)

train_idx = np.where(dates <= "2019-01-28")[0]
test_idx = np.where(dates > "2021-01-01")[0]

class NNInterp(nn.Module):
    def __init__(self, d=96, heads=4, nfeat=9):
        super().__init__()
        self.femb = nn.Embedding(nfeat, d)
        self.ctx_in = nn.Linear(4, d)               # [latn, lonn, elevn, value_n]
        self.qry_in = nn.Linear(3, d)               # [latn, lonn, elevn]
        self.attn1 = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.attn2 = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, ctx, qry, f_id, ctx_mask=None):
        # ctx (B,S,4), qry (B,Q,3), f_id (B,); ctx_mask (B,S) True = EXCLUDE station
        fe = self.femb(f_id)[:, None, :]
        c = self.ctx_in(ctx) + fe
        q = self.qry_in(qry) + fe
        q = q + self.attn1(q, c, c, key_padding_mask=ctx_mask, need_weights=False)[0]
        q = q + self.ff1(q)
        q = q + self.attn2(q, c, c, key_padding_mask=ctx_mask, need_weights=False)[0]
        return self.out(q).squeeze(-1)              # (B,Q) normalized value

if __name__ == "__main__":
    model = NNInterp().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k | device {DEV}")
    
    B = 64
    STEPS = 6000
    for step in range(STEPS):
        di = np.random.choice(train_idx, B)
        fi = np.random.randint(0, 9, B)
        vals = (M[di, :, fi] - MU[fi, None]) / SD[fi, None]     # (B,68) normalized
        nq = np.random.randint(8, 21)
        qsel = np.stack([np.random.permutation(68)[:nq] for _ in range(B)])
        mask = np.zeros((B, 68), bool)
        np.put_along_axis(mask, qsel, True, 1)
        ctx = np.concatenate([np.broadcast_to(ST, (B, 68, 3)).copy(),
                              vals[:, :, None]], -1)
        ctx_t = torch.tensor(ctx, device=DEV)
        mask_t = torch.tensor(mask, device=DEV)                  # True = excluded from context
        qry_t = torch.tensor(np.take_along_axis(np.broadcast_to(ST, (B, 68, 3)).copy(), qsel[:, :, None], 1), device=DEV)
        tgt = torch.tensor(np.take_along_axis(vals, qsel, 1), device=DEV)
        pred = model(ctx_t, qry_t, torch.tensor(fi, device=DEV), ctx_mask=mask_t)
        loss = (pred - tgt).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"step {step}: train masked-MAE (norm) {loss.item():.4f}", flush=True)
    
    # LOSO eval on TEST-period days, max_dry_temp (feature index of 'max_dry_temp')
    feats = [l.strip() for l in open("/home/weizyuv/Deep Learning Models/Cluster Center/"
             "dataset_FULL_h180_next30_DOM_1_7_14_21_28/features_order.txt") if l.strip()]
    fmax = feats.index("max_dry_temp")
    model.eval()
    errs = []
    with torch.no_grad():
        for di in test_idx[::3]:                                 # every 3rd test day (speed)
            vals = (M[di, :, fmax] - MU[fmax]) / SD[fmax]
            ctx = np.concatenate([ST, vals[:, None]], -1)[None].repeat(68, 0)
            m = np.eye(68, dtype=bool)
            q = ST[:, None, :]
            p = model(torch.tensor(ctx, device=DEV), torch.tensor(q, device=DEV),
                      torch.tensor([fmax] * 68, device=DEV),
                      ctx_mask=torch.tensor(m, device=DEV)).squeeze(-1).cpu().numpy()
            errs.extend(np.abs(p - vals) * SD[fmax])
    print(f"\nLOSO test-period MAE (max_dry_temp): {np.mean(errs):.3f} C "
          f"(kernel reference 1.183, KED 0.909)")
    torch.save({"model": model.state_dict(), "LAT": (LAT0, LAT1), "LON": (LON0, LON1)},
               "/home/weizyuv/expreal/nninterp/nninterp.pt")
    print("saved nninterp.pt")
