# ============================================================
# FULL SCRIPT — Percentile Weighted MAE + Alpha for underprediction
# CELL 1: LOADER
# ============================================================

import os, zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# -------------------------
# CONFIG
# -------------------------
# Parametrized via env so the same script serves all regions.
# sbatch sets --chdir to the cluster root (parent of the dataset dir), so the
# relative sample paths inside the split CSVs resolve from cwd.
save_dir  = os.environ.get("DATASET_DIR", r"./dataset_FULL_h180_next30_DOM_1_7_14_21_28")
TAG = "yNONE_v1"
norm_path = os.path.join(save_dir, f"norm_stats_extremes_full_{TAG}.npz")
train_csv = os.path.join(save_dir, f"split_train_{TAG}.csv")
val_csv   = os.path.join(save_dir, f"split_val_{TAG}.csv")
test_csv  = os.path.join(save_dir, f"split_test_{TAG}.csv")

KEEP_DOM    = {1, 7, 14, 21, 28}
BATCH_SIZE  = 4
NUM_WORKERS = 4
PIN_MEMORY  = True
EPS         = 1e-8
EXTREME_Q   = 0.90    # used for EVAL + SELECTION only — NOT in loss

for p in [norm_path, train_csv, val_csv, test_csv]:
    if not os.path.exists(p):
        raise FileNotFoundError(p)

# -------------------------
# Normalization stats
# -------------------------
ns = np.load(norm_path, allow_pickle=True)
x_mean      = ns["x_mean"].astype(np.float32)
x_std       = ns["x_std"].astype(np.float32)
y_median    = ns["y_median"].astype(np.float32)
y_iqr       = ns["y_iqr"].astype(np.float32)
y_transform = str(ns["y_transform"][0])
T = int(ns["T"]); C = int(ns["C"]); H = int(ns["H"]); W = int(ns["W"])
print(f"[Info] X: T={T} C={C} HxW={H}x{W} | y_transform={y_transform}")

# -------------------------
# Splits
# -------------------------
def load_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["tag"] = pd.to_datetime(df["tag"])
    df = df[df["tag"].dt.day.isin(KEEP_DOM)].copy()
    return df.sort_values("tag").reset_index(drop=True)

df_train = load_split(train_csv)
df_val   = load_split(val_csv)
df_test  = load_split(test_csv)
print(f"[Filtered DOM] train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")
if len(df_train) == 0:
    raise RuntimeError("Train empty after DOM filter.")

# -------------------------
# y normalize
# -------------------------
def normalize_y(y_raw: np.ndarray) -> np.ndarray:
    y0 = (y_raw.astype(np.float32) - y_median) / np.maximum(y_iqr, 1e-6)
    if y_transform == "asinh":
        y0 = np.arcsinh(y0).astype(np.float32)
    return y0.astype(np.float32)

# -------------------------
# Dataset
# -------------------------
class FullSampleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        good = []; bad = 0
        for row in df.itertuples(index=False):
            try:
                with np.load(row.path, allow_pickle=True) as z:
                    if "X" in z.files and "y" in z.files:
                        good.append((row.path, row.tag))
                    else:
                        bad += 1
            except (zipfile.BadZipFile, OSError, EOFError, ValueError):
                bad += 1
        self.df = pd.DataFrame(good, columns=["path","tag"]).sort_values("tag").reset_index(drop=True)
        if bad > 0: print(f"[Dataset] skipped {bad} corrupted files")
        if len(self.df) == 0: raise RuntimeError("Dataset empty.")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with np.load(row["path"], allow_pickle=True) as z:
            X     = z["X"].astype(np.float32)
            y_raw = z["y"].astype(np.float32)
        if X.shape[0] != T or X.shape[1] != C:
            raise RuntimeError(f"Bad shape {X.shape} in {os.path.basename(row['path'])}")
        if os.environ.get("SPATIAL", "full") == "mean":  # flatten spatial structure (spatial-use test)
            X = np.broadcast_to(X.mean(axis=(2, 3), keepdims=True), X.shape).copy()
        X = (X - x_mean.reshape(1,C,1,1)) / (x_std.reshape(1,C,1,1) + EPS)
        return (torch.from_numpy(X),
                torch.from_numpy(normalize_y(y_raw)),
                torch.from_numpy(y_raw),
                row["tag"].strftime("%Y-%m-%d"))

# -------------------------
# Build datasets
# -------------------------
train_ds = FullSampleDataset(df_train)
val_ds   = FullSampleDataset(df_val)
test_ds  = FullSampleDataset(df_test)
print(f"[After filter] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

# -------------------------
# Train hot distribution — needed for ECDF in loss
# -------------------------
train_hot_raw = np.array([
    float(np.load(p, allow_pickle=True)["y"].astype(np.float32)[0])
    for p in train_ds.df["path"]
], dtype=np.float32)

train_hot_sorted = np.sort(train_hot_raw)
hot_thr_eval     = float(np.quantile(train_hot_raw, EXTREME_Q))   # p90 — eval + selection
hot_thr_p50      = float(np.quantile(train_hot_raw, 0.50))        # floor for loss
hot_p95          = hot_thr_eval   # alias kept for downstream compatibility

print(f"[Thresholds] train_p50={hot_thr_p50:.3f} | train_p90(eval/select)={hot_thr_eval:.3f}")

# -------------------------
# DataLoaders
# -------------------------
train_loader_plain = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS>0))
val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS>0))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS>0))

print(f"[Loaders] train={len(train_loader_plain)} val={len(val_loader)} test={len(test_loader)} batches")
Xb, yb, yrawb, tagsb = next(iter(train_loader_plain))
print(f"[Sanity] batch={Xb.shape} | yraw_hot=({yrawb[:,0].min():.2f},{yrawb[:,0].max():.2f})")


# ============================================================
# CELL 2) MODEL + LOSS
# ============================================================

import torch.nn as nn
try:
    import timm
except Exception as e:
    raise RuntimeError("timm required.") from e


def denorm_y_torch(y_norm, y_median, y_iqr, y_transform="none"):
    y0 = torch.sinh(y_norm) if str(y_transform) == "asinh" else y_norm
    return y0 * y_iqr + y_median


# ============================================================
# LOSS: Percentile weight × alpha for underprediction on hot samples
#
# For each sample:
#   r   = ECDF rank of true_hot in training distribution
#   w   = 1.0              if r < 0.5   (floor — below median)
#         exp(k*(r-0.5))   if r >= 0.5  (exponential — above median)
#   hot = r >= 0.5   (same boundary as weight floor)
#
# Per output error:
#   loss = w * alpha * |err|   if underprediction AND hot sample
#   loss = w * |err|           otherwise
#
# Two mechanisms, single design:
#   k     → how much hotter samples matter overall
#   alpha → additional directional push against missing peaks
# ============================================================
class PercentileWeightedMAE(nn.Module):
    def __init__(self, y_median, y_iqr, y_transform, out_weights,
                 train_hot_sorted, k=2.0, alpha=2.0):
        """
        k:     steepness of percentile weight above median
        alpha: underprediction multiplier for hot samples (r >= 0.5)
               alpha=1.0 means no extra penalty for underprediction
        """
        super().__init__()
        self.register_buffer("y_median",     torch.as_tensor(y_median,         dtype=torch.float32))
        self.register_buffer("y_iqr",        torch.as_tensor(y_iqr,            dtype=torch.float32))
        self.register_buffer("w_out",        torch.tensor(out_weights,         dtype=torch.float32))
        self.register_buffer("train_sorted", torch.as_tensor(train_hot_sorted, dtype=torch.float32))
        self.y_transform = str(y_transform)
        self.k     = float(k)
        self.alpha = float(alpha)

    def get_sample_weights(self, true_hot):
        """true_hot: (B,) → sample_weight (B,), is_hot (B,) bool"""
        idx      = torch.searchsorted(self.train_sorted, true_hot.contiguous())
        ranks    = idx.float() / float(len(self.train_sorted))
        is_hot   = ranks >= 0.5
        w        = torch.where(is_hot,
                               torch.exp(self.k * (ranks - 0.5)),
                               torch.ones_like(ranks))
        return w, is_hot

    def forward(self, pred_norm, true_raw_c):
        pred_c   = denorm_y_torch(pred_norm, self.y_median, self.y_iqr, self.y_transform)
        true_hot = true_raw_c[:, 0]
        sample_w, is_hot = self.get_sample_weights(true_hot)   # (B,), (B,)

        err     = pred_c - true_raw_c                          # (B,4)
        abs_err = err.abs()                                    # (B,4)

        # alpha applies to underpredictions on hot samples only
        is_under_hot = (err < 0) & is_hot.unsqueeze(1)        # (B,4)
        penalized    = torch.where(is_under_hot,
                                   abs_err * self.alpha,
                                   abs_err)                    # (B,4)

        per_sample = (penalized * self.w_out.to(penalized.device)).mean(dim=1)  # (B,)
        return (per_sample * sample_w).mean()


class PercentileWeightedHinge(nn.Module):
    """
    Optional additive hinge on output 0 underprediction,
    weighted by the same percentile weight.
    beta_hot=0 → pure base loss.
    """
    def __init__(self, base_loss, beta_hot=1.0):
        super().__init__()
        self.base     = base_loss
        self.beta_hot = float(beta_hot)

    def forward(self, pred_norm, true_raw_c):
        base = self.base(pred_norm, true_raw_c)
        if self.beta_hot == 0.0:
            return base
        pred_c   = denorm_y_torch(pred_norm, self.base.y_median, self.base.y_iqr, self.base.y_transform)
        true_hot = true_raw_c[:, 0]
        sample_w, _ = self.base.get_sample_weights(true_hot)
        extra    = (torch.relu(true_hot - pred_c[:, 0]) * sample_w).mean()
        return base + self.beta_hot * extra


# ============================================================
# BACKBONE — unchanged
# ============================================================
class ConvNeXtTinyBackbone(nn.Module):
    def __init__(self, in_chans, pretrained=True):
        super().__init__()
        self.m = timm.create_model("convnext_tiny", pretrained=pretrained,
                                   in_chans=in_chans, num_classes=0, global_pool="avg")
        self.out_dim = int(getattr(self.m, "num_features", 0)) or -1
        if self.out_dim <= 0: raise RuntimeError("Cannot infer out_dim.")
    def forward(self, x): return self.m(x)


def backbone_chunked_forward(backbone, x2d, chunk):
    if chunk <= 0 or x2d.shape[0] <= chunk:
        return backbone(x2d)
    return torch.cat([backbone(x2d[i:i+chunk]) for i in range(0, x2d.shape[0], chunk)])


# ============================================================
# HEADS — unchanged from original
# ============================================================
class PatchTSTHead(nn.Module):
    def __init__(self, d_in, d_model=256, nhead=8, layers=4, patch_len=30, stride=30, dropout=0.1, out_dim=4):
        super().__init__()
        self.patch_len = int(patch_len); self.stride = int(stride)
        self.proj = nn.Linear(d_in * self.patch_len, d_model)
        self.enc  = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True), num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model,d_model),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model,out_dim))
    def forward(self, seq):
        B, T, F = seq.shape
        patches = [seq[:,s:s+self.patch_len,:].reshape(B,-1)
                   for s in range(0, T-self.patch_len+1, self.stride)]
        if not patches:
            seq = seq[:,-self.patch_len:,:] if T>=self.patch_len else torch.cat([seq.new_zeros(B,self.patch_len-T,F),seq],1)
            patches = [seq.reshape(B,-1)]
        x = self.proj(torch.stack(patches,1))
        return self.head(self.enc(x)[:,-1,:])


class iTransformerHead(nn.Module):
    def __init__(self, d_in, T, d_model=256, nhead=8, layers=2, dropout=0.1, out_dim=4):
        super().__init__()
        self.T = int(T)
        self.time_proj = nn.Linear(self.T, d_model)
        self.enc  = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True), num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model,d_model),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model,out_dim))
    def forward(self, seq):
        B, T, F = seq.shape
        if T < self.T: seq = torch.cat([seq.new_zeros(B,self.T-T,F),seq],1)
        elif T > self.T: seq = seq[:,-self.T:,:]
        return self.head(self.enc(self.time_proj(seq.transpose(1,2))).mean(1))


class GLU(nn.Module):
    def __init__(self, d):
        super().__init__(); self.fc = nn.Linear(d,2*d)
    def forward(self, x):
        a, b = self.fc(x).chunk(2,dim=-1); return a * torch.sigmoid(b)


class TemporalFusionLiteHead(nn.Module):
    def __init__(self, d_in, d_model=256, nhead=8, layers=2, dropout=0.1, out_dim=4):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.glu     = GLU(d_model)
        self.norm1   = nn.LayerNorm(d_model)
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
            for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model,d_model),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model,out_dim))
    def forward(self, seq):
        x = self.in_proj(seq); x = x + self.glu(x); x = self.norm1(x)
        for layer in self.attn_layers: x = layer(x)
        return self.head(x[:,-1,:])


class LSTMHead(nn.Module):
    def __init__(self, d_in, d_model=256, layers=2, dropout=0.1, out_dim=4, bidirectional=False):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(d_in, d_model, num_layers=int(layers), batch_first=True,
            dropout=float(dropout) if int(layers)>1 else 0.0, bidirectional=self.bidirectional)
        hdim = d_model*(2 if self.bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim,hdim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hdim,out_dim))
    def forward(self, seq):
        _, (h_n, _) = self.lstm(seq)
        h = torch.cat([h_n[-2],h_n[-1]],1) if self.bidirectional else h_n[-1]
        return self.head(h)


class OutputAffine(nn.Module):
    def __init__(self, out_dim=4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(out_dim))
        self.bias  = nn.Parameter(torch.zeros(out_dim))
    def forward(self, y): return y * self.scale + self.bias


class ConvNeXtTiny_WithHead(nn.Module):
    def __init__(self, in_chans, T, head_type, backbone_chunk=256, pretrained=True,
                 d_model=256, nhead=8, layers=4, dropout=0.1, use_out_affine=True,
                 lstm_bidirectional=False):
        super().__init__()
        self.backbone       = ConvNeXtTinyBackbone(in_chans, pretrained)
        self.backbone_chunk = int(backbone_chunk)
        F  = self.backbone.out_dim
        ht = str(head_type).lower()
        if   ht == "patchtst":      self.head = PatchTSTHead(F, d_model, nhead, layers, 30, 30, dropout, 4)
        elif ht == "itransformer":  self.head = iTransformerHead(F, T, d_model, nhead, layers, dropout, 4)
        elif ht == "temporalfusion":self.head = TemporalFusionLiteHead(F, d_model, nhead, max(2,layers//2), dropout, 4)
        elif ht == "lstm":          self.head = LSTMHead(F, d_model, max(1,layers//2), dropout, 4, lstm_bidirectional)
        else: raise ValueError(f"Unknown head_type: {head_type}")
        self.out_affine = OutputAffine(4) if bool(use_out_affine) else nn.Identity()

    def forward(self, X):
        B, T, C, H, W = X.shape
        feats = backbone_chunked_forward(
            self.backbone, X.reshape(B*T,C,H,W), self.backbone_chunk).reshape(B,T,-1)
        return self.out_affine(self.head(feats))


# ============================================================
# CELL 3) GRID — Percentile-Weighted MAE + Alpha underprediction
#   Sweep: head_type × k × alpha × beta_hot
#   Selection: avg(val_mae_hot_all, val_mae_hot_ext) with EMA
#   out_weights: (1, 0.5, 0.25, 0.125) — geometric decay, fixed
# ============================================================

import os, time, json
import numpy as np
import pandas as pd
import torch

torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[Device]", DEVICE)

# -------------------------
# Fixed across all runs
# -------------------------
OUT_WEIGHTS     = (1.0, 0.5, 0.25, 0.125)  # geometric decay
D_MODEL         = 256
NHEAD           = 4
LAYERS          = 2
DROPOUT         = 0.1
BACKBONE_CHUNK  = 256
LR_BACKBONE     = 1e-5
LR_HEAD         = 1e-4
WD_BACKBONE     = 1e-3   # split decay
WD_HEAD         = 1e-4
GRAD_CLIP       = 1.0
USE_AMP         = True
EMA_ALPHA_SCORE = 0.35

y_median_np  = ns["y_median"].astype(np.float32)
y_iqr_np     = ns["y_iqr"].astype(np.float32)
T_from_stats = int(ns["T"])

AMP_ENABLED = bool(USE_AMP and DEVICE == "cuda")
y_median_t  = torch.as_tensor(y_median_np, device=DEVICE, dtype=torch.float32)
y_iqr_t     = torch.as_tensor(y_iqr_np,   device=DEVICE, dtype=torch.float32)


# -------------------------
# Eval helpers
# -------------------------
@torch.no_grad()
def collect_pred_table(model, loader):
    model.eval()
    rows = []
    for X, y_norm, y_raw, tags in loader:
        X     = X.to(DEVICE, non_blocking=True).float()
        y_raw = y_raw.to(DEVICE, non_blocking=True).float()
        with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
            pred_norm = model(X)
        pred_c = denorm_y_torch(pred_norm.float(), y_median_t, y_iqr_t, y_transform).cpu().numpy()
        true_c = y_raw.cpu().numpy()
        for i in range(pred_c.shape[0]):
            rows.append({
                "tag":         tags[i],
                "true_m1_hot": float(true_c[i,0]),
                "true_m1_p3":  float(true_c[i,1]),
                "true_m1_p7":  float(true_c[i,2]),
                "true_m1_p15": float(true_c[i,3]),
                "pred_m1_hot": float(pred_c[i,0]),
                "pred_m1_p3":  float(pred_c[i,1]),
                "pred_m1_p7":  float(pred_c[i,2]),
                "pred_m1_p15": float(pred_c[i,3]),
            })
    return pd.DataFrame(rows).sort_values("tag").reset_index(drop=True)


def compute_hot_metrics(df, hot_thr):
    """Output 0 only — for selection and reporting."""
    mae_all = float((df["pred_m1_hot"] - df["true_m1_hot"]).abs().mean())
    ext = df[df["true_m1_hot"] >= float(hot_thr)]
    n_ext = len(ext)
    if n_ext > 0:
        mae_ext        = float((ext["pred_m1_hot"] - ext["true_m1_hot"]).abs().mean())
        under_mae_ext  = float(np.maximum(0, ext["true_m1_hot"].values - ext["pred_m1_hot"].values).mean())
        under_rate_ext = float((ext["pred_m1_hot"] < ext["true_m1_hot"]).mean())
    else:
        mae_ext = under_mae_ext = under_rate_ext = float("nan")
    return {"n_all": len(df), "n_ext": n_ext,
            "mae_hot_all":    mae_all,
            "mae_hot_ext":    mae_ext,
            "under_mae_ext":  under_mae_ext,
            "under_rate_ext": under_rate_ext}


def _arrs(df):
    t = df[["true_m1_hot","true_m1_p3","true_m1_p7","true_m1_p15"]].to_numpy(np.float32)
    p = df[["pred_m1_hot","pred_m1_p3","pred_m1_p7","pred_m1_p15"]].to_numpy(np.float32)
    return t, p


def suite(df, hot_thr):
    """Full diagnostics for logging/CSV."""
    m = compute_hot_metrics(df, hot_thr)
    t, p = _arrs(df)
    mae_out = np.mean(np.abs(p - t), axis=0)
    return {**m,
            "mae_out_all": mae_out.tolist(),
            "mae_mean_all": float(mae_out.mean())}


# -------------------------
# SELECTION METRIC
# avg(val_mae_hot_all, val_mae_hot_ext) — output 0 only, EMA smoothed
# Paper: "Model selected by average of overall and extreme-day MAE
#         on the hottest-day output over the validation set."
# -------------------------
def select_score(mae_hot_all, mae_hot_ext, n_ext):
    if n_ext == 0 or not np.isfinite(mae_hot_ext):
        return float(mae_hot_all)
    return float((mae_hot_all + mae_hot_ext) / 2.0)


# -------------------------
# Single run
# -------------------------
def run_one_config(
    head_type, k, alpha, beta_hot, TAG,
    epochs=120, patience=10, seed=333,
    ckpt_root="./grid_ckpts_pct_weighted",
):
    head_type = str(head_type).lower()
    assert head_type in ["temporalfusion","lstm","patchtst","itransformer"], f"Unknown head: {head_type}"
    os.makedirs(ckpt_root, exist_ok=True)

    torch.manual_seed(seed); np.random.seed(seed)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    X0, *_ = next(iter(train_loader_plain))
    in_chans = int(X0.shape[2])

    run_id  = (f"{TAG}__{head_type}"
               f"__k{k:g}__a{alpha:g}__bh{beta_hot:g}__seed{seed}")
    run_dir = os.path.join(ckpt_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    epoch_log_path = os.path.join(run_dir, "epoch_log.csv")

    # Print weight table for this k
    print(f"\n[k={k}] sample weights at key percentiles:")
    for q, r in [(0.50,0.0),(0.75,0.25),(0.90,0.40),(0.95,0.45),(0.99,0.49)]:
        w = 1.0 if r == 0.0 else float(np.exp(k * r))
        print(f"  p{int(q*100):02d}: w={w:.3f}  (alpha on underpredict: {w*alpha:.3f})")

    model = ConvNeXtTiny_WithHead(
        in_chans=in_chans, T=T_from_stats, head_type=head_type,
        backbone_chunk=BACKBONE_CHUNK, pretrained=True,
        d_model=D_MODEL, nhead=NHEAD, layers=LAYERS, dropout=DROPOUT,
        use_out_affine=True,
    ).to(DEVICE)

    if os.environ.get("FREEZE", "0") == "1":  # frozen-backbone ablation: does the CNN learn task-specific filters?
        for p in model.backbone.parameters():
            p.requires_grad = False

    base_loss = PercentileWeightedMAE(
        y_median=y_median_np, y_iqr=y_iqr_np, y_transform=y_transform,
        out_weights=OUT_WEIGHTS,
        train_hot_sorted=train_hot_sorted,
        k=float(k), alpha=float(alpha),
    ).to(DEVICE)

    loss_fn = PercentileWeightedHinge(base_loss, beta_hot=float(beta_hot)).to(DEVICE)

    # Split weight decay
    bb_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (bb_params if name.startswith("backbone.") else head_params).append(p)

    opt = torch.optim.AdamW([
        {"params": bb_params,   "lr": LR_BACKBONE, "weight_decay": WD_BACKBONE},
        {"params": head_params, "lr": LR_HEAD,     "weight_decay": WD_HEAD},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_val   = float("inf")
    best_ep    = -1
    best_state = None
    bad        = 0
    ema_score  = None
    epoch_rows = []
    t0 = time.time()

    for ep in range(1, epochs + 1):

        # ---- TRAIN ----
        model.train()
        run_loss = 0.; n = 0
        for X, y_norm, y_raw, tags in train_loader_plain:
            X     = X.to(DEVICE, non_blocking=True).float()
            y_raw = y_raw.to(DEVICE, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                pred = model(X)
                loss = loss_fn(pred.float(), y_raw)
            scaler.scale(loss).backward()
            if GRAD_CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
            run_loss += float(loss.item()) * X.shape[0]
            n        += X.shape[0]
        tr_loss = run_loss / max(1, n)
        scheduler.step()

        # ---- EVAL ----
        df_val_ep = collect_pred_table(model, val_loader)
        m         = compute_hot_metrics(df_val_ep, hot_thr_eval)
        raw_score = select_score(m["mae_hot_all"], m["mae_hot_ext"], m["n_ext"])
        ema_score = (EMA_ALPHA_SCORE * raw_score + (1.0 - EMA_ALPHA_SCORE) * ema_score
                     if ema_score is not None else raw_score)

        improved = ema_score < (best_val - 1e-6)
        if improved:
            best_val   = ema_score
            best_ep    = ep
            best_state = {kk: v.detach().cpu() for kk, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        # Per-epoch CSV (live monitoring via tail -f)
        epoch_rows.append({
            "ep": ep, "tr_loss": tr_loss,
            "val_mae_hot_all":    m["mae_hot_all"],
            "val_mae_hot_ext":    m["mae_hot_ext"],
            "val_under_mae_ext":  m["under_mae_ext"],
            "val_under_rate_ext": m["under_rate_ext"],
            "val_n_ext": m["n_ext"],
            "raw_score": raw_score, "ema_score": ema_score,
            "best_val": best_val, "best_ep": best_ep, "bad": bad,
        })
        pd.DataFrame(epoch_rows).to_csv(epoch_log_path, index=False)

        print(
            f"[{head_type}|k={k}|a={alpha}|bh={beta_hot}] "
            f"ep{ep:03d} tr={tr_loss:.4f} | "
            f"mae_all={m['mae_hot_all']:.3f} mae_ext={m['mae_hot_ext']:.3f} "
            f"underRate={m['under_rate_ext']:.3f} n_ext={m['n_ext']} | "
            f"sel_ema={ema_score:.4f} best={best_val:.4f}@ep{best_ep} bad={bad}/{patience}"
        )

        if bad >= patience:
            print(f"[EarlyStop] {run_id} at ep{ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed_min = (time.time() - t0) / 60.

    # Final metrics on best checkpoint
    df_val_f  = collect_pred_table(model, val_loader)
    df_test_f = collect_pred_table(model, test_loader)
    val_m     = compute_hot_metrics(df_val_f,  hot_thr_eval)
    test_m    = compute_hot_metrics(df_test_f, hot_thr_eval)
    val_suite = suite(df_val_f,  hot_thr_eval)
    tst_suite = suite(df_test_f, hot_thr_eval)

    # Disk-light: always persist the (tiny) per-config predictions for
    # downstream significance testing / plots; only persist the heavy model
    # checkpoint when explicitly requested (SAVE_CKPT=1).
    df_val_f.to_csv(os.path.join(run_dir, "preds_val.csv"), index=False)
    df_test_f.to_csv(os.path.join(run_dir, "preds_test.csv"), index=False)

    ckpt_path = os.path.join(run_dir, "best.pt")
    meta = {
        "run_id": run_id,
        "head_type": head_type,
        "k": float(k), "alpha": float(alpha), "beta_hot": float(beta_hot), "seed": int(seed),
        "best_ep": int(best_ep),
        "loss_design": {
            "type": "PercentileWeightedMAE + PercentileWeightedHinge",
            "w(r)": "1.0 if r<0.5 else exp(k*(r-0.5))",
            "alpha": "multiplier for underpredictions on hot samples (r>=0.5)",
            "out_weights": list(OUT_WEIGHTS),
        },
        "selection": {
            "metric": "avg(val_mae_hot_all, val_mae_hot_ext) — output 0 only",
            "ext_def": f"true_hot >= train_p{int(EXTREME_Q*100)} = {hot_thr_eval:.3f}",
            "EMA_ALPHA": EMA_ALPHA_SCORE,
        },
        "val":  {k2: (v if not isinstance(v,list) else v) for k2,v in val_suite.items()},
        "test": {k2: (v if not isinstance(v,list) else v) for k2,v in tst_suite.items()},
        "elapsed_min": elapsed_min,
    }
    if os.environ.get("SAVE_CKPT", "0") == "1":
        torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt_path)
    else:
        ckpt_path = ""  # skipped to keep disk footprint minimal during sweeps
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[DONE] {run_id} | best_ema={best_val:.4f}@ep{best_ep} | "
        f"VAL mae_all={val_m['mae_hot_all']:.3f} mae_ext={val_m['mae_hot_ext']:.3f} "
        f"underRate={val_m['under_rate_ext']:.3f} | "
        f"TEST mae_all={test_m['mae_hot_all']:.3f} mae_ext={test_m['mae_hot_ext']:.3f} "
        f"underRate={test_m['under_rate_ext']:.3f} | {elapsed_min:.1f}m"
    )

    return {
        "run_id": run_id, "head_type": head_type,
        "k": float(k), "alpha": float(alpha), "beta_hot": float(beta_hot),
        "seed": int(seed), "best_ep": int(best_ep),
        "val_mae_hot_all":    val_m["mae_hot_all"],
        "val_mae_hot_ext":    val_m["mae_hot_ext"],
        "val_under_mae_ext":  val_m["under_mae_ext"],
        "val_under_rate_ext": val_m["under_rate_ext"],
        "val_n_ext":          val_m["n_ext"],
        "test_mae_hot_all":   test_m["mae_hot_all"],
        "test_mae_hot_ext":   test_m["mae_hot_ext"],
        "test_under_mae_ext": test_m["under_mae_ext"],
        "test_under_rate_ext":test_m["under_rate_ext"],
        "test_n_ext":         test_m["n_ext"],
        "ckpt_path": ckpt_path, "run_dir": run_dir,
    }


# -------------------------
# Grid runner
# -------------------------
def run_big_grid(
    TAG="GRID_EXP",
    head_types=("temporalfusion","lstm"),
    ks=(1.0, 2.0, 3.0),
    alphas=(1.0, 2.0, 3.0),
    beta_hots=(0.0, 1.0, 1.5),
    patience=10,
    epochs=120,
    seed=333,
    ckpt_root="./grid_ckpts_pct_weighted",
):
    os.makedirs(ckpt_root, exist_ok=True)
    results  = []
    csv_path = os.path.join(ckpt_root, "grid_results.csv")

    for head in head_types:
        for k in ks:
            for a in alphas:
                for bh in beta_hots:
                    res = run_one_config(
                        head_type=head, k=float(k), alpha=float(a),
                        beta_hot=float(bh), TAG=str(TAG),
                        epochs=int(epochs), patience=int(patience),
                        seed=int(seed), ckpt_root=str(ckpt_root),
                    )
                    results.append(res)

                    # Incremental CSV save after every run
                    df = pd.DataFrame(results).sort_values(
                        ["head_type","val_mae_hot_all"]).reset_index(drop=True)
                    df.to_csv(csv_path, index=False)

                    # Live best-so-far per head
                    for hh in sorted(set(df["head_type"])):
                        best = df[df["head_type"]==hh].sort_values("val_mae_hot_all").iloc[0]
                        print(
                            f"[BEST {hh}] val_mae_all={best['val_mae_hot_all']:.3f} "
                            f"val_mae_ext={best['val_mae_hot_ext']:.3f} "
                            f"underRate={best['val_under_rate_ext']:.3f} "
                            f"k={best['k']} a={best['alpha']} bh={best['beta_hot']} ep={best['best_ep']}"
                        )

    df = pd.DataFrame(results).sort_values(["head_type","val_mae_hot_all"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)

    best_per_head = {}
    for hh in sorted(set(df["head_type"])):
        best_per_head[hh] = df[df["head_type"]==hh].sort_values("val_mae_hot_all").iloc[0].to_dict()

    with open(os.path.join(ckpt_root, "best_overall.json"), "w") as f:
        json.dump(best_per_head, f, indent=2)

    print("\n" + "="*100)
    print("[GRID DONE] results:", csv_path)
    print("="*100 + "\n")
    return df, best_per_head


# -------------------------
# RUN
# -------------------------
def _parse_floats(env_key, default):
    v = os.environ.get(env_key, "").strip()
    if not v:
        return default
    return tuple(float(x) for x in v.split(","))

RUN_TAG   = os.environ.get("RUN_TAG", "Center_pct_v1")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "./grid_ckpts_pct_weighted_IDW")
HEADS     = tuple(h.strip() for h in os.environ.get("HEADS", "temporalfusion,lstm").split(","))
KS        = _parse_floats("KS",    (1.0, 2.0, 3.0))
ALPHAS    = _parse_floats("ALPHAS",(1.0, 2.0, 3.0))
BETAS     = _parse_floats("BETAS", (0.0, 1.0, 1.5))
EPOCHS    = int(os.environ.get("EPOCHS", "120"))
PATIENCE  = int(os.environ.get("PATIENCE", "10"))
SEED      = int(os.environ.get("SEED", "333"))

print(f"[RUN] tag={RUN_TAG} ckpt_root={CKPT_ROOT}\n"
      f"      heads={HEADS} ks={KS} alphas={ALPHAS} betas={BETAS}\n"
      f"      epochs={EPOCHS} patience={PATIENCE} seed={SEED} save_ckpt={os.environ.get('SAVE_CKPT','0')}")

df_grid, best_per_head = run_big_grid(
    TAG=RUN_TAG,
    head_types=HEADS,
    ks=KS,
    alphas=ALPHAS,
    beta_hots=BETAS,
    patience=PATIENCE,
    epochs=EPOCHS,
    seed=SEED,
    ckpt_root=CKPT_ROOT,
)

print(df_grid.head(20))
print("\n[Best per head]\n", best_per_head)
