"""
Shared, importable model + loss definitions (extracted verbatim from the proven
run_grid_2.py / train_grid_hot.py). No side effects on import, so both the grid
trainer and the frame-based trainer use byte-identical model code -> reproducible.

Architecture: ConvNeXt-Tiny per-frame encoder -> temporal head (TFT-Lite / LSTM /
PatchTST / iTransformer). Loss: percentile-weighted MAE (k) with under-prediction
multiplier (alpha) + optional hinge (beta), out-weights geometric decay.
"""
import torch
import torch.nn as nn

try:
    import timm
except Exception as e:  # pragma: no cover
    raise RuntimeError("timm required.") from e


def denorm_y_torch(y_norm, y_median, y_iqr, y_transform="none"):
    y0 = torch.sinh(y_norm) if str(y_transform) == "asinh" else y_norm
    return y0 * y_iqr + y_median


class PercentileWeightedMAE(nn.Module):
    """w(r)=1 if r<0.5 else exp(k*(r-0.5)); alpha multiplies under-predictions on
    hot samples (r>=0.5). r = ECDF rank of true anchor in TRAIN distribution."""
    def __init__(self, y_median, y_iqr, y_transform, out_weights,
                 train_hot_sorted, k=2.0, alpha=2.0):
        super().__init__()
        self.register_buffer("y_median", torch.as_tensor(y_median, dtype=torch.float32))
        self.register_buffer("y_iqr", torch.as_tensor(y_iqr, dtype=torch.float32))
        self.register_buffer("w_out", torch.tensor(out_weights, dtype=torch.float32))
        self.register_buffer("train_sorted", torch.as_tensor(train_hot_sorted, dtype=torch.float32))
        self.y_transform = str(y_transform)
        self.k = float(k)
        self.alpha = float(alpha)

    def get_sample_weights(self, true_hot):
        idx = torch.searchsorted(self.train_sorted, true_hot.contiguous())
        ranks = idx.float() / float(len(self.train_sorted))
        is_hot = ranks >= 0.5
        w = torch.where(is_hot, torch.exp(self.k * (ranks - 0.5)), torch.ones_like(ranks))
        return w, is_hot

    def forward(self, pred_norm, true_raw_c):
        pred_c = denorm_y_torch(pred_norm, self.y_median, self.y_iqr, self.y_transform)
        true_hot = true_raw_c[:, 0]
        sample_w, is_hot = self.get_sample_weights(true_hot)
        err = pred_c - true_raw_c
        abs_err = err.abs()
        is_under_hot = (err < 0) & is_hot.unsqueeze(1)
        penalized = torch.where(is_under_hot, abs_err * self.alpha, abs_err)
        per_sample = (penalized * self.w_out.to(penalized.device)).mean(dim=1)
        return (per_sample * sample_w).mean()


class PercentileWeightedHinge(nn.Module):
    def __init__(self, base_loss, beta_hot=1.0):
        super().__init__()
        self.base = base_loss
        self.beta_hot = float(beta_hot)

    def forward(self, pred_norm, true_raw_c):
        base = self.base(pred_norm, true_raw_c)
        if self.beta_hot == 0.0:
            return base
        pred_c = denorm_y_torch(pred_norm, self.base.y_median, self.base.y_iqr, self.base.y_transform)
        true_hot = true_raw_c[:, 0]
        sample_w, _ = self.base.get_sample_weights(true_hot)
        extra = (torch.relu(true_hot - pred_c[:, 0]) * sample_w).mean()
        return base + self.beta_hot * extra


class AttnPool(nn.Module):
    """Learned attention pooling over spatial tokens — keeps WHERE the signal is,
    instead of GAP averaging it away. (Architecture experiment, see HYPERPARAMETERS.)"""
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim) * dim ** -0.5)
        self.scale = dim ** -0.5

    def forward(self, x):  # x: (B, C, H, W)
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)            # (B, HW, C)
        a = (t @ self.q) * self.scale               # (B, HW)
        w = torch.softmax(a, dim=1).unsqueeze(-1)   # (B, HW, 1)
        return (w * t).sum(1)                        # (B, C)


class ConvNeXtTinyBackbone(nn.Module):
    """pool='avg' (default, GAP — published) or 'attn' (learned attention pooling)
    or 'avgmax' (concat mean+max, then linear back to dim)."""
    def __init__(self, in_chans, pretrained=True, pool="avg"):
        super().__init__()
        self.pool = str(pool)
        gp = "avg" if self.pool == "avg" else ""
        self.m = timm.create_model("convnext_tiny", pretrained=pretrained,
                                   in_chans=in_chans, num_classes=0, global_pool=gp)
        self.out_dim = int(getattr(self.m, "num_features", 0)) or 768
        if self.pool == "attn":
            self.attn = AttnPool(self.out_dim)
        elif self.pool == "avgmax":
            self.reduce = nn.Linear(2 * self.out_dim, self.out_dim)
        elif self.pool == "stats":  # mean+max+std: std captures spatial STRUCTURE the mean discards
            self.reduce = nn.Linear(3 * self.out_dim, self.out_dim)

    def forward(self, x):
        if self.pool == "avg":
            return self.m(x)
        feat = self.m.forward_features(x)            # (B, C, H, W)
        if self.pool == "attn":
            return self.attn(feat)
        g = feat.flatten(2)                          # (B, C, HW)
        if self.pool == "avgmax":
            return self.reduce(torch.cat([g.mean(-1), g.amax(-1)], dim=1))
        # stats: mean + max + std
        return self.reduce(torch.cat([g.mean(-1), g.amax(-1), g.std(-1)], dim=1))


def backbone_chunked_forward(backbone, x2d, chunk):
    if chunk <= 0 or x2d.shape[0] <= chunk:
        return backbone(x2d)
    return torch.cat([backbone(x2d[i:i + chunk]) for i in range(0, x2d.shape[0], chunk)])


class PatchTSTHead(nn.Module):
    def __init__(self, d_in, d_model=256, nhead=8, layers=4, patch_len=30, stride=30, dropout=0.1, out_dim=4):
        super().__init__()
        self.patch_len = int(patch_len); self.stride = int(stride)
        self.proj = nn.Linear(d_in * self.patch_len, d_model)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True), num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, out_dim))

    def forward(self, seq):
        B, T, F = seq.shape
        patches = [seq[:, s:s + self.patch_len, :].reshape(B, -1)
                   for s in range(0, T - self.patch_len + 1, self.stride)]
        if not patches:
            seq = seq[:, -self.patch_len:, :] if T >= self.patch_len else torch.cat([seq.new_zeros(B, self.patch_len - T, F), seq], 1)
            patches = [seq.reshape(B, -1)]
        x = self.proj(torch.stack(patches, 1))
        return self.head(self.enc(x)[:, -1, :])


class iTransformerHead(nn.Module):
    def __init__(self, d_in, T, d_model=256, nhead=8, layers=2, dropout=0.1, out_dim=4):
        super().__init__()
        self.T = int(T)
        self.time_proj = nn.Linear(self.T, d_model)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True), num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, out_dim))

    def forward(self, seq):
        B, T, F = seq.shape
        if T < self.T:
            seq = torch.cat([seq.new_zeros(B, self.T - T, F), seq], 1)
        elif T > self.T:
            seq = seq[:, -self.T:, :]
        return self.head(self.enc(self.time_proj(seq.transpose(1, 2))).mean(1))


class GLU(nn.Module):
    def __init__(self, d):
        super().__init__(); self.fc = nn.Linear(d, 2 * d)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class TemporalFusionLiteHead(nn.Module):
    def __init__(self, d_in, d_model=256, nhead=8, layers=2, dropout=0.1, out_dim=4):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.glu = GLU(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
            for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, out_dim))

    def forward(self, seq):
        x = self.in_proj(seq); x = x + self.glu(x); x = self.norm1(x)
        for layer in self.attn_layers:
            x = layer(x)
        return self.head(x[:, -1, :])


class LSTMHead(nn.Module):
    def __init__(self, d_in, d_model=256, layers=2, dropout=0.1, out_dim=4, bidirectional=False):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(d_in, d_model, num_layers=int(layers), batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0, bidirectional=self.bidirectional)
        hdim = d_model * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim, hdim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hdim, out_dim))

    def forward(self, seq):
        _, (h_n, _) = self.lstm(seq)
        h = torch.cat([h_n[-2], h_n[-1]], 1) if self.bidirectional else h_n[-1]
        return self.head(h)


class OutputAffine(nn.Module):
    def __init__(self, out_dim=4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, y):
        return y * self.scale + self.bias


class ConvNeXtTiny_WithHead(nn.Module):
    def __init__(self, in_chans, T, head_type, backbone_chunk=256, pretrained=True,
                 d_model=256, nhead=8, layers=4, dropout=0.1, use_out_affine=True,
                 lstm_bidirectional=False, backbone_pool="avg", out_dim=4):
        super().__init__()
        self.backbone = ConvNeXtTinyBackbone(in_chans, pretrained, pool=backbone_pool)
        self.backbone_chunk = int(backbone_chunk)
        F = self.backbone.out_dim
        ht = str(head_type).lower()
        od = int(out_dim)
        if ht == "patchtst":
            self.head = PatchTSTHead(F, d_model, nhead, layers, 30, 30, dropout, od)
        elif ht == "itransformer":
            self.head = iTransformerHead(F, T, d_model, nhead, layers, dropout, od)
        elif ht == "temporalfusion":
            self.head = TemporalFusionLiteHead(F, d_model, nhead, max(2, layers // 2), dropout, od)
        elif ht == "lstm":
            self.head = LSTMHead(F, d_model, max(1, layers // 2), dropout, od, lstm_bidirectional)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
        self.out_affine = OutputAffine(od) if bool(use_out_affine) else nn.Identity()

    def forward(self, X):
        B, T, C, H, W = X.shape
        feats = backbone_chunked_forward(
            self.backbone, X.reshape(B * T, C, H, W), self.backbone_chunk).reshape(B, T, -1)
        return self.out_affine(self.head(feats))
