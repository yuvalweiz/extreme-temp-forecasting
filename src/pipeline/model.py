"""
Model — ported 1:1 from run_grid.py (CELL 2). ConvNeXt-Tiny spatial encoder applied
per daily frame -> (B,T,F) sequence -> temporal head -> OutputAffine -> (B,4).
Heads: PatchTST / iTransformer / TemporalFusionLite / LSTM. Arch identical to published.
"""
import torch
import torch.nn as nn

try:
    import timm
except Exception as e:  # pragma: no cover
    raise RuntimeError("timm is required (convnext_tiny).") from e


def denorm_y_torch(y_norm, y_median, y_iqr, y_transform="none"):
    y0 = y_norm
    if str(y_transform) == "asinh":
        y0 = torch.sinh(y0)
    return y0 * y_iqr + y_median


class ConvNeXtTinyBackbone(nn.Module):
    """pool='avg' = published (GAP over the final 1x4 map). 'avgmax'/'max'/'attn' pool the
    final map differently. spatial_stage (A1 fix): take an EARLIER, less-downsampled stage's
    feature map (0:4x->11x34, 1:8x->6x17, 2:16x->3x9 tokens vs final 32x->1x4) and pool over
    it -> the head can weight grid REGIONS instead of averaging all of Israel. Keeps the
    interpolated-frame spatial idea; just stops washing the grid out."""
    def __init__(self, in_chans: int, pretrained: bool = True, pool: str = "avg",
                 model_name: str = "convnext_tiny", spatial_stage=None):
        super().__init__()
        self.pool = str(pool)
        self.spatial_stage = spatial_stage
        if spatial_stage is not None:
            self.m = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans,
                                       features_only=True, out_indices=(int(spatial_stage),))
            base = int(self.m.feature_info.channels()[-1])
        else:
            gp = "avg" if self.pool == "avg" else ""
            self.m = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans,
                                       num_classes=0, global_pool=gp)
            base = int(getattr(self.m, "num_features", 0)) or -1
        if base <= 0:
            raise RuntimeError("Could not infer ConvNeXt feature dim.")
        self.out_dim = base * 2 if self.pool == "avgmax" else base
        if self.pool == "attn":
            self.attn = nn.Sequential(nn.Conv2d(base, max(base // 4, 16), 1), nn.GELU(),
                                      nn.Conv2d(max(base // 4, 16), 1, 1))

    def _featmap(self, x):
        if self.spatial_stage is not None:
            return self.m(x)[-1]                          # (B,C,H',W') from intermediate stage
        return self.m.forward_features(x)                 # (B,C,1,4) final map

    def forward(self, x):
        if self.pool == "avg" and self.spatial_stage is None:
            return self.m(x)
        fm = self._featmap(x)
        if self.pool in ("avg",):
            return fm.mean(dim=(2, 3))
        if self.pool == "max":
            return fm.amax(dim=(2, 3))
        if self.pool == "avgmax":
            return torch.cat([fm.mean(dim=(2, 3)), fm.amax(dim=(2, 3))], dim=1)
        if self.pool == "attn":
            B, C, H, W = fm.shape
            a = torch.softmax(self.attn(fm).reshape(B, 1, H * W), dim=-1).reshape(B, 1, H, W)
            return (fm * a).sum(dim=(2, 3))
        raise ValueError(self.pool)


def backbone_chunked_forward(backbone, x2d, chunk):
    N = x2d.shape[0]
    if chunk <= 0 or N <= chunk:
        return backbone(x2d)
    return torch.cat([backbone(x2d[i:i + chunk]) for i in range(0, N, chunk)], dim=0)


class PatchTSTHead(nn.Module):
    def __init__(self, d_in, d_model=256, nhead=8, layers=4, patch_len=30, stride=30, dropout=0.1, out_dim=4):
        super().__init__()
        self.patch_len, self.stride = int(patch_len), int(stride)
        self.proj = nn.Linear(d_in * self.patch_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                                         dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, out_dim))

    def forward(self, seq):
        B, T, F = seq.shape
        patches = [seq[:, s:s + self.patch_len, :].reshape(B, -1)
                   for s in range(0, T - self.patch_len + 1, self.stride)]
        if len(patches) == 0:
            if T < self.patch_len:
                seq = torch.cat([seq.new_zeros(B, self.patch_len - T, F), seq], dim=1)
            else:
                seq = seq[:, -self.patch_len:, :]
            patches = [seq.reshape(B, -1)]
        x = self.proj(torch.stack(patches, dim=1))
        return self.head(self.enc(x)[:, -1, :])


class iTransformerHead(nn.Module):
    def __init__(self, d_in, T, d_model=256, nhead=8, layers=2, dropout=0.1, out_dim=4):
        super().__init__()
        self.T = int(T)
        self.time_proj = nn.Linear(self.T, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                                         dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, out_dim))

    def forward(self, seq):
        B, T, F = seq.shape
        if T != self.T:
            seq = (torch.cat([seq.new_zeros(B, self.T - T, F), seq], dim=1) if T < self.T else seq[:, -self.T:, :])
        x = self.time_proj(seq.transpose(1, 2))
        return self.head(self.enc(x).mean(dim=1))


class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 2 * d)

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
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, out_dim))

    def forward(self, seq):
        x = self.in_proj(seq)
        x = x + self.glu(x)
        x = self.norm1(x)
        for layer in self.attn_layers:
            x = layer(x)
        return self.head(x[:, -1, :])


class LSTMHead(nn.Module):
    def __init__(self, d_in, d_model=256, layers=2, dropout=0.1, out_dim=4, bidirectional=False):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(input_size=d_in, hidden_size=d_model, num_layers=int(layers), batch_first=True,
                            dropout=float(dropout) if int(layers) > 1 else 0.0, bidirectional=self.bidirectional)
        hdim = d_model * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim, hdim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hdim, out_dim))

    def forward(self, seq):
        out, (h_n, c_n) = self.lstm(seq)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.head(h_last)


class VanillaTransformerHead(nn.Module):
    """The most standard Transformer encoder head (HEAD=vanilla): linear input
    projection -> learned positional embedding -> nn.TransformerEncoder
    (post-norm, GELU, batch_first) -> mean pooling -> linear output. No gating,
    no patching, no channel attention -- the textbook baseline."""
    def __init__(self, d_in, T, d_model=256, nhead=4, layers=2, dropout=0.1, out_dim=4):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, int(T), d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=4 * d_model,
                                               dropout=dropout, activation="gelu",
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(layers))
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))

    def forward(self, seq):                      # (B, T, F)
        x = self.proj(seq) + self.pos[:, :seq.shape[1]]
        x = self.encoder(x)
        return self.out(x.mean(dim=1))


class OutputAffine(nn.Module):
    def __init__(self, out_dim=4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, y):
        return y * self.scale + self.bias


class HeadOnlyModel(nn.Module):
    """Non-spatial control (tabular ablation): the (B,T,F) series goes straight to the
    temporal head — NO ConvNeXt, NO spatial structure. Same head/loss/protocol as the
    spatial model, so the difference isolates the spatial contribution."""
    def __init__(self, F, T, head_type, d_model=256, nhead=8, layers=2, dropout=0.1,
                 out_dim=4, head_layers=None):
        super().__init__()
        hl = head_layers if head_layers is not None else max(2, layers // 2)
        ht = str(head_type).lower()
        if ht == "temporalfusion":  self.head = TemporalFusionLiteHead(F, d_model, nhead, hl, dropout, out_dim)
        elif ht == "lstm":          self.head = LSTMHead(F, d_model, max(1, hl), dropout, out_dim)
        elif ht == "patchtst":      self.head = PatchTSTHead(F, d_model, nhead, hl, 30, 30, dropout, out_dim)
        elif ht == "itransformer":  self.head = iTransformerHead(F, T, d_model, nhead, hl, dropout, out_dim)
        elif ht == "vanilla":       self.head = VanillaTransformerHead(F, T, d_model, nhead, hl, dropout, out_dim)
        else: raise ValueError(head_type)
        self.out_affine = OutputAffine(out_dim)

    def forward(self, x):           # x: (B,T,F)
        return self.out_affine(self.head(x))


class FlatFrameEncoder(nn.Module):
    """No-CNN ablation control (BACKBONE=flat): flattens each daily frame and applies a
    single linear projection to the same 768-d embedding dimension ConvNeXt-Tiny
    produces. Identical input frames and temporal heads; only the spatial inductive
    bias (convolutional weight sharing / locality) is removed. Uses LazyLinear so the
    flattened dimension (C*H*W) is bound on first forward."""
    def __init__(self, out_dim: int = 768):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.LazyLinear(out_dim)

    def forward(self, x):            # x: (N, C, H, W)
        return self.proj(x.flatten(1))


class ConvNeXtTiny_WithHead(nn.Module):
    def __init__(self, in_chans, T, head_type, backbone_chunk=256, pretrained=True,
                 d_model=256, nhead=8, layers=4, dropout=0.1, use_out_affine=True,
                 out_dim=4, lstm_bidirectional=False, backbone_pool="avg",
                 backbone_name="convnext_tiny", head_layers=None, spatial_stage=None):
        super().__init__()
        if str(backbone_name).lower() == "flat":   # no-CNN linear-projection control
            self.backbone = FlatFrameEncoder(out_dim=768)
        else:
            self.backbone = ConvNeXtTinyBackbone(in_chans=in_chans, pretrained=pretrained,
                                                 pool=backbone_pool, model_name=backbone_name,
                                                 spatial_stage=spatial_stage)
        self.backbone_chunk = int(backbone_chunk)
        F = self.backbone.out_dim
        ht = str(head_type).lower()
        # head_layers=None reproduces the published derivation (TFT=2, LSTM=1 for layers=2)
        tf_l = head_layers if head_layers is not None else max(2, layers // 2)
        ls_l = head_layers if head_layers is not None else max(1, layers // 2)
        if ht == "patchtst":
            self.head = PatchTSTHead(F, d_model, nhead, head_layers or layers, 30, 30, dropout, out_dim)
        elif ht == "itransformer":
            self.head = iTransformerHead(F, T, d_model, nhead, head_layers or layers, dropout, out_dim)
        elif ht == "temporalfusion":
            self.head = TemporalFusionLiteHead(F, d_model, nhead, tf_l, dropout, out_dim)
        elif ht == "lstm":
            self.head = LSTMHead(F, d_model, ls_l, dropout, out_dim, lstm_bidirectional)
        elif ht == "vanilla":
            self.head = VanillaTransformerHead(F, T, d_model, nhead, tf_l, dropout, out_dim)
        else:
            raise ValueError("head_type must be one of: patchtst / itransformer / temporalfusion / lstm / vanilla")
        self.out_affine = OutputAffine(out_dim=out_dim) if use_out_affine else nn.Identity()

    def forward(self, X):
        B, T, C, H, W = X.shape
        x2d = X.reshape(B * T, C, H, W)
        feats = backbone_chunked_forward(self.backbone, x2d, self.backbone_chunk).reshape(B, T, -1)
        return self.out_affine(self.head(feats))


# ============================================================
# TASK C (opt-in, MULTITASK=1 in train.py) — multi-task cross-region model:
# ONE shared ConvNeXt backbone + one full temporal head (+ OutputAffine) per region.
# Reuses the exact head constructors of ConvNeXtTiny_WithHead (simplest faithful
# sharing scheme: shared spatial encoder, independent per-region heads).
# Nothing above this block is modified; the published single-head model is untouched.
# ============================================================
def build_temporal_head(head_type, F, T, d_model=256, nhead=8, layers=4, dropout=0.1,
                        out_dim=4, head_layers=None, lstm_bidirectional=False):
    """One temporal head, EXACTLY as ConvNeXtTiny_WithHead constructs it (same
    head_layers derivation: TFT=2 / LSTM=1 for layers=2 when head_layers=None)."""
    ht = str(head_type).lower()
    tf_l = head_layers if head_layers is not None else max(2, layers // 2)
    ls_l = head_layers if head_layers is not None else max(1, layers // 2)
    if ht == "patchtst":
        return PatchTSTHead(F, d_model, nhead, head_layers or layers, 30, 30, dropout, out_dim)
    if ht == "itransformer":
        return iTransformerHead(F, T, d_model, nhead, head_layers or layers, dropout, out_dim)
    if ht == "temporalfusion":
        return TemporalFusionLiteHead(F, d_model, nhead, tf_l, dropout, out_dim)
    if ht == "lstm":
        return LSTMHead(F, d_model, ls_l, dropout, out_dim, lstm_bidirectional)
    raise ValueError("head_type must be one of: patchtst / itransformer / temporalfusion / lstm")


class ConvNeXtTiny_MultiHead(nn.Module):
    """Shared backbone -> (B,T,F) sequence -> n_heads independent (head + OutputAffine)
    branches -> (B, n_heads, out_dim). Head params do NOT start with 'backbone.' so the
    existing backbone/head optimizer split in train.py applies unchanged. The backbone
    (self.backbone.m) is state-dict compatible with ConvNeXtTinyBackbone.m, so it can be
    saved in the pretrain enc_state_dict format and reloaded via PRETRAINED."""
    def __init__(self, in_chans, T, head_type, n_heads=3, backbone_chunk=256, pretrained=True,
                 d_model=256, nhead=8, layers=4, dropout=0.1, use_out_affine=True,
                 out_dim=4, lstm_bidirectional=False, backbone_pool="avg",
                 backbone_name="convnext_tiny", head_layers=None, spatial_stage=None):
        super().__init__()
        self.backbone = ConvNeXtTinyBackbone(in_chans=in_chans, pretrained=pretrained,
                                             pool=backbone_pool, model_name=backbone_name,
                                             spatial_stage=spatial_stage)
        self.backbone_chunk = int(backbone_chunk)
        F = self.backbone.out_dim
        self.heads = nn.ModuleList([
            build_temporal_head(head_type, F, T, d_model, nhead, layers, dropout,
                                out_dim, head_layers, lstm_bidirectional)
            for _ in range(int(n_heads))])
        self.out_affines = nn.ModuleList([
            OutputAffine(out_dim=out_dim) if use_out_affine else nn.Identity()
            for _ in range(int(n_heads))])

    def forward(self, X):                    # X: (B,T,C,H,W) -> (B, n_heads, out_dim)
        B, T, C, H, W = X.shape
        x2d = X.reshape(B * T, C, H, W)
        feats = backbone_chunked_forward(self.backbone, x2d, self.backbone_chunk).reshape(B, T, -1)
        return torch.stack([oa(h(feats)) for h, oa in zip(self.heads, self.out_affines)], dim=1)


# ============================================================
# HYBRID (opt-in, HYBRID=1 in train.py, PAPERW synth mode only) — frames + raw
# station vectors: per timestep, CNN(frame) features CONCAT a projection of the raw
# (T, sv_dim=612) station vector -> the standard temporal head. Strictly more
# information than the fair tabular control (which sees only the raw vectors).
# Nothing above this block is modified; the published single-input model is untouched.
# ============================================================
class ConvNeXtTiny_Hybrid(nn.Module):
    """frames (B,T,C,H,W) -> ConvNeXt backbone -> (B,T,768); sv (B,T,sv_dim) ->
    Linear(sv_dim->d_sv)+GELU+LayerNorm -> (B,T,d_sv); concat -> (B,T,768+d_sv) ->
    temporal head (built EXACTLY as ConvNeXtTiny_WithHead builds it, d_in=768+d_sv)
    -> OutputAffine. sv_proj does NOT start with 'backbone.', so the existing
    backbone/head optimizer split in train.py gives it the head LR (fresh layer).
    forward accepts a HybridBatch / (frames, sv) pair as one arg, or model(frames, sv)."""
    def __init__(self, in_chans, sv_dim, T, head_type, d_sv=256, backbone_chunk=256,
                 pretrained=True, d_model=256, nhead=8, layers=4, dropout=0.1,
                 use_out_affine=True, out_dim=4, lstm_bidirectional=False,
                 backbone_pool="avg", backbone_name="convnext_tiny", head_layers=None,
                 spatial_stage=None):
        super().__init__()
        self.backbone = ConvNeXtTinyBackbone(in_chans=in_chans, pretrained=pretrained,
                                             pool=backbone_pool, model_name=backbone_name,
                                             spatial_stage=spatial_stage)
        self.backbone_chunk = int(backbone_chunk)
        self.sv_proj = nn.Sequential(nn.Linear(int(sv_dim), int(d_sv)), nn.GELU(),
                                     nn.LayerNorm(int(d_sv)))
        F = self.backbone.out_dim + int(d_sv)
        self.head = build_temporal_head(head_type, F, T, d_model, nhead, layers, dropout,
                                        out_dim, head_layers, lstm_bidirectional)
        self.out_affine = OutputAffine(out_dim=out_dim) if use_out_affine else nn.Identity()

    def forward(self, X, sv=None):
        if sv is None:   # HybridBatch (duck-typed) or a plain (frames, sv) pair
            X, sv = (X.frames, X.sv) if hasattr(X, "frames") else (X[0], X[1])
        B, T, C, H, W = X.shape
        x2d = X.reshape(B * T, C, H, W)
        feats = backbone_chunked_forward(self.backbone, x2d, self.backbone_chunk).reshape(B, T, -1)
        seq = torch.cat([feats, self.sv_proj(sv)], dim=-1)         # (B,T,768+d_sv)
        return self.out_affine(self.head(seq))


# ============================================================
# BACKBONE3D (opt-in, BACKBONE3D=1 in train.py) — joint SPACE-TIME encoder, the one
# architecture family never tested: every existing model encodes each daily frame with a
# 2D CNN and leaves ALL temporal mixing to the head; this one convolves jointly over
# (time, H, W) with a small factorized (2+1)D stem+stages, then reuses the standard
# temporal head on the temporally-downsampled feature sequence. Deliberately small
# (~2.3M params with the published head config; 813 train samples — a big video model
# would overfit). Same input contract as ConvNeXtTiny_WithHead: (B,T,C,H,W) -> (B,out_dim),
# so the frames and PAPERW-synth pipelines drive it unchanged.
# Nothing above this block is modified; the published per-frame model is untouched.
# ============================================================
class SpatioTemporal3D(nn.Module):
    """(B,T,C,H,W) -> permute -> (B,C,T,H,W) volume ->
    stage1: Conv3d(C->48, k=(1,4,4), s=(1,4,4)) [per-day spatial patchify] + norm + GELU;
    stage2: 2x factorized (2+1)D block [Conv3d k=(1,3,3) pad spatial -> norm/GELU ->
            Conv3d k=(5,1,1) s=(2,1,1) pad temporal -> norm/GELU], channels 48->96->192,
            T downsampled 180->90->45 (formula robust to ANY T>=1, e.g. smoke T=8 -> 2);
    stage3: spatial GAP -> (B,T',192) feature sequence ->
    temporal head built by build_temporal_head (EXACTLY as ConvNeXtTiny_WithHead builds
    its head, d_in=192, T=T') -> OutputAffine -> (B,out_dim).
    Norms are GroupNorm(1,C) (per-sample, batch-size independent — published batch=4).
    NO parameter is named 'backbone.*' ON PURPOSE: the encoder trains FROM SCRATCH, so
    train.py's existing optimizer split gives every parameter the head LR (1e-4) instead
    of the pretrained-finetune backbone LR (1e-5)."""
    def __init__(self, in_chans, T, head_type, d_model=256, nhead=8, layers=4,
                 dropout=0.1, use_out_affine=True, out_dim=4, lstm_bidirectional=False,
                 head_layers=None, widths=(48, 96, 192), t_kernel=5, t_stride=2):
        super().__init__()
        w0, w1, w2 = (int(w) for w in widths)
        tk, ts = int(t_kernel), int(t_stride)

        def _block(ci, co):   # (2+1)D: spatial-only conv, then strided temporal-only conv
            return nn.Sequential(
                nn.Conv3d(ci, co, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.GroupNorm(1, co), nn.GELU(),
                nn.Conv3d(co, co, kernel_size=(tk, 1, 1), stride=(ts, 1, 1),
                          padding=(tk // 2, 0, 0)),
                nn.GroupNorm(1, co), nn.GELU())

        self.stem = nn.Sequential(
            nn.Conv3d(int(in_chans), w0, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.GroupNorm(1, w0), nn.GELU())
        self.stages = nn.Sequential(_block(w0, w1), _block(w1, w2))
        t_out = int(T)
        for _ in range(2):                    # temporal length after the 2 strided convs
            t_out = (t_out + 2 * (tk // 2) - tk) // ts + 1
        self.t_out, self.feat_dim = t_out, w2
        self.head = build_temporal_head(head_type, w2, t_out, d_model, nhead, layers,
                                        dropout, out_dim, head_layers, lstm_bidirectional)
        self.out_affine = OutputAffine(out_dim=out_dim) if use_out_affine else nn.Identity()

    def forward(self, X):                     # X: (B,T,C,H,W)
        v = X.permute(0, 2, 1, 3, 4)          # (B,C,T,H,W) joint space-time volume
        f = self.stages(self.stem(v))         # (B,192,T',H',W')
        seq = f.mean(dim=(3, 4)).transpose(1, 2)   # spatial GAP -> (B,T',192)
        return self.out_affine(self.head(seq))
