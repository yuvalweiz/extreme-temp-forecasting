"""
Loss — ported 1:1 from run_grid.py (CELL 2). This is the PUBLISHED loss.

WeightedMAE_ExtremeUnderOnly_Celsius:
  - always MAE (in Celsius, on denormalized predictions)
  - on EXTREME samples only (true_hot >= hot_thr): under-predictions are multiplied
    by under_alpha_ext; non-extreme samples use multiplier 1.0
  - per-output weights `w` (published = (1,1,1,1))
ExtremeUnderHinge:
  - base + beta_hot * mean( ReLU(true_hot - pred_hot) on extreme samples )

There is NO percentile weight w(r)=exp(k(r-0.5)) and NO `k` here — that belonged to a
different (run_grid_2/avg3) variant the author did NOT publish from.
"""
import torch
import torch.nn as nn
from .model import denorm_y_torch


class WeightedMAE_ExtremeUnderOnly_Celsius(nn.Module):
    def __init__(self, y_median, y_iqr, y_transform="none",
                 out_weights=(1.0, 1.0, 1.0, 1.0), hot_thr=0.0, under_alpha_ext=4.0):
        super().__init__()
        self.register_buffer("y_median", torch.as_tensor(y_median, dtype=torch.float32))
        self.register_buffer("y_iqr", torch.as_tensor(y_iqr, dtype=torch.float32))
        self.y_transform = str(y_transform)
        self.register_buffer("w", torch.tensor(out_weights, dtype=torch.float32))
        self.hot_thr = float(hot_thr)
        self.under_alpha_ext = float(under_alpha_ext)

    def forward(self, pred_norm, true_raw_c):
        pred_c = denorm_y_torch(pred_norm, self.y_median, self.y_iqr, self.y_transform)
        true_hot = true_raw_c[:, 0]
        is_ext = (true_hot >= self.hot_thr).unsqueeze(1)
        err = pred_c - true_raw_c
        abs_err = err.abs()
        under = (err < 0)
        alpha = torch.where(is_ext, self.under_alpha_ext, 1.0)
        penalized = torch.where(under, abs_err * alpha, abs_err)
        weighted = penalized * self.w.to(penalized.device)
        return weighted.mean()


class PinballAnchor(nn.Module):
    """Opt-in (PINBALL env): base + weight * pinball_tau on output 0 (anchor), in Celsius.
    tau>0.5 penalizes under-prediction more (hot peaks); tau<0.5 over-prediction (cold
    troughs). Wraps the FULL existing loss (incl. hinge); published losses untouched."""
    def __init__(self, base_loss, tau=0.7, weight=1.0):
        super().__init__()
        self.base = base_loss
        self.tau, self.weight = float(tau), float(weight)

    def forward(self, pred_norm, true_raw_c):
        b = self.base(pred_norm, true_raw_c)
        inner = self.base.base if hasattr(self.base, "base") else self.base
        pred_c = denorm_y_torch(pred_norm, inner.y_median, inner.y_iqr, inner.y_transform)
        u = true_raw_c[:, 0] - pred_c[:, 0]          # >0 = under-prediction of the anchor
        pb = torch.maximum(self.tau * u, (self.tau - 1.0) * u).mean()
        return b + self.weight * pb


class ExtremeUnderHinge(nn.Module):
    def __init__(self, base_loss, hot_thr, beta_hot=1.5):
        super().__init__()
        self.base = base_loss
        self.hot_thr = float(hot_thr)
        self.beta_hot = float(beta_hot)

    def forward(self, pred_norm, true_raw_c):
        base = self.base(pred_norm, true_raw_c)
        pred_c = denorm_y_torch(pred_norm, self.base.y_median, self.base.y_iqr, self.base.y_transform)
        true_hot = true_raw_c[:, 0]
        is_ext = (true_hot >= self.hot_thr).float()
        hot_under = torch.relu(true_hot - pred_c[:, 0])
        extra = (hot_under * is_ext).mean()
        return base + self.beta_hot * extra


# ============================================================
# COLD (MIN targets) — ported 1:1 from run_grid_cold_winter.py (CELL 2).
# Opt-in only (COLD=1); the hot path above is untouched.
#
# PercentileWeightedMAE_Min:
#   r = ECDF rank of true_cold (output 0) in the TRAIN cold distribution
#       (low temps => low ranks). is_cold = (r <= 0.5).
#   sample weight w = exp(k*(0.5 - r)) on cold samples, else 1.0.
#   Directional α penalty: on COLD samples, "too-warm" errors (pred > true, err>0)
#   are multiplied by alpha. Per-output weights `w_out` then per-sample weight `w`.
# PercentileWeightedHinge_Min:
#   base + beta_hot * mean( ReLU(pred_cold - true_cold) * w )  (extra "too-warm" push).
# ============================================================
class PercentileWeightedMAE_Min(nn.Module):
    def __init__(self, y_median, y_iqr, y_transform, out_weights,
                 train_cold_sorted, k=2.0, alpha=2.0):
        super().__init__()
        self.register_buffer("y_median", torch.as_tensor(y_median, dtype=torch.float32))
        self.register_buffer("y_iqr", torch.as_tensor(y_iqr, dtype=torch.float32))
        self.register_buffer("w_out", torch.tensor(out_weights, dtype=torch.float32))
        self.register_buffer("train_sorted", torch.as_tensor(train_cold_sorted, dtype=torch.float32))
        self.y_transform = str(y_transform)
        self.k = float(k)
        self.alpha = float(alpha)

    def get_sample_weights(self, true_cold):
        """true_cold: (B,) -> sample_weight (B,), is_cold (B,) bool."""
        idx = torch.searchsorted(self.train_sorted, true_cold.contiguous())
        ranks = idx.float() / float(len(self.train_sorted))   # low cold -> low rank
        is_cold = ranks <= 0.5
        w = torch.where(is_cold, torch.exp(self.k * (0.5 - ranks)), torch.ones_like(ranks))
        return w, is_cold

    def forward(self, pred_norm, true_raw_c):
        pred_c = denorm_y_torch(pred_norm, self.y_median, self.y_iqr, self.y_transform)
        true_cold = true_raw_c[:, 0]
        sample_w, is_cold = self.get_sample_weights(true_cold)

        err = pred_c - true_raw_c                  # (B,4)
        abs_err = err.abs()
        # alpha on "too-warm" errors (pred > true -> err > 0) on cold samples
        is_too_warm_on_cold = (err > 0) & is_cold.unsqueeze(1)
        penalized = torch.where(is_too_warm_on_cold, abs_err * self.alpha, abs_err)

        per_sample = (penalized * self.w_out.to(penalized.device)).mean(dim=1)
        return (per_sample * sample_w).mean()


class PercentileWeightedHinge_Min(nn.Module):
    """Optional additive hinge on output 0 for "too-warm" predictions:
    extra = relu(pred_cold - true_cold), weighted by the same percentile weight
    (colder => higher weight). beta_hot kept as a name for grid compatibility."""
    def __init__(self, base_loss, beta_hot=1.0):
        super().__init__()
        self.base = base_loss
        self.beta_hot = float(beta_hot)

    def forward(self, pred_norm, true_raw_c):
        base = self.base(pred_norm, true_raw_c)
        if self.beta_hot == 0.0:
            return base
        pred_c = denorm_y_torch(pred_norm, self.base.y_median, self.base.y_iqr, self.base.y_transform)
        true_cold = true_raw_c[:, 0]
        sample_w, is_cold = self.base.get_sample_weights(true_cold)
        extra = (torch.relu(pred_c[:, 0] - true_cold) * sample_w).mean()
        return base + self.beta_hot * extra
