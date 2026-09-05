"""Independent Gaussian/Fisher objective and predetermined channel schedule."""

import numpy as np


class NormalFisher:
    """Weighted Normal NLL gradients with diagonal expected Fisher curvature.

    CPU is the verified capability. No raw clipping or exposure support.
    """

    channel_names = ("mu", "log_sigma")
    supported_devices = frozenset({"cpu"})

    @staticmethod
    def _inputs(y, sample_weight, extra):
        if extra:
            raise ValueError("NormalFisher does not support extra targets/exposure")
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or not y.size or not np.isfinite(y).all():
            raise ValueError("Require nonempty finite one-dimensional targets")
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("Require finite nonnegative weights with positive sum")
        return y, w

    def init_raw(self, y, sample_weight=None, extra=None):
        y, w = self._inputs(y, sample_weight, extra)
        mu = np.average(y, weights=w)
        variance = max(float(np.average((y - mu) ** 2, weights=w)), 1e-6)
        return {"mu": float(mu), "log_sigma": float(0.5 * np.log(variance))}

    def _terms(self, raw, y, sample_weight, extra, context):
        if context.device != "cpu" or context.xp is not np:
            raise ValueError("NormalFisher currently supports CPU only")
        y, w = self._inputs(y, sample_weight, extra)
        if set(raw) != set(self.channel_names):
            raise ValueError("Require mu and log_sigma channels")
        if any(v.shape != y.shape or not np.isfinite(v).all() for v in raw.values()):
            raise ValueError("Invalid raw shape or non-finite state")
        residual = raw["mu"].astype(float) - y
        logs = raw["log_sigma"].astype(float)
        with np.errstate(over="ignore", invalid="ignore"):
            precision = np.exp(-2 * logs)
            squared = residual**2 * precision
        if not np.isfinite(precision).all() or not np.isfinite(squared).all():
            raise ValueError("Non-finite Normal state")
        return residual, logs, precision, squared, w

    def step(self, raw, y, sample_weight=None, extra=None, *, context):
        residual, _, precision, squared, w = self._terms(raw, y, sample_weight, extra, context)
        with np.errstate(over="ignore", invalid="ignore"):
            arrays = [
                np.ascontiguousarray(v, dtype=np.float32)
                for v in (residual * precision * w, precision * w, (1 - squared) * w, 2 * w)
            ]
        if any(not np.isfinite(v).all() for v in arrays):
            raise ValueError("Non-finite float32 Normal statistics")
        return {"mu": (arrays[0], arrays[1]), "log_sigma": (arrays[2], arrays[3])}

    def loss_value(self, raw, y, sample_weight=None, extra=None, *, context):
        _, logs, _, squared, w = self._terms(raw, y, sample_weight, extra, context)
        return float(np.average(logs + 0.5 * squared + 0.5 * np.log(2 * np.pi), weights=w))

    def constrain(self, raw, extra=None):
        if extra or set(raw) != set(self.channel_names):
            raise ValueError("Require mu/log_sigma without extra targets")
        with np.errstate(over="ignore", invalid="ignore"):
            sigma = np.exp(raw["log_sigma"].astype(float))
        if not np.isfinite(raw["mu"]).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("Non-finite or nonpositive Normal scale")
        return {"mu": raw["mu"].copy(), "sigma": sigma}


class ChannelDecay:
    """Full coefficients base_lr * channel_scale / (1 + round_idx / tau)."""

    def __init__(self, tau=1.0):
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        self.tau = float(tau)

    def coefficients(self, round_idx, channel_names, base_learning_rate):
        if tuple(channel_names) != ("mu", "log_sigma"):
            raise ValueError("ChannelDecay requires ordered mu/log_sigma channels")
        if round_idx < 0 or not np.isfinite(base_learning_rate) or base_learning_rate < 0:
            raise ValueError("Invalid round or base learning rate")
        rate = base_learning_rate / (1 + round_idx / self.tau)
        return {"mu": rate, "log_sigma": 0.5 * rate}
