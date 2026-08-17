"""Objectives for the unified boosting trainer.

An objective turns current raw scores F (one channel per boosted parameter)
into per-sample step directions (grad, hess) that ``fit_tree`` consumes.

Two first-class implementations:

- ``DistributionObjective`` — NLL / natural-gradient path used by
  DistributionalGBDT and NaturalBoost.
- ``FormulaObjective`` — user formula ``f(theta, x)`` with damped generalized
  Gauss-Newton preconditioning (the FormulaBoost path).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ._distributions import Distribution

RawScores = dict[str, NDArray]
GradHess = dict[str, tuple[NDArray, NDArray]]


class Objective(Protocol):
    """Internal protocol consumed by ``fit_boosting``."""

    channel_names: list[str]

    def init_raw(
        self,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Return per-channel base scores in raw (unconstrained) space."""
        ...

    def step(
        self,
        raw: RawScores,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> GradHess:
        """Per-channel (gradient, hessian) w.r.t. raw scores."""
        ...

    def loss_value(
        self,
        raw: RawScores,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> float:
        """Scalar loss (lower is better) for train reporting / default eval."""
        ...

    def constrain(
        self,
        raw: RawScores,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, NDArray]:
        """Map raw scores to constrained parameters."""
        ...


def _is_device(arr) -> bool:
    return hasattr(arr, "__cuda_array_interface__")


def _apply_sample_weight(grads: GradHess, sample_weight: NDArray | None) -> GradHess:
    if sample_weight is None:
        return grads
    w = sample_weight.astype(np.float32, copy=False)
    return {
        name: ((g * w).astype(np.float32), (h * w).astype(np.float32))
        for name, (g, h) in grads.items()
    }


# =============================================================================
# Distribution objective (NaturalBoost / DistributionalGBDT)
# =============================================================================


class DistributionObjective:
    """Boost each distribution parameter from an NLL (optionally natural)."""

    def __init__(
        self,
        distribution: Distribution,
        *,
        natural: bool = False,
        exposure_param: str | None = None,
        exposure_sign: float = 0.0,
    ):
        self.distribution = distribution
        self.natural = natural
        self.exposure_param = exposure_param
        self.exposure_sign = exposure_sign
        self.channel_names = list(distribution.param_names)
        self._device_kernels_ok = True

    @property
    def device_capable(self) -> bool:
        """True when grad/hess can be computed on device-resident raw scores."""
        return type(self.distribution).__name__ in ("Normal", "Poisson")

    @property
    def unit_hessian(self) -> bool:
        return bool(self.natural)

    def init_raw(
        self,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        return {k: float(v) for k, v in self.distribution.init_params(y).items()}

    def constrain(
        self,
        raw: RawScores,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, NDArray]:
        log_offset = None if extra is None else extra.get("log_offset")
        params = {}
        for name in self.channel_names:
            score = raw[name]
            if log_offset is not None and name == self.exposure_param:
                score = score + log_offset
            params[name] = self.distribution.link(name, score)
        return params

    def step(
        self,
        raw: RawScores,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> GradHess:
        first = next(iter(raw.values()))
        if (
            self.device_capable
            and self._device_kernels_ok
            and _is_device(first)
            and (extra is None or extra.get("log_offset") is None)
        ):
            return self._step_device(raw, y, sample_weight)
        params = self.constrain(raw, extra)
        if self.natural:
            grads = self.distribution.natural_gradient(y, params)
        else:
            grads = self.distribution.nll_gradient(y, params)
        return _apply_sample_weight(grads, sample_weight)

    def _step_device(
        self,
        raw: RawScores,
        y: NDArray,
        sample_weight: NDArray | None,
    ) -> GradHess:
        try:
            from ._backends._cuda import normal_step_gpu, poisson_step_gpu, scale_gh_gpu

            name = type(self.distribution).__name__
            if name == "Normal":
                g_loc, h_loc, g_scale, h_scale = normal_step_gpu(
                    raw["loc"], raw["scale"], y, natural=self.natural
                )
                grads: GradHess = {"loc": (g_loc, h_loc), "scale": (g_scale, h_scale)}
            elif name == "Poisson":
                g, h = poisson_step_gpu(raw["rate"], y, natural=self.natural)
                grads = {"rate": (g, h)}
            else:  # pragma: no cover
                raise RuntimeError(f"No device step for {name}")
            if sample_weight is not None:
                for g, h in grads.values():
                    scale_gh_gpu(g, h, sample_weight)
            return grads
        except Exception:
            self._device_kernels_ok = False
            # Kernel compile can fail on some numpy/numba-cuda combos;
            # fall back to the host path and let the trainer upload grads.
            raw_host = {
                k: (v.copy_to_host() if hasattr(v, "copy_to_host") else np.asarray(v))
                for k, v in raw.items()
            }
            y_host = y.copy_to_host() if hasattr(y, "copy_to_host") else np.asarray(y)
            sw_host = None
            if sample_weight is not None:
                sw_host = (
                    sample_weight.copy_to_host()
                    if hasattr(sample_weight, "copy_to_host")
                    else np.asarray(sample_weight)
                )
            params = self.constrain(raw_host)
            if self.natural:
                grads = self.distribution.natural_gradient(y_host, params)
            else:
                grads = self.distribution.nll_gradient(y_host, params)
            return _apply_sample_weight(grads, sw_host)

    def loss_value(
        self,
        raw: RawScores,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> float:
        params = self.constrain(raw, extra)
        nll = self.distribution.nll(y, params)
        return float(np.average(nll, weights=sample_weight))


# =============================================================================
# Formula objective (FormulaBoost)
# =============================================================================

# link(raw) -> constrained; link_prime(raw) -> d(constrained)/d(raw)
_LINKS: dict[str, tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]] = {
    "identity": (lambda r: r, lambda r: np.ones_like(r, dtype=np.float64)),
    "log": (np.exp, np.exp),
    "softplus": (
        lambda r: np.logaddexp(0.0, r),
        lambda r: 1.0 / (1.0 + np.exp(-r)),
    ),
    "sigmoid": (
        lambda r: 1.0 / (1.0 + np.exp(-r)),
        lambda r: (s := 1.0 / (1.0 + np.exp(-r))) * (1.0 - s),
    ),
}


def _link_pair(name: str):
    if name not in _LINKS:
        raise ValueError(
            f"Unknown link '{name}'. Available: {', '.join(sorted(_LINKS))}."
        )
    return _LINKS[name]


def _formula_and_jac(
    formula: Callable,
    theta: list[NDArray],
    x: NDArray,
    eps: float = 1e-5,
) -> tuple[NDArray, NDArray]:
    """Evaluate ``formula(theta, x)`` and a forward-difference Jacobian.

    Returns ``(f, J)`` with ``f`` shape ``(n,)`` and ``J`` shape ``(n, K)``
    where ``J[:, k] = df / d theta_k``.
    """
    f0 = np.asarray(formula(tuple(theta), x), dtype=np.float64).ravel()
    k = len(theta)
    jac = np.empty((f0.shape[0], k), dtype=np.float64)
    for j in range(k):
        bumped = list(theta)
        bumped[j] = theta[j] + eps
        f1 = np.asarray(formula(tuple(bumped), x), dtype=np.float64).ravel()
        jac[:, j] = (f1 - f0) / eps
    return f0, jac


def _ggn_step(
    residual: NDArray,
    jac_raw: NDArray,
    precond: str,
    damp: float,
) -> NDArray:
    """Per-sample GGN step directions, shape ``(n, K)``.

    MSE / 0.5*(f-y)^2 => g = residual * J, G = J^T J (per sample).
    """
    g = residual[:, None] * jac_raw  # (n, K)
    if precond == "plain":
        return g
    if precond == "diag":
        return g / (jac_raw * jac_raw + damp)
    if precond != "full":
        raise ValueError(
            f"Unknown precond '{precond}'. Use 'plain', 'diag', or 'full'."
        )

    n, k = jac_raw.shape
    if k == 1:
        return g / (jac_raw * jac_raw + damp)
    if k == 2:
        g00 = jac_raw[:, 0] * jac_raw[:, 0] + damp
        g11 = jac_raw[:, 1] * jac_raw[:, 1] + damp
        g01 = jac_raw[:, 0] * jac_raw[:, 1]
        det = g00 * g11 - g01 * g01
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        d0 = (g11 * g[:, 0] - g01 * g[:, 1]) / det
        d1 = (g00 * g[:, 1] - g01 * g[:, 0]) / det
        return np.stack([d0, d1], axis=1)

    # General K: batched (J J^T + damp I)^{-1} g via small dense solves
    eye = np.eye(k, dtype=np.float64)
    out = np.empty_like(g)
    for i in range(n):
        ji = jac_raw[i]
        gmat = np.outer(ji, ji) + damp * eye
        try:
            out[i] = np.linalg.solve(gmat, g[i])
        except np.linalg.LinAlgError:
            out[i] = np.linalg.solve(gmat + 1e-6 * eye, g[i])
    return out


def _fit_global_raw(
    formula: Callable,
    links: list[str],
    x: NDArray,
    y: NDArray,
    sample_weight: NDArray | None,
) -> np.ndarray:
    """Fit a single global raw-parameter vector by L-BFGS-B."""
    from scipy.optimize import minimize

    k = len(links)
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    v0 = np.zeros(k, dtype=np.float64)
    if links[0] == "log" and np.all(y > 0):
        v0[0] = float(np.log(np.average(y, weights=sample_weight)))

    def packed_loss(v: np.ndarray) -> float:
        theta = []
        for j, name in enumerate(links):
            fn, _ = _link_pair(name)
            raw_j = np.full_like(x, v[j])
            theta.append(fn(raw_j))
        pred = np.asarray(formula(tuple(theta), x), dtype=np.float64).ravel()
        if not np.all(np.isfinite(pred)):
            return 1e12
        err = 0.5 * (pred - y) ** 2
        return float(np.average(err, weights=sample_weight))

    result = minimize(packed_loss, v0, method="L-BFGS-B")
    return result.x if result.success else v0


class FormulaObjective:
    """Boost the parameters of a user formula ``f(theta, x)``.

    ``formula(theta, x)`` receives ``theta`` as a tuple of K arrays (constrained
    space) and ``x`` as the structural input (e.g. spend). Features ``Z`` that
    the trees split on never enter the formula — they only determine
    ``theta(Z)``.
    """

    def __init__(
        self,
        formula: Callable,
        n_params: int,
        links: tuple[str, ...] | list[str],
        *,
        loss: str = "mse",
        precond: str = "full",
        damp: float = 1.0,
        param_names: tuple[str, ...] | list[str] | None = None,
        fd_eps: float = 1e-5,
    ):
        if loss != "mse":
            raise ValueError("FormulaObjective currently supports loss='mse' only.")
        if len(links) != n_params:
            raise ValueError(
                f"links has length {len(links)}, expected n_params={n_params}."
            )
        for name in links:
            _link_pair(name)
        if precond not in ("plain", "diag", "full"):
            raise ValueError(
                f"Unknown precond '{precond}'. Use 'plain', 'diag', or 'full'."
            )

        self.formula = formula
        self.n_params = n_params
        self.links = list(links)
        self.loss = loss
        self.precond = precond
        self.damp = float(damp)
        self.fd_eps = float(fd_eps)
        if param_names is None:
            self.channel_names = [f"theta_{j}" for j in range(n_params)]
        else:
            if len(param_names) != n_params:
                raise ValueError("param_names length must equal n_params.")
            self.channel_names = list(param_names)

    @property
    def device_capable(self) -> bool:
        # Formula + GGN stay on host; the trainer still builds trees on GPU.
        return False

    @property
    def unit_hessian(self) -> bool:
        return True

    def _require_x(self, extra: dict[str, Any] | None) -> NDArray:
        if extra is None or extra.get("model_input") is None:
            raise ValueError(
                "FormulaObjective requires extra['model_input'] "
                "(the structural input x that enters the formula)."
            )
        return np.asarray(extra["model_input"], dtype=np.float64).ravel()

    def init_raw(
        self,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        x = self._require_x(extra)
        v = _fit_global_raw(self.formula, self.links, x, y, sample_weight)
        return {name: float(v[j]) for j, name in enumerate(self.channel_names)}

    def constrain(
        self,
        raw: RawScores,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, NDArray]:
        params = {}
        for name, link_name in zip(self.channel_names, self.links, strict=True):
            fn, _ = _link_pair(link_name)
            params[name] = fn(np.asarray(raw[name], dtype=np.float64))
        return params

    def _predict_f(self, raw: RawScores, extra: dict[str, Any] | None) -> NDArray:
        x = self._require_x(extra)
        params = self.constrain(raw, extra)
        theta = [params[name] for name in self.channel_names]
        return np.asarray(self.formula(tuple(theta), x), dtype=np.float64).ravel()

    def step(
        self,
        raw: RawScores,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> GradHess:
        x = self._require_x(extra)
        y = np.asarray(y, dtype=np.float64).ravel()
        theta = []
        link_prime = []
        for name, link_name in zip(self.channel_names, self.links, strict=True):
            fn, dfn = _link_pair(link_name)
            r = np.asarray(raw[name], dtype=np.float64).ravel()
            theta.append(fn(r))
            link_prime.append(dfn(r))

        f, jac_theta = _formula_and_jac(self.formula, theta, x, eps=self.fd_eps)
        jac_raw = jac_theta * np.stack(link_prime, axis=1)
        residual = f - y
        direction = _ggn_step(residual, jac_raw, self.precond, self.damp)

        n = y.shape[0]
        ones = np.ones(n, dtype=np.float32)
        grads: GradHess = {}
        for j, name in enumerate(self.channel_names):
            grads[name] = (direction[:, j].astype(np.float32), ones.copy())
        return _apply_sample_weight(grads, sample_weight)

    def loss_value(
        self,
        raw: RawScores,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> float:
        pred = self._predict_f(raw, extra)
        y = np.asarray(y, dtype=np.float64).ravel()
        return float(np.average(0.5 * (pred - y) ** 2, weights=sample_weight))


# =============================================================================
# Weibull AFT objective (survival / censored regression)
# =============================================================================

# NGBoost's built-in families have no censored likelihood, and XGBoost's
# survival:aft objective learns only the location while holding the
# distribution scale (shape) as a single global hyperparameter. WeibullAFT
# boosts BOTH the scale lambda(z) and the shape k(z) surfaces from a censored
# NLL, which is the capability neither can express.
#
# Parameterization (both channels live in log space; that is the raw score):
#   u = raw["scale"] = log(lambda),  lambda = exp(u)
#   w = raw["shape"] = log(k),        k = exp(w)
# For time t with event indicator delta in {0, 1} (1 = observed, 0 = right
# censored), with m = log(t) - u and z = (t/lambda)^k = exp(k * m):
#   NLL = -delta * (w - u + (k - 1) * m) + z
# (i.e. -delta * log hazard + cumulative hazard).


class WeibullAFTObjective:
    """Boost Weibull scale ``lambda(z)`` and shape ``k(z)`` under a censored NLL.

    ``extra['event']`` is the event indicator (1 observed, 0 right-censored);
    defaults to all-observed. Preconditioning is a damped, PD-safeguarded
    per-sample 2x2 observed-information solve (Newton-natural step), sharing
    the trainer's unit-hessian / GPU-tree path with FormulaBoost.
    """

    _W_CLIP = 12.0  # clip log-shape to keep k = exp(w) finite
    _Z_CLIP = 60.0  # clip k*m before exp
    _EULER = 0.5772156649015329
    # Expected Fisher info of the log-shape channel for an observed Weibull
    # event (constant in the log parameterization): (1-gamma)^2 + pi^2/6.
    _I_WW = (1.0 - _EULER) ** 2 + (np.pi ** 2) / 6.0

    def __init__(self, *, damp: float = 1.0):
        self.channel_names = ["scale", "shape"]
        self.damp = float(damp)

    @property
    def device_capable(self) -> bool:
        return False

    @property
    def unit_hessian(self) -> bool:
        return True

    def _event(self, y: NDArray, extra: dict[str, Any] | None) -> NDArray:
        n = len(y)
        if extra is None or extra.get("event") is None:
            return np.ones(n, dtype=np.float64)
        e = np.asarray(extra["event"], dtype=np.float64).ravel()
        if e.shape[0] != n:
            raise ValueError(
                f"event has length {e.shape[0]}, expected {n} (matching y)."
            )
        return e

    def _pieces(self, u, w, t):
        """Shared intermediates for grad/hess/nll."""
        u = np.asarray(u, dtype=np.float64).ravel()
        w = np.clip(np.asarray(w, dtype=np.float64).ravel(), -self._W_CLIP, self._W_CLIP)
        k = np.exp(w)
        m = np.log(t) - u
        z = np.exp(np.clip(k * m, -self._Z_CLIP, self._Z_CLIP))
        return k, m, z

    def init_raw(
        self,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        from scipy.optimize import minimize

        t = np.asarray(y, dtype=np.float64).ravel()
        if np.any(t <= 0):
            raise ValueError("WeibullAFT requires strictly positive times y.")
        delta = self._event(t, extra)

        def mean_nll(v):
            u, w = np.full_like(t, v[0]), np.full_like(t, v[1])
            k, m, z = self._pieces(u, w, t)
            nll = -delta * (w - u + (k - 1.0) * m) + z
            if not np.all(np.isfinite(nll)):
                return 1e12
            return float(np.average(nll, weights=sample_weight))

        v0 = np.array([float(np.log(np.median(t))), 0.0])
        res = minimize(mean_nll, v0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
        v = res.x if res.success else v0
        return {"scale": float(v[0]), "shape": float(v[1])}

    def constrain(
        self,
        raw: RawScores,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, NDArray]:
        u = np.asarray(raw["scale"], dtype=np.float64).ravel()
        w = np.clip(np.asarray(raw["shape"], dtype=np.float64).ravel(),
                    -self._W_CLIP, self._W_CLIP)
        return {"scale": np.exp(u), "shape": np.exp(w)}

    def step(
        self,
        raw: RawScores,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> GradHess:
        t = np.asarray(y, dtype=np.float64).ravel()
        delta = self._event(t, extra)
        u = np.asarray(raw["scale"], dtype=np.float64).ravel()
        w = np.asarray(raw["shape"], dtype=np.float64).ravel()
        k, m, z = self._pieces(u, w, t)

        # Gradient of the censored NLL w.r.t. raw (u = log-scale, w = log-shape).
        g_u = k * (delta - z)
        g_w = -delta * (1.0 + m * k) + k * m * z

        # Damped natural gradient: precondition with the EXPECTED Fisher
        # information (not the observed Hessian, whose scale term k^2 z blows
        # up when lambda is wrong and freezes the update). In the log
        # parameterization the per-observed-event Fisher is
        #   [[k^2, -k(1-gamma)], [-k(1-gamma), (1-gamma)^2 + pi^2/6]].
        off = -k * (1.0 - self._EULER)
        a = k * k + self.damp
        c = self._I_WW + self.damp
        b = off
        det = a * c - b * b

        d_u = (c * g_u - b * g_w) / det
        d_w = (a * g_w - b * g_u) / det

        ones = np.ones(len(t), dtype=np.float32)
        grads: GradHess = {
            "scale": (d_u.astype(np.float32), ones.copy()),
            "shape": (d_w.astype(np.float32), ones.copy()),
        }
        return _apply_sample_weight(grads, sample_weight)

    def loss_value(
        self,
        raw: RawScores,
        y: NDArray,
        sample_weight: NDArray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> float:
        t = np.asarray(y, dtype=np.float64).ravel()
        delta = self._event(t, extra)
        u = np.asarray(raw["scale"], dtype=np.float64).ravel()
        w = np.asarray(raw["shape"], dtype=np.float64).ravel()
        k, m, z = self._pieces(u, w, t)
        nll = -delta * (w - u + (k - 1.0) * m) + z
        return float(np.average(nll, weights=sample_weight))
