"""CPU parametric comparison controls with explicit exposure and structure.

These are evaluation adapters, not OpenBoost production recipes. Objects are
trusted local Python bundles; model inference needs the pinned sklearn/SciPy stack.
"""

import warnings

import numpy as np


def _inputs(x, y, weight):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError("aligned nonempty scalar targets and matrix required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite inputs required")
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid business weights")
    return x, y, w


def _exposure(exposure, rows):
    e = np.asarray(exposure, dtype=float)
    if e.shape != (rows,) or not np.isfinite(e).all() or np.any(e <= 0):
        raise ValueError("positive finite exposure required")
    return e


def fit_glm(task, x, target, *, exposure=None, weight=None, alpha=0.1, max_iter=1000):
    """Fit counts (A7), individual positive payments (A8), or period totals (A9)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
    from sklearn.preprocessing import StandardScaler

    if task not in ["A7", "A8", "A9"] or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("invalid GLM task/penalty")
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError("positive convergence budget required")
    x, y, w = _inputs(x, target, weight)
    if np.any(y < 0) or (task == "A8" and np.any(y <= 0)):
        raise ValueError("invalid positive-family target")
    if task == "A7" and np.any(y != np.floor(y)):
        raise ValueError("integer counts required")
    if task == "A8":
        if exposure is not None:
            raise ValueError("severity has no exposure adjustment")
        model = GammaRegressor(alpha=alpha, max_iter=max_iter)
    else:
        e = _exposure(exposure, len(y))
        y, w = y / e, w * e
        model = (
            PoissonRegressor(alpha=alpha, max_iter=max_iter)
            if task == "A7"
            else TweedieRegressor(power=1.5, link="log", alpha=alpha, max_iter=max_iter)
        )
    if np.dot(y, w) <= 0:
        raise ValueError("positive weighted target mass required for log-link fit")
    scaler = StandardScaler().fit(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(x), y, sample_weight=w)
    return dict(task=task, scaler=scaler, model=model)


def predict_glm(saved, x, exposure=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite feature matrix required")
    rate = saved["model"].predict(saved["scaler"].transform(x))
    if not np.isfinite(rate).all() or np.any(rate <= 0):
        raise ValueError("invalid log-link mean")
    if saved["task"] == "A8":
        if exposure is not None:
            raise ValueError("severity has no exposure adjustment")
        return {"mean": rate}
    e = _exposure(exposure, len(x))
    return {"annualized": rate, "period": e * rate}


def fit_paid_composition(
    x,
    paid_count,
    period_total,
    exposure,
    claim_policy,
    claim_amount,
    *,
    weight=None,
    count_alpha=0.1,
    severity_alpha=0.1,
    max_iter=1000,
):
    """Use exactly the positive-payment records underlying each policy's total."""
    x, counts, w = _inputs(x, paid_count, weight)
    totals = np.asarray(period_total, dtype=float)
    index, amounts = np.asarray(claim_policy), np.asarray(claim_amount, dtype=float)
    if totals.shape != counts.shape or not np.isfinite(totals).all() or np.any(totals < 0):
        raise ValueError("invalid period totals")
    if (
        index.ndim != 1
        or index.dtype.kind not in "iu"
        or amounts.shape != index.shape
        or not len(index)
    ):
        raise ValueError("aligned paid-claim policy indices required")
    if (
        np.any(index < 0)
        or np.any(index >= len(x))
        or not np.isfinite(amounts).all()
        or np.any(amounts <= 0)
    ):
        raise ValueError("orphan or nonpositive paid claim")
    actual_counts = np.bincount(index, minlength=len(x))
    actual_totals = np.bincount(index, weights=amounts, minlength=len(x))
    if not np.array_equal(actual_counts, counts) or not np.allclose(
        actual_totals, totals, rtol=1e-12, atol=1e-8
    ):
        raise ValueError("counts/totals must come from these positive-payment records")
    frequency = fit_glm(
        "A7", x, counts, exposure=exposure, weight=w, alpha=count_alpha, max_iter=max_iter
    )
    severity = fit_glm(
        "A8", x[index], amounts, weight=w[index], alpha=severity_alpha, max_iter=max_iter
    )
    return dict(frequency=frequency, severity=severity)


def predict_paid_composition(saved, x, exposure):
    counts = predict_glm(saved["frequency"], x, exposure)
    severity = predict_glm(saved["severity"], x)["mean"]
    return {
        "annualized": counts["annualized"] * severity,
        "period": counts["period"] * severity,
        "paid_count": counts["period"],
        "severity": severity,
    }


def fit_global_formula(
    age, target, *, weight=None, initial_amplitude_multiplier=1.0, initial_rate=1.0, max_nfev=2000
):
    """Fit a global positive a,b in a*(1-exp(-b*age)); age is already days/28."""
    from scipy.optimize import least_squares

    age = np.asarray(age, dtype=float)
    _, y, w = _inputs(age[:, None], target, weight)
    if np.any(age <= 0) or np.any(y < 0) or np.dot(y, w) <= 0:
        raise ValueError("positive age and positive weighted target mass required")
    if (
        not np.isfinite([initial_amplitude_multiplier, initial_rate]).all()
        or min(initial_amplitude_multiplier, initial_rate) <= 0
    ):
        raise ValueError("positive formula initialization required")
    if type(max_nfev) is not int or max_nfev <= 0:
        raise ValueError("positive optimizer budget required")
    start = np.array(
        [max(float(np.average(y, weights=w)), 1e-6) * initial_amplitude_multiplier, initial_rate]
    )
    raw = start + np.log(-np.expm1(-start))

    def residual(theta):
        a, b = np.logaddexp(0, theta)
        return np.sqrt(w) * (a * -np.expm1(-b * age) - y)

    fitted = least_squares(residual, raw, max_nfev=max_nfev)
    a, b = np.logaddexp(0, fitted.x)
    if not fitted.success or not np.isfinite([a, b, fitted.cost]).all() or min(a, b) <= 0:
        raise ValueError("formula optimizer did not converge to finite positive parameters")
    return dict(
        amplitude=float(a),
        rate=float(b),
        nfev=fitted.nfev,
        train_age_min=float(age.min()),
        train_age_max=float(age.max()),
    )


def predict_global_formula(saved, age):
    age = np.asarray(age, dtype=float)
    if age.ndim != 1 or not np.isfinite(age).all() or np.any(age <= 0):
        raise ValueError("positive finite age vector required")
    a, b = saved["amplitude"], saved["rate"]
    if not np.isfinite([a, b]).all() or min(a, b) <= 0:
        raise ValueError("invalid formula state")
    return a * -np.expm1(-b * age)
