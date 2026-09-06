"""Count/positive-target formulas and tiny policy joins; no production model API."""

import math

import numpy as np

from .scalar import finite_vector, training_weights


def _positive(values, name, length=None):
    values = finite_vector(values, name, length)
    if np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return values


def _target(values, *, strictly_positive=False, count=False):
    y = finite_vector(values, "target")
    if np.any(y <= 0 if strictly_positive else y < 0) or count and np.any(y != np.floor(y)):
        raise ValueError("invalid target support")
    return y


def _exp(values, name):
    with np.errstate(over="ignore", under="ignore"):
        result = np.exp(values)
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{name} exponential is outside positive float64 range")
    return result


def _result(loss, g, h, weight):
    if not all(np.all(np.isfinite(v)) for v in (loss, g, h)):
        raise ValueError("non-finite objective geometry")
    w = training_weights(weight, len(g))
    aggregate = float(sum((w / sum(w)) * loss))
    if not math.isfinite(aggregate):
        raise ValueError("non-finite weighted objective")
    return aggregate, g, h


def poisson(raw, target, exposure, *, weight=None):
    y = _target(target, count=True)
    f = finite_vector(raw, "raw", len(y))
    e = _positive(exposure, "exposure", len(y))
    log_mu = f + np.log(e)
    mu = _exp(log_mu, "count mean")
    loss = mu - y * log_mu + np.array([math.lgamma(v + 1) for v in y])
    return _result(loss, mu - y, mu, weight)


def poisson_base(target, exposure, *, weight=None, minimum_rate):
    y = _target(target, count=True)
    e = _positive(exposure, "exposure", len(y))
    w = training_weights(weight, len(y))
    if not np.isscalar(minimum_rate) or not np.isfinite(minimum_rate) or minimum_rate <= 0:
        raise ValueError("minimum_rate must be finite and positive")
    numerator, denominator = sum(w * y), sum(w * e)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("invalid rate totals")
    if numerator == 0:
        return math.log(minimum_rate)
    return math.log(numerator) - math.log(denominator)


def gamma(raw, target, *, weight=None):
    y = _target(target, strictly_positive=True)
    f = finite_vector(raw, "raw", len(y))
    ratio = _exp(np.log(y) - f, "Gamma target/mean ratio")
    return _result(ratio + f, 1 - ratio, ratio, weight)


def tweedie(raw, target, *, power=1.5, weight=None):
    if not np.isscalar(power) or not np.isfinite(power) or not 1 < power < 2:
        raise ValueError("power must be finite and strictly between one and two")
    y = _target(target)
    f = finite_vector(raw, "raw", len(y))
    a = _exp((2 - power) * f, "Tweedie mean power")
    b = np.zeros(len(y))
    positive = y > 0
    b[positive] = _exp(np.log(y[positive]) + (1 - power) * f[positive], "Tweedie target term")
    return _result(
        b / (power - 1) + a / (2 - power), a - b, (2 - power) * a + (power - 1) * b, weight
    )


def policy_losses(policies, payments):
    """Return sorted (ID,e,total,annualized,paid_count,paid_mean) plus exclusions.

    Fixture IDs are strings; raw ClaimNb is only a consistency signal. It is
    never substituted for the count of positive payment records.
    """
    table, paid, excluded = {}, {}, []
    for policy, count, exposure in policies:
        if not isinstance(policy, str) or policy in table:
            raise ValueError("policy IDs must be unique strings")
        _target([count], count=True)
        _positive([exposure], "exposure")
        table[policy] = (count, exposure)
        paid[policy] = []
    for policy, amount in payments:
        if not np.isscalar(amount) or not np.isfinite(amount):
            raise ValueError("payment must be finite")
        if policy not in table:
            excluded.append((policy, "orphan_payment"))
        elif amount <= 0:
            excluded.append((policy, "nonpositive_payment"))
        else:
            paid[policy].append(float(amount))
    rows = []
    for policy in sorted(table):
        count, exposure = table[policy]
        values = paid[policy]
        if count > 0 and not values:
            excluded.append((policy, "count_without_payment"))
        elif count == 0 and values:
            excluded.append((policy, "payment_without_count"))
        else:
            total = math.fsum(values)
            annualized = total / exposure
            if not np.isfinite(total) or not np.isfinite(annualized):
                raise ValueError("policy total exceeds float64")
            rows.append(
                (
                    policy,
                    exposure,
                    total,
                    annualized,
                    len(values),
                    total / len(values) if values else 0.0,
                )
            )
    return tuple(rows), tuple(excluded)


def poisson_predict(raw, exposure):
    f = finite_vector(raw, "raw")
    e = _positive(exposure, "exposure", len(f))
    return {"rate": _exp(f, "rate"), "count_mean": _exp(f + np.log(e), "count mean")}


def gamma_base(target, *, weight=None):
    y = _target(target, strictly_positive=True)
    w = training_weights(weight, len(y))
    mean = sum((w / sum(w)) * y)
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("invalid Gamma mean")
    return math.log(mean)
