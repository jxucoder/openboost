"""Independent prediction-space metrics and preregistered five-fold E3 comparisons.

This module does not import OpenBoost or accept producer-supplied quality claims.
Full gate acceptance also requires artifact integrity, row identity, and coverage.
"""

import math

import numpy as np


def _finite(value):
    a = np.asarray(value, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("nonempty finite values required")
    return a


def _log_survival(z):
    if z < 8:
        return math.log(math.erfc(z / math.sqrt(2)) / 2)
    # Asymptotic Mills expansion; switch far enough into the tail for accuracy.
    # Use erfc while representable, asymptotics only after underflow.
    tail = math.erfc(z / math.sqrt(2)) / 2
    if tail > 0:
        return math.log(tail)
    inv = 1 / (z * z)
    series = 1 - inv + 3 * inv**2 - 15 * inv**3 + 105 * inv**4
    return -0.5 * z * z - 0.5 * math.log(2 * math.pi) - math.log(z) + math.log(series)


def metrics(
    application, target, prediction, *, weight=None, event=None, query=None, row_ids=None, power=1.5
):
    y, p = _finite(target), _finite(prediction)
    if y.ndim not in (1, 2) or p.ndim not in (1, 2) or p.shape[0] != len(y):
        raise ValueError("unaligned target/prediction")
    w = np.ones(len(y)) if weight is None else _finite(weight)
    if w.shape != (len(y),) or np.any(w < 0) or not w.sum() > 0:
        raise ValueError("invalid weights")

    def mean(a):
        return float(np.dot(w, a) / w.sum())

    result = {}
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        if application in ("A1", "A12", "A6"):
            if p.shape != y.shape or (application == "A6" and y.ndim != 2):
                raise ValueError("wrong regression output schema")
            if y.ndim == 1:
                result = {"rmse": math.sqrt(mean((p - y) ** 2))}
            else:
                result = {
                    f"rmse_{k}": math.sqrt(mean((p[:, k] - y[:, k]) ** 2))
                    for k in range(y.shape[1])
                }
        elif application == "A2":
            if p.shape != y.shape or not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
                raise ValueError("binary labels/probabilities required")
            # Freeze epsilon for exact boundary probabilities, never accept logits here.
            q = np.clip(p, 1e-15, 1 - 1e-15)
            result = {"logloss": mean(-y * np.log(q) - (1 - y) * np.log1p(-q))}
        elif application == "A3":
            if (
                y.ndim != 1
                or p.ndim != 2
                or p.shape[1] < 2
                or np.any(y != np.floor(y))
                or np.any((y < 0) | (y >= p.shape[1]))
            ):
                raise ValueError("invalid multiclass schema")
            if np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-7):
                raise ValueError("probabilities must sum to one")
            result = {
                "logloss": mean(-np.log(np.clip(p[np.arange(len(y)), y.astype(int)], 1e-15, 1)))
            }
        elif application == "A5":
            if y.ndim != 1 or p.shape != (len(y), 3):
                raise ValueError("three declared quantile columns required")
            for k, q in enumerate([0.1, 0.5, 0.9]):
                r = y - p[:, k]
                result[f"pinball_{q}"] = mean(np.maximum(q * r, (q - 1) * r))
            result["crossing_rate"] = mean(np.any(np.diff(p, axis=1) < 0, axis=1))
        elif application in ("A7", "A8", "A9"):
            if p.shape != y.shape or y.ndim != 1 or np.any(p <= 0) or np.any(y < 0):
                raise ValueError("nonnegative scalar targets and positive means required")
            if application == "A7":
                if np.any(y != np.floor(y)):
                    raise ValueError("integer counts required")
                term = np.zeros_like(y)
                positive = y > 0
                term[positive] = y[positive] * np.log(y[positive] / p[positive])
                result = {"poisson_deviance": mean(2 * (term - y + p))}
            elif application == "A8":
                if np.any(y <= 0):
                    raise ValueError("positive severity required")
                ratio = y / p
                result = {"gamma_deviance": mean(2 * (ratio - 1 - np.log(ratio)))}
            else:
                if not 1 < power < 2:
                    raise ValueError("Tweedie power must lie between 1 and 2")
                dev = 2 * (
                    y ** (2 - power) / ((1 - power) * (2 - power))
                    - y * p ** (1 - power) / (1 - power)
                    + p ** (2 - power) / (2 - power)
                )
                result = {"tweedie_deviance": mean(dev)}
        elif application in ("A10", "A11"):
            if y.ndim != 1 or p.shape != (len(y), 2) or np.any(p[:, 1] <= 0):
                raise ValueError("location and positive standard deviation required")
            mu, sigma = p.T
            if application == "A11":
                z = (y - mu) / sigma
                cdf = np.array([0.5 * math.erfc(-v / math.sqrt(2)) for v in z])
                pdf = np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
                result = {
                    "nll": mean(np.log(sigma) + z * z / 2 + 0.5 * math.log(2 * math.pi)),
                    "crps": mean(sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))),
                    "coverage90": mean(np.abs(z) <= 1.6448536269514722),
                    "width90": mean(2 * 1.6448536269514722 * sigma),
                }
            else:
                e = np.asarray(event)
                if np.any(y <= 0) or e.shape != y.shape or not np.isin(e, [0, 1]).all():
                    raise ValueError("positive times and event indicators required")
                z = (np.log(y) - mu) / sigma
                loss = np.log(y) + np.log(sigma) + z * z / 2 + 0.5 * math.log(2 * math.pi)
                loss[e == 0] = [-_log_survival(v) for v in z[e == 0]]
                result = {"nll": mean(loss)}
        elif application == "A4":
            q = np.asarray(query)
            ids = np.arange(len(y)) if row_ids is None else np.asarray(row_ids)
            if (
                p.shape != y.shape
                or y.ndim != 1
                or q.shape != y.shape
                or ids.shape != y.shape
                or len(np.unique(ids)) != len(ids)
            ):
                raise ValueError("ranking requires aligned query and unique row IDs")
            if not np.all(w == 1) or np.any(y < 0) or np.any(y > 4) or np.any(y != np.floor(y)):
                raise ValueError("ranking uses unit query weights, relevance 0..4")
            values = []
            zero = 0
            for group in np.unique(q):
                rows = np.flatnonzero(q == group)
                order = rows[np.lexsort((ids[rows], -p[rows]))][:10]
                ideal = rows[np.lexsort((ids[rows], -y[rows]))][:10]
                discount = np.log2(np.arange(len(order)) + 2)
                idcg = np.sum(np.expm1(y[ideal] * math.log(2)) / discount)
                if idcg == 0:
                    values.append(1.0)
                    zero += 1
                else:
                    values.append(float(np.sum(np.expm1(y[order] * math.log(2)) / discount) / idcg))
            result = {"ndcg10": float(np.mean(values)), "zero_idcg_queries": float(zero)}
        else:
            raise ValueError(
                "unknown task; A13 must evaluate the validation-selected original task"
            )
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite metric")
    return result


def compare_folds(candidate, baseline, kind):
    """Apply E3 to exactly five aligned folds of one primary metric."""
    try:
        c, b = _finite(candidate), _finite(baseline)
        if c.shape != (5,) or b.shape != (5,):
            raise ValueError("exactly five folds required")
        if kind == "loss":
            if np.any(c < 0) or np.any(b < 0):
                raise ValueError("loss ratios require nonnegative values")
            near = b <= 1e-8
            ratio = c[~near] / b[~near]
            ok = bool(np.all(np.abs(c[near] - b[near]) <= 1e-8))
            if len(ratio):
                ok = ok and np.median(ratio) <= 1.05 and np.max(ratio) <= 1.15
            return {"pass": bool(ok), "ratios": ratio.tolist(), "near_perfect": int(near.sum())}
        if kind not in ["nll", "ndcg"]:
            raise ValueError("unknown metric comparison kind")
        difference = c - b if kind == "nll" else b - c
        med, worst = (0.02, 0.10) if kind == "nll" else (0.01, 0.03)
        return {
            "pass": bool(np.median(difference) <= med and np.max(difference) <= worst),
            "differences": difference.tolist(),
        }
    except (ValueError, TypeError, OverflowError) as exc:
        return {"pass": False, "reason": str(exc)}


def select_validation(records, *, maximize=False, expected=16):
    """Strict complete search; test predictions/scores are deliberately not inputs."""
    if len(records) != expected or len({r["id"] for r in records}) != expected:
        raise ValueError("missing/duplicate search trial")
    if any(
        set(r) != {"id", "validation", "status"}
        or r["status"] != "pass"
        or not math.isfinite(r["validation"])
        for r in records
    ):
        raise ValueError("failed or invalid search trial")
    return min(records, key=lambda r: ((-1 if maximize else 1) * r["validation"], r["id"]))["id"]
