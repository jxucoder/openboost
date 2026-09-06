"""Independent auxiliary metrics; diagnostics never replace primary task gates."""

import math

import numpy as np

from benchmarks.v1.quality import metrics


def _weights(rows, weight):
    w = np.ones(rows) if weight is None else np.asarray(weight, dtype=float)
    if w.shape != (rows,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid evaluation weights")
    return w


def classification(application, target, prediction, weight=None):
    primary = metrics(application, target, prediction, weight=weight)
    if application not in ["A2", "A3"]:
        raise ValueError("classification application required")
    y, p = np.asarray(target), np.asarray(prediction)
    if y.ndim != 1:
        raise ValueError("scalar class IDs required")
    if application == "A2":
        p = np.column_stack([1 - p, p])
    w = _weights(len(y), weight)
    selected = np.argmax(p, axis=1)
    one_hot = np.eye(p.shape[1])[y.astype(int)]
    brier = (p[:, 1] - y) ** 2 if application == "A2" else ((p - one_hot) ** 2).sum(axis=1)
    classes = {}
    for k in range(p.shape[1]):
        positive, chosen = y == k, selected == k
        mass, predicted_mass = w[positive].sum(), w[chosen].sum()
        true_positive = w[positive & chosen].sum()
        classes[str(k)] = dict(
            rows=int(positive.sum()),
            weight=float(mass),
            recall=float(true_positive / mass) if mass else None,
            precision=float(true_positive / predicted_mass) if predicted_mass else None,
        )
    auc = None
    if application == "A2":
        pos_mass, neg_mass = w[y == 1].sum(), w[y == 0].sum()
        if pos_mass and neg_mass:
            # Ascending score groups give half credit to positive/negative ties.
            order = np.argsort(p[:, 1], kind="stable")
            score, yy, ww = p[order, 1], y[order], w[order]
            starts = np.r_[0, np.flatnonzero(score[1:] != score[:-1]) + 1, len(y)]
            below, numerator = 0.0, 0.0
            for a, b in zip(starts[:-1], starts[1:], strict=True):
                positive = ww[a:b][yy[a:b] == 1].sum()
                negative = ww[a:b][yy[a:b] == 0].sum()
                numerator += positive * (below + 0.5 * negative)
                below += negative
            auc = float(numerator / (pos_mass * neg_mass))
    return dict(
        **primary,
        accuracy=float(np.average(selected == y, weights=w)),
        brier=float(np.average(brier, weights=w)),
        binary_auc=auc,
        classes=classes,
    )


def normal_pit(target, prediction, weight=None):
    result = metrics("A11", target, prediction, weight=weight)
    y, p = np.asarray(target), np.asarray(prediction)
    z = (y - p[:, 0]) / p[:, 1]
    pit = np.array([0.5 * math.erfc(-v / math.sqrt(2)) for v in z])
    w = _weights(len(y), weight)
    result["pit_decile_mass"] = (
        np.histogram(pit, bins=np.linspace(0, 1, 11), weights=w)[0] / w.sum()
    ).tolist()
    return result


def survival(time, event, prediction, support, weight=None):
    """Log-normal IPCW Brier on a frozen training censoring grid, plus Harrell C.

    G is evaluated as a right-continuous step, matching the documented Brier
    convention. No G values beyond the frozen support are extrapolated. Harrell
    C uses unit pairs and is not an IPCW estimate; weighted C is rejected.
    """
    t, e, p = np.asarray(time, dtype=float), np.asarray(event), np.asarray(prediction, dtype=float)
    result = metrics("A10", t, p, event=e, weight=weight)
    w = _weights(len(t), weight)
    if not np.all(w == 1):
        raise ValueError("survival auxiliary contract currently requires unit row weights")
    if set(support) != {"times", "survival", "grid", "strict_upper", "tie_rule"}:
        raise ValueError("invalid censoring support fields")
    times = np.asarray(support["times"], dtype=float)
    g = np.asarray(support["survival"], dtype=float)
    grid = np.asarray(support["grid"], dtype=float)
    upper = support["strict_upper"]
    if (
        times.ndim != 1
        or not len(times)
        or g.shape != times.shape
        or not np.isfinite(times).all()
        or np.any(times <= 0)
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(g).all()
        or np.any((g < 0) | (g > 1))
        or np.any(np.diff(g) > 0)
    ):
        raise ValueError("invalid training censoring curve")
    expected_upper = times[np.flatnonzero(g == 0)[0]] if np.any(g == 0) else times[-1]
    if upper != expected_upper or support.get("tie_rule") != "events removed before censoring risk":
        raise ValueError("censoring support/tie contract differs")
    if (
        grid.ndim != 1
        or not len(grid)
        or not np.isfinite(grid).all()
        or np.any(grid <= 0)
        or np.any(np.diff(grid) <= 0)
        or np.any(grid >= upper)
    ):
        raise ValueError("grid outside training censoring support")

    def G(value):
        index = np.searchsorted(times, value, side="right") - 1
        return np.where(index < 0, 1.0, g[np.maximum(index, 0)])

    scores = []
    contributing = []
    for point in grid:
        z = (math.log(point) - p[:, 0]) / p[:, 1]
        s = np.array([0.5 * math.erfc(v / math.sqrt(2)) for v in z])
        deaths = (t <= point) & (e == 1)
        alive = t > point
        denominator = G(t[deaths])
        gt = float(G(point))
        if np.any(denominator <= 0) or gt <= 0 or not np.any(deaths | alive):
            raise ValueError("no supported IPCW contributions")
        values = np.zeros(len(t))
        values[deaths] = s[deaths] ** 2 / denominator
        values[alive] = (1 - s[alive]) ** 2 / gt
        scores.append(float(values.mean()))
        contributing.append(int(np.sum(deaths | alive)))
    # Unit comparable pairs: event precedes later observation, including an event
    # tied with a censor. Two tied deaths do not form an ordered pair.
    risk = -p[:, 0]
    concordant = 0.0
    comparable = 0
    tied = 0
    for i in np.flatnonzero(e):
        js = np.flatnonzero((t > t[i]) | ((t == t[i]) & (e == 0)))
        delta = risk[i] - risk[js]
        ties = np.abs(delta) <= 1e-8
        concordant += float(np.sum(delta > 1e-8)) + 0.5 * float(ties.sum())
        comparable += len(js)
        tied += int(ties.sum())
    result.update(
        grid=grid.tolist(),
        ipcw_brier=scores,
        contributing_rows=contributing,
        integrated_brier=float(np.trapezoid(scores, grid) / (grid[-1] - grid[0]))
        if len(grid) > 1
        else None,
        harrell_c=concordant / comparable if comparable else None,
        comparable_pairs=comparable,
        tied_risk_pairs=tied,
    )
    return result


def structure_errors(age, target, prediction, train_min, train_max, weight=None):
    metrics("A12", target, prediction, weight=weight)
    age, y, p = np.asarray(age, dtype=float), np.asarray(target), np.asarray(prediction)
    if (
        age.shape != y.shape
        or not np.isfinite(age).all()
        or np.any(age <= 0)
        or not np.isfinite([train_min, train_max]).all()
        or not 0 < train_min <= train_max
    ):
        raise ValueError("invalid structural support")
    w = _weights(len(y), weight)
    result = {}
    for name, mask in dict(
        below=age < train_min, inside=(age >= train_min) & (age <= train_max), above=age > train_max
    ).items():
        result[name] = dict(
            rows=int(mask.sum()),
            rmse=float(np.sqrt(np.average((y[mask] - p[mask]) ** 2, weights=w[mask])))
            if w[mask].sum() > 0
            else None,
        )
    return result


def paired_interval(candidate, baseline):
    """Exact five-fold empirical bootstrap of mean paired differences, descriptive only."""
    c, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if c.shape != (5,) or b.shape != (5,) or not np.isfinite([c, b]).all():
        raise ValueError("five finite paired folds required")
    delta = c - b
    indices = np.indices((5,) * 5).reshape(5, -1)
    means = delta[indices].mean(axis=0)
    return dict(
        differences=delta.tolist(),
        mean=float(delta.mean()),
        percentile95=np.quantile(means, [0.025, 0.975]).tolist(),
        interpretation="descriptive empirical fold bootstrap; overlapping splits are not independent evidence",
    )
