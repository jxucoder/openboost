"""Independent development oracles; also executable against installed wheels."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from openboost import MixedData, NumericData, Problem, RunContext
from openboost.binning import Binning
from openboost.leaves import ResidualView
from openboost.recipes import quantile, squared
from openboost.stats import newton
from openboost.tree import best_first, depthwise, symmetric


def optimum(residual, weight, q, penalty, anchor):
    """Enumerate every breakpoint and each interval's stationary candidate."""
    candidates = list(residual)
    edges = [-np.inf, *sorted(set(residual)), np.inf]
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mass_left = weight[residual <= low].sum()
        root = anchor + (q * weight.sum() - mass_left) / penalty
        if low <= root <= high:
            candidates.append(root)

    def loss(value):
        r = residual - value
        return np.dot(weight, np.maximum(q * r, (q - 1) * r)) + penalty * (value - anchor) ** 2 / 2

    return min(candidates, key=loss)


def run_checks(cohort, leaves, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    x = NumericData(np.arange(6)[:, None], np.arange(6), ("x",))
    p = Problem(x, np.zeros((6, 1)), x.row_ids)
    gradient = np.array([-6.0, 1, 1, 1, 1, 2])
    information = np.eye(2)[np.arange(6) % 2]
    binned = Binning.fit(x, bins=6).transform(x)
    choices = []
    for cut in range(5):
        left = binned.codes[0] <= cut
        right = ~left
        if np.all(information[left].sum(0) >= 1) and np.all(information[right].sum(0) >= 1):
            gain = 0.5 * (
                gradient[left].sum() ** 2 / (left.sum() + 1)
                + gradient[right].sum() ** 2 / (right.sum() + 1)
                - gradient.sum() ** 2 / 7
            )
            choices.append((gain, -cut))
    expected_cut = -max(choices)[1]
    cuts = []
    for grower in (depthwise, best_first, symmetric):
        learner = cohort.CohortLearner(p, information, grower=grower, max_depth=1)
        tree = learner(binned, newton(p, gradient, np.ones(6)))
        assert tree.threshold[0] == expected_cut
        assert (
            depthwise(binned, newton(p, gradient, np.ones(6)), max_depth=1).threshold[0]
            != expected_cut
        )
        cuts.append(int(tree.threshold[0]))
    # Information survives zero objective weights; it is never reweighted.
    weighted = replace(p, weight=[0, 1, 1, 1, 1, 1])
    learner = cohort.CohortLearner(weighted, information, max_depth=1)
    captured = []

    def capture(data, fields, **options):
        captured.append(fields)
        return depthwise(data, fields, **options)

    learner.grower = capture
    learner(binned, newton(weighted, gradient, np.ones(6)))
    np.testing.assert_array_equal(captured[0].values[:, -2:], information)
    try:
        learner(binned, newton(p, gradient, np.ones(6)))
    except ValueError:
        pass
    else:
        raise AssertionError("foreign problem accepted")
    grouped = NumericData([[0], [0], [0], [1], [1], [1]], np.arange(6), ("x",))
    impossible = Problem(grouped, p.target, grouped.row_ids)
    no_split = cohort.CohortLearner(impossible, np.eye(2)[[0, 0, 0, 1, 1, 1]])(
        Binning.fit(grouped, bins=2).transform(grouped), newton(impossible, gradient, np.ones(6))
    )
    assert no_split.feature.tolist() == [-1]

    rng = np.random.default_rng(123)
    errors = []
    for i in range(30):
        residual = rng.integers(-5, 6, 9).astype(float)
        weight = rng.integers(0, 5, 9).astype(float)
        q, penalty, anchor = (0.2, 0.5, 0.8)[i % 3], (0.3, 2.0, 20.0)[i % 3], 1.3
        view = ResidualView(np.arange(9), residual, weight)
        actual = leaves.PenalizedLeaves(q=q, penalty=penalty, anchor=anchor).solve(view)
        expected = optimum(residual, weight, q, penalty, anchor)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        smooth = penalty * (actual - anchor) - q * weight.sum()
        assert smooth + weight[residual < actual - 1e-9].sum() <= 1e-8
        assert smooth + weight[residual <= actual + 1e-9].sum() >= -1e-8
        stronger = leaves.PenalizedLeaves(q=q, penalty=penalty * 10, anchor=anchor).solve(view)
        assert abs(stronger - anchor) <= abs(actual - anchor) + 1e-10
        errors.append(abs(actual - expected))
    empty = ResidualView([1], [3.0], [0.0])
    assert leaves.PenalizedLeaves(anchor=2).solve(empty) == 2
    for penalty in [0, -1, np.nan]:
        try:
            leaves.PenalizedLeaves(penalty=penalty)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid penalty accepted")

    mixed = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [None, "b"], [5, "a"]],
        np.arange(6),
        ("x", "category"),
        ("numeric", "categorical"),
    )
    train = Problem(
        mixed, [[-3], [-1], [0], [2], [4], [7]], mixed.row_ids, weight=[1, 0, 2, 1, 3, 1]
    )
    d2 = squared(
        train,
        train,
        context=RunContext("d2", 7),
        rounds=3,
        learner=cohort.CohortLearner(train, information),
    )
    plugin = leaves.PenalizedLeaves(q=0.7, penalty=5, anchor=4)
    d3 = quantile(
        train, train, context=RunContext("d3", 7), rounds=3, q=0.7, max_depth=2, grower=plugin
    )
    plain = quantile(train, train, context=RunContext("d3", 7), rounds=3, q=0.7, max_depth=2)
    assert not np.allclose(d3.steps[1].raw_before, plain.steps[1].raw_before)
    for step, term in zip(d3.steps, d3.state.model.terms, strict=True):
        residual = train.target[:, 0] - step.raw_before[:, 0]
        tree = term.learner
        assert len(tree.value) > 1
        codes = tree.binning.transform(mixed)
        pending = [(0, np.arange(len(residual)))]
        while pending:
            node, rows = pending.pop()
            feature = tree.feature[node]
            if feature == -1:
                expected = optimum(residual[rows], train.weight[rows], 0.7, 5, 4)
                np.testing.assert_allclose(tree.value[node, 0], expected, atol=1e-10)
            else:
                left = np.where(
                    codes.missing[feature, rows],
                    tree.missing_left[node],
                    codes.codes[feature, rows] == tree.threshold[node]
                    if tree.binning.categories[feature] is not None
                    else codes.codes[feature, rows] <= tree.threshold[node],
                )
                pending.extend(((tree.left[node], rows[left]), (tree.right[node], rows[~left])))
    records = {}
    for name, fit in (("d2", d2), ("d3", d3)):
        fit.state.model.save(destination / f"{name}.json")
        records[name] = fit.state.model.predict(mixed).tolist()
    payload = dict(
        values=mixed.values.tolist(),
        row_ids=mixed.row_ids.tolist(),
        names=list(mixed.feature_names),
        kinds=list(mixed.feature_kinds),
        predictions=records,
        d2_cuts=cuts,
        d3_max_absolute_error=max(errors),
        rounds=3,
    )
    (destination / "checks.json").write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    import sys

    import ob_cohort_splits
    import ob_penalized_leaves

    import openboost

    for module in (openboost, ob_cohort_splits, ob_penalized_leaves):
        assert "site-packages" in module.__file__, module.__file__
    run_checks(ob_cohort_splits, ob_penalized_leaves, sys.argv[1])
