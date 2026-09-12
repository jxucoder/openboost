"""Independent rounded-operation oracle for explicit histogram-level choice."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from openboost import NumericData, Problem, ops
from openboost.binning import Binning
from openboost.stats import newton


def fixture(values, gradient, curvature=None, weight=None):
    values = np.asarray(values, dtype=float)
    data = NumericData(
        values, np.arange(len(values)) + 71, tuple(f"f{i}" for i in range(values.shape[1]))
    )
    problem = Problem(data, np.zeros((len(values), 1)), data.row_ids, weight=weight)
    binned = Binning.fit(data, bins=min(8, len(values))).transform(data)
    fields = newton(problem, gradient, np.ones(len(values)) if curvature is None else curvature)
    return binned, fields


def rounded_oracle(hist, regularizer=1.0, minimum=0.0, penalty=0.0):
    """Scalar binary64 additions, divisions and multiplications; no production scoring."""
    g, h = hist.fields.names.index("gradient"), hist.fields.names.index("curvature")

    def node(row):
        leaf = -float(row[g]) / (float(row[h]) + regularizer)
        return (-0.5 * float(row[g])) * leaf

    records = []
    for feature, sums in enumerate(hist.sums):
        for threshold in sorted(set(hist.data.codes[feature, ~hist.data.missing[feature]])):
            left = [0.0] * len(hist.fields.names)
            right = [0.0] * len(hist.fields.names)
            # cumsum starts with the first element, including its signed zero.
            for i in range(int(threshold) + 1):
                left = [
                    float(v) if i == 0 else a + float(v) for a, v in zip(left, sums[i], strict=True)
                ]
            for i in reversed(range(int(threshold) + 1, len(sums) - 1)):
                right = [
                    float(v) if i == len(sums) - 2 else a + float(v)
                    for a, v in zip(right, sums[i], strict=True)
                ]
            count = sum(map(int, hist.counts[feature][: threshold + 1]))
            for missing in (False, True):
                a = [
                    v + (float(m) if missing else 0.0) for v, m in zip(left, sums[-1], strict=True)
                ]
                b = [
                    v + (0.0 if missing else float(m)) for v, m in zip(right, sums[-1], strict=True)
                ]
                n = count + (int(hist.counts[feature][-1]) if missing else 0)
                if min(n, len(hist.rows) - n) <= 0 or min(a[h], b[h]) <= 0:
                    continue
                if min(a[h], b[h]) < minimum:
                    continue
                gain = ((node(a) + node(b)) - node(hist.total)) - penalty
                records.append(((feature, int(threshold), missing), gain, a, b))
    positive = [r for r in records if r[1] > 0]
    return min(positive, key=lambda r: (-r[1], r[0])) if positive else None


def assert_choice(hist, **kwargs):
    expected = rounded_oracle(
        hist,
        kwargs.get("reg_lambda", 1.0),
        kwargs.get("min_child_h", 0.0),
        kwargs.get("split_penalty", 0.0),
    )
    result = ops.newton_choice(hist, **kwargs)
    if expected is None:
        assert result is None
        return
    candidate, gain = result
    assert candidate.key == expected[0]
    assert float(gain).hex() == expected[1].hex()
    assert candidate.left.tobytes() == np.asarray(expected[2], dtype="<f8").tobytes()
    assert candidate.right.tobytes() == np.asarray(expected[3], dtype="<f8").tobytes()
    assert candidate.parent.tobytes() == hist.total.tobytes()
    for value in (candidate.left, candidate.right, candidate.parent):
        with pytest.raises(ValueError):
            value.flags.writeable = True
    legacy = ops.choose(
        ops.candidates(hist),
        scoring=partial(
            ops.score,
            reg_lambda=kwargs.get("reg_lambda", 1.0),
            split_penalty=kwargs.get("split_penalty", 0.0),
        ),
        legality=partial(ops.feasible, min_child_h=kwargs.get("min_child_h", 0.0)),
    )
    assert legacy.key == candidate.key


@pytest.mark.parametrize("epsilon,winner", [(0.0, 0), (2.0**-53, 1)])
def test_exact_tie_and_tiny_positive_non_tie(epsilon, winner):
    # First feature groups (-1,+eps); second groups (-1,-eps).
    b, f = fixture([[0, 0], [1, 0], [0, 1], [1, 1]], [-1, -epsilon, epsilon, 1])
    hist = ops.histogram(b, f)
    expected = rounded_oracle(hist, regularizer=0.0)
    assert expected[0] == (winner, 0, False)
    assert_choice(hist, reg_lambda=0.0)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("rows", [None, [], [0, 2, 5, 9, 12, 17]])
@pytest.mark.parametrize("regularizer,minimum,penalty", [(0.0, 0.0, 0.0), (10.0, 2.0, 0.3)])
def test_independent_rounded_histogram_enumeration(seed, rows, regularizer, minimum, penalty):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, (24, 4)).astype(float)
    x[rng.random(x.shape) < 0.2] = np.nan
    x[:, 3] = np.nan
    g = rng.normal(size=24) * np.exp2(rng.integers(-25, 26, 24))
    b, f = fixture(x, g, rng.uniform(0.1, 3, 24), rng.integers(0, 4, 24))
    assert_choice(
        ops.histogram(b, f, rows),
        reg_lambda=regularizer,
        min_child_h=minimum,
        split_penalty=penalty,
    )


def test_missing_only_split_zero_mass_and_information_roles():
    b, f = fixture([[1], [1], [np.nan], [np.nan]], [-2, -2, 2, 2], weight=[0, 1, 1, 0])
    f = f.add_independent("cohort", [1, 1, 1, 1])
    hist = ops.histogram(b, f)
    assert_choice(hist)
    candidate, gain = ops.newton_choice(hist, min_information={"cohort": 2})
    assert candidate.key == (0, 0, False) and gain == ops.score(candidate)
    with pytest.raises(ValueError, match="independent"):
        ops.newton_choice(hist, min_information={"curvature": 1})


@pytest.mark.parametrize(
    "kwargs", [dict(reg_lambda=-1), dict(min_child_h=np.nan), dict(split_penalty=np.inf)]
)
def test_invalid_configuration_uses_original_rejection(kwargs):
    b, f = fixture([[0], [1]], [-1, 1])
    with pytest.raises(ValueError):
        ops.newton_choice(ops.histogram(b, f), **kwargs)


def test_nonfinite_leaf_and_candidate_rejections():
    b, f = fixture([[0], [1]], [-1e308, 1e308], [1e-300, 1e-300])
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="nonfinite Newton leaf"):
        ops.newton_choice(ops.histogram(b, f), reg_lambda=0.0)
    b, f = fixture([[0], [1], [2]], [-1, 0, 1])
    hist = ops.histogram(b, f)
    sums = hist.sums[0].copy()
    sums[:3, 0] = 1e308
    broken = replace(hist, sums=(sums,))
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        ops.newton_choice(broken)


def test_no_candidate_keeps_lazy_configuration_behavior():
    b, f = fixture([[np.nan], [np.nan]], [-1, 1])
    assert ops.newton_choice(ops.histogram(b, f), reg_lambda=-1) is None
