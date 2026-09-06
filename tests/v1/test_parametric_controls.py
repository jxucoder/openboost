import numpy as np
import pytest
from benchmarks.v1.parametric import (
    fit_global_formula,
    fit_paid_composition,
    predict_global_formula,
)


def test_global_formula_recovers_known_curve_and_monotonicity():
    x = np.linspace(0.1, 4, 40)
    y = 3 * -np.expm1(-0.7 * x)
    saved = fit_global_formula(x, y, weight=np.linspace(0.5, 2, 40))
    np.testing.assert_allclose([saved["amplitude"], saved["rate"]], [3, 0.7], rtol=1e-6)
    p = predict_global_formula(saved, x)
    np.testing.assert_allclose(p, y, atol=1e-7)
    assert np.all(np.diff(p) > 0)


@pytest.mark.parametrize(
    "counts,totals,index,amount",
    [
        ([2, 0], [10, 0], [0], [10]),
        ([1, 0], [11, 0], [0], [10]),
        ([1, 0], [10, 0], [2], [10]),
        ([1, 0], [0, 0], [0], [0]),
    ],
)
def test_paid_composition_rejects_mismatched_or_orphan_claims(counts, totals, index, amount):
    with pytest.raises(ValueError):
        fit_paid_composition(np.ones((2, 2)), counts, totals, [1, 1], index, amount)


def test_formula_rejects_invalid_structure_and_budget():
    with pytest.raises(ValueError):
        fit_global_formula(np.array([0.0, 1.0]), [1.0, 2.0])
    with pytest.raises(ValueError):
        fit_global_formula(np.array([1.0, 2.0]), [1.0, 2.0], max_nfev=0)
