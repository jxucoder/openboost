import numpy as np
import pytest
from benchmarks.v1.preprocessing import censoring_support, fit_encoder, fit_target_scale, transform


def test_test_categories_and_outliers_cannot_change_training_encoder():
    x = np.array([[1, np.nan], [3, np.nan]])
    enc = fit_encoder(x, {"kind": ["a", None]})
    transformed = transform(enc, [[100, 7]], {"kind": ["unseen"]})
    assert enc["median"] == [2, 0]
    assert enc["categories"] == {"kind": ["a"]}
    np.testing.assert_array_equal(transformed, [[100, 7, 0, 0, 0, 1]])
    with pytest.raises(ValueError):
        transform(enc, [[1, 2]], {})


def test_scaling_keeps_constant_output_invertible():
    scale = fit_target_scale([[1, 4], [3, 4]])
    assert scale == {"mean": [2.0, 4.0], "std": [1.0, 1.0], "constant": [False, True]}


def test_censoring_ties_and_supported_grid():
    r = censoring_support([1, 1, 2, 3], [1, 0, 1, 0])
    # Four at risk at t=1, one event removed, one censor among three.
    assert r["survival"][0] == pytest.approx(2 / 3)
    assert r["survival"][-1] == 0 and max(r["grid"]) < 3
