"""Hand-derived training-only transformation cases for the future data layer."""

import numpy as np
import pytest

from .reference.data import CategoryMap, NumericBinning


def test_linear_cuts_and_boundary_routing_are_training_only():
    values = np.array([0.0, 2.0, 4.0, 6.0])
    fitted = NumericBinning.fit(values, bins=4)
    assert fitted.cuts == (1.5, 3.0, 4.5)
    values[:] = 100
    codes, missing = fitted.transform([-100, 1.5, 3, 4.5, 100, np.nan])
    assert codes == (0, 0, 1, 2, 3, 0)
    assert missing == (False, False, False, False, False, True)
    assert fitted.cuts == (1.5, 3.0, 4.5)


@pytest.mark.parametrize(
    "values,bins,cuts",
    [
        ([0, 0, 0, 1], 2, (0.0,)),
        ([2, 2, 2], 4, ()),
        ([np.nan, np.nan], 4, ()),
        ([0, 1], 1, ()),
        ([0, 0, 0, 1], 4, (0.0, 0.25)),
    ],
)
def test_degenerate_numeric_columns(values, bins, cuts):
    assert NumericBinning.fit(values, bins=bins).cuts == cuts


@pytest.mark.parametrize("values", [[np.inf], [-np.inf], [], [[1, 2]]])
def test_invalid_numeric_training(values):
    with pytest.raises(ValueError):
        NumericBinning.fit(values)


@pytest.mark.parametrize("bins", [0, -1, 1.5, True])
def test_invalid_bin_budget(bins):
    with pytest.raises(ValueError):
        NumericBinning.fit([0, 1], bins=bins)


def test_numeric_transform_rejects_inf_and_interpolation_does_not_overflow():
    fitted = NumericBinning.fit([-1e308, 1e308], bins=2)
    assert fitted.cuts == (0.0,)
    with pytest.raises(ValueError):
        fitted.transform([np.inf])


def test_categories_are_stable_training_only_and_route_by_equality():
    values = ["z", "a", "m", None, "a"]
    fitted = CategoryMap.fit(values)
    assert fitted.values == ("a", "m", "z")
    values[0] = "new"
    codes, missing = fitted.transform(["z", "a", "m", "unseen", None, float("nan")])
    assert codes == (2, 0, 1, 0, 0, 0)
    assert missing == (False, False, False, True, True, True)
    assert fitted.route(["a", "m", "z", None, "unseen"], "m", missing_left=True) == (
        False,
        True,
        False,
        True,
        True,
    )
    assert fitted.route(["a", "m", "z", None], "m", missing_left=False) == (
        False,
        True,
        False,
        False,
    )
    assert fitted.values == ("a", "m", "z")


def test_category_types_are_not_silently_coerced():
    assert CategoryMap.fit([10, -2, 10]).values == (-2, 10)
    assert CategoryMap.fit([None, np.nan]).values == ()
    for values in ([1, "1"], [True, 1], [1.5], [["a"]]):
        with pytest.raises(ValueError):
            CategoryMap.fit(values)
    with pytest.raises(ValueError):
        CategoryMap.fit(["a"]).route(["a"], "unknown", missing_left=False)


def test_middle_category_candidate_cannot_be_replaced_by_numeric_threshold():
    from .reference.scalar import node_score

    tokens = ["a", "a", "m", "m", "z", "z"]
    gradient = np.array([1.0, 1.0, -2.0, -2.0, 1.0, 1.0])
    fitted = CategoryMap.fit(tokens)
    gains = {}
    for category in fitted.values:
        left = np.array(fitted.route(tokens, category, missing_left=False))
        gains[category] = (
            node_score(sum(gradient[left]), sum(left))
            + node_score(sum(gradient[~left]), sum(~left))
            - node_score(sum(gradient), len(tokens))
        )
    assert max(gains, key=gains.get) == "m"
    assert gains["m"] == pytest.approx(64 / 15)
    # Neither threshold on sorted codes can isolate the middle category.
    codes = np.array(fitted.transform(tokens)[0])
    for threshold in (0, 1):
        left = codes <= threshold
        gain = node_score(sum(gradient[left]), sum(left)) + node_score(
            sum(gradient[~left]), sum(~left)
        )
        assert gain < gains["m"]
