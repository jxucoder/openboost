"""Integration coverage for unified models and the generic model loader."""

import numpy as np
import pytest

import openboost as ob


def linear_formula(theta, x):
    return theta[0] * x


@pytest.mark.parametrize("kind", ["formula", "survival"])
@pytest.mark.parametrize("mixed_features", [False, True])
def test_generic_load_unified_model(kind, mixed_features, tmp_path):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(48, 2)).astype(np.float32)
    x = rng.uniform(0.5, 2, len(X))
    y = np.exp(0.2 * X[:, 0]) * x
    if mixed_features:
        X[:, 1] = np.arange(len(X)) % 3
        X[::7, 0] = np.nan
        X = ob.array(X, categorical_features=[1])
    if kind == "formula":
        model = ob.FormulaBoost(
            formula=linear_formula, n_params=1, links=("log",),
            n_trees=2, max_depth=2,
        ).fit(X, y, model_input=x)
        predict_kwargs = {"model_input": x}
    else:
        model = ob.WeibullAFT(n_trees=2, max_depth=2).fit(
            X, y, event=(np.arange(len(y)) % 4 != 0).astype(float),
        )
        predict_kwargs = {}
    expected = model.predict(X, **predict_kwargs)
    path = tmp_path / f"{kind}.joblib"
    model.save(path)
    with pytest.warns(UserWarning, match="trusted"):
        loaded = ob.load(path)
    assert type(loaded) is type(model)
    np.testing.assert_array_equal(loaded.predict(X, **predict_kwargs), expected)
    for name, values in model.predict_params(X).items():
        np.testing.assert_array_equal(loaded.predict_params(X)[name], values)
