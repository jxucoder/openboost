"""Tests for FormulaBoost and the unified trainer path it shares."""

from __future__ import annotations

import numpy as np
import pytest


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-np.clip(t, -30.0, 30.0)))


def sales_curve(theta, x):
    a, b = theta
    return a * x ** _sigmoid(b * x)


def _sales_data(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (n, 4))
    x = rng.uniform(0.3, 2.5, n)
    u = 0.4 + 0.8 * Z[:, 0] - 0.5 * Z[:, 1]
    a = np.exp(u)
    b = 0.6 + 1.5 * Z[:, 2]
    f = sales_curve((a, b), x)
    y = f + 0.05 * f.std() * rng.standard_normal(n)
    return Z, x, y, a, b, f


class TestFormulaBoost:
    def test_import(self):
        import openboost as ob

        assert hasattr(ob, "FormulaBoost")

    def test_fit_predict_shapes(self):
        from openboost import FormulaBoost

        Z, x, y, *_ = _sales_data(n=800, seed=1)
        model = FormulaBoost(
            formula=sales_curve,
            n_params=2,
            links=("log", "identity"),
            param_names=("a", "b"),
            n_trees=20,
            max_depth=3,
            learning_rate=0.1,
        )
        model.fit(Z, y, model_input=x)
        params = model.predict_params(Z)
        assert set(params) == {"a", "b"}
        assert params["a"].shape == (len(y),)
        assert np.all(params["a"] > 0)
        pred = model.predict(Z, model_input=x)
        assert pred.shape == (len(y),)
        assert np.all(np.isfinite(pred))

    def test_beats_global_and_recovers_a(self):
        """GGN boosting should recover the scale surface and beat a global curve."""
        from openboost import FormulaBoost

        Z, x, y, a, b, f = _sales_data(n=3_000, seed=2)
        n_tr = 2_400
        model = FormulaBoost(
            formula=sales_curve,
            n_params=2,
            links=("log", "identity"),
            param_names=("a", "b"),
            n_trees=80,
            max_depth=3,
            learning_rate=0.1,
            precond="full",
        )
        model.fit(Z[:n_tr], y[:n_tr], model_input=x[:n_tr])
        params = model.predict_params(Z[n_tr:])
        pred = model.predict(Z[n_tr:], model_input=x[n_tr:])
        rmse = float(np.sqrt(np.mean((pred - y[n_tr:]) ** 2)))
        global_rmse = float(np.sqrt(np.mean((y[n_tr:] - y[:n_tr].mean()) ** 2)))
        assert rmse < 0.5 * global_rmse
        assert np.corrcoef(params["a"], a[n_tr:])[0, 1] > 0.9

    def test_extrapolation_beats_blackbox(self):
        from openboost import FormulaBoost, GradientBoosting

        Z, x, y, a, b, f = _sales_data(n=3_000, seed=3)
        n_tr = 2_400
        model = FormulaBoost(
            formula=sales_curve,
            n_params=2,
            links=("log", "identity"),
            param_names=("a", "b"),
            n_trees=80,
            max_depth=3,
            learning_rate=0.1,
        )
        model.fit(Z[:n_tr], y[:n_tr], model_input=x[:n_tr])

        rng = np.random.default_rng(3)
        Z_ex = rng.uniform(0, 1, (600, 4))
        x_ex = rng.uniform(3.0, 5.0, 600)
        u = 0.4 + 0.8 * Z_ex[:, 0] - 0.5 * Z_ex[:, 1]
        a_ex = np.exp(u)
        b_ex = 0.6 + 1.5 * Z_ex[:, 2]
        f_ex = sales_curve((a_ex, b_ex), x_ex)

        fb_rmse = float(
            np.sqrt(np.mean((model.predict(Z_ex, model_input=x_ex) - f_ex) ** 2))
        )

        gb = GradientBoosting(n_trees=80, max_depth=6, learning_rate=0.1)
        gb.fit(np.column_stack([Z[:n_tr], x[:n_tr]]), y[:n_tr])
        gb_rmse = float(
            np.sqrt(np.mean((gb.predict(np.column_stack([Z_ex, x_ex])) - f_ex) ** 2))
        )
        assert fb_rmse < 0.5 * gb_rmse

    def test_plain_precond_worse_than_full(self):
        from openboost import FormulaBoost

        Z, x, y, *_ = _sales_data(n=1_500, seed=4)
        n_tr, n_te = 1_200, 300
        kwargs = dict(
            formula=sales_curve,
            n_params=2,
            links=("log", "identity"),
            n_trees=40,
            max_depth=3,
            learning_rate=0.1,
        )
        full = FormulaBoost(precond="full", **kwargs)
        plain = FormulaBoost(precond="plain", **kwargs)
        full.fit(Z[:n_tr], y[:n_tr], model_input=x[:n_tr])
        plain.fit(Z[:n_tr], y[:n_tr], model_input=x[:n_tr])
        rmse_full = np.sqrt(
            np.mean((full.predict(Z[-n_te:], model_input=x[-n_te:]) - y[-n_te:]) ** 2)
        )
        rmse_plain = np.sqrt(
            np.mean((plain.predict(Z[-n_te:], model_input=x[-n_te:]) - y[-n_te:]) ** 2)
        )
        assert rmse_full < rmse_plain

    def test_eval_set_and_early_stopping(self):
        from openboost import FormulaBoost

        Z, x, y, *_ = _sales_data(n=1_200, seed=5)
        model = FormulaBoost(
            formula=sales_curve,
            n_params=2,
            links=("log", "identity"),
            n_trees=200,
            max_depth=3,
            learning_rate=0.2,
        )
        model.fit(
            Z[:800],
            y[:800],
            model_input=x[:800],
            eval_set=[(Z[800:], y[800:], x[800:])],
            early_stopping_rounds=15,
        )
        assert "eval_0" in model.evals_result_
        assert len(model.evals_result_["eval_0"]["mse"]) < 200
        assert model.best_iteration_ < 200

    def test_bad_eval_set_raises(self):
        from openboost import FormulaBoost

        Z, x, y, *_ = _sales_data(n=100, seed=6)
        model = FormulaBoost(
            formula=sales_curve, n_params=2, links=("log", "identity"), n_trees=2
        )
        with pytest.raises(ValueError, match="eval_set"):
            model.fit(Z, y, model_input=x, eval_set=[(Z, y)])

    def test_unknown_link_raises(self):
        from openboost import FormulaBoost

        with pytest.raises(ValueError, match="Unknown link"):
            FormulaBoost(
                formula=sales_curve, n_params=2, links=("log", "relu")
            )._make_objective()


class TestTrainerNaturalBoostParity:
    """NaturalBoost still trains and predicts after the trainer port."""

    def test_naturalboost_fit_predict(self):
        from openboost import NaturalBoostNormal

        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4)).astype(np.float32)
        y = (X[:, 0] * 2 + rng.normal(size=300)).astype(np.float32)
        model = NaturalBoostNormal(n_trees=15, max_depth=3)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (300,)
        params = model.predict_params(X)
        assert params["scale"].min() > 0
        assert model.nll(X, y) < 3.0
