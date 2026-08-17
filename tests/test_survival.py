"""Tests for WeibullAFT: censored survival boosting on the unified trainer."""

from __future__ import annotations

import numpy as np
import pytest


def _weibull_data(n=6000, d=5, censor_q=0.7, seed=0, vary_shape=True):
    """Weibull AFT DGP with covariate-dependent scale (and optionally shape)."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (n, d))
    lam = np.exp(0.5 + 1.0 * Z[:, 0] - 0.8 * Z[:, 1])
    k = (np.exp(0.2 + 1.2 * Z[:, 2] - 0.6 * Z[:, 3]) if vary_shape
         else np.full(n, 1.3))
    u = rng.uniform(1e-9, 1.0, n)
    T = lam * (-np.log(u)) ** (1.0 / k)
    C = rng.exponential(np.quantile(T, censor_q), n)
    t = np.minimum(T, C)
    event = (T <= C).astype(float)
    return Z, t, event, lam, k


class TestWeibullAFT:
    def test_import(self):
        import openboost as ob

        assert hasattr(ob, "WeibullAFT")

    def test_fit_predict_shapes(self):
        from openboost import WeibullAFT

        Z, t, ev, *_ = _weibull_data(n=800, seed=1)
        m = WeibullAFT(n_trees=30, max_depth=3, learning_rate=0.1)
        m.fit(Z, t, event=ev)
        params = m.predict_params(Z)
        assert set(params) == {"scale", "shape"}
        assert params["scale"].shape == (len(t),)
        assert np.all(params["scale"] > 0)
        assert np.all(params["shape"] > 0)
        med = m.predict(Z)
        assert med.shape == (len(t),)
        assert np.all(np.isfinite(med)) and np.all(med > 0)

    def test_global_init_recovers_constant_params(self):
        """On covariate-free data the global MLE init should match truth."""
        from openboost._objectives import WeibullAFTObjective

        rng = np.random.default_rng(3)
        n = 20000
        lam_true, k_true = 2.5, 1.7
        u = rng.uniform(1e-9, 1.0, n)
        t = lam_true * (-np.log(u)) ** (1.0 / k_true)  # uncensored
        obj = WeibullAFTObjective()
        init = obj.init_raw(t, None, {"event": np.ones(n)})
        assert np.exp(init["scale"]) == pytest.approx(lam_true, rel=0.05)
        assert np.exp(init["shape"]) == pytest.approx(k_true, rel=0.05)

    def test_recovers_scale_and_shape_surfaces(self):
        """The capability: boost BOTH lambda(z) and k(z) from censored data."""
        from openboost import WeibullAFT

        Z, t, ev, lam, k = _weibull_data(n=6000, seed=0, vary_shape=True)
        m = WeibullAFT(n_trees=200, max_depth=3, learning_rate=0.1)
        m.fit(Z, t, event=ev)
        p = m.predict_params(Z)
        corr_scale = np.corrcoef(np.log(p["scale"]), np.log(lam))[0, 1]
        corr_shape = np.corrcoef(np.log(p["shape"]), np.log(k))[0, 1]
        assert corr_scale > 0.85
        assert corr_shape > 0.80  # shape varying with covariates is recovered

    def test_censoring_is_used(self):
        """Ignoring right-censoring biases survival time downward."""
        from openboost import WeibullAFT

        Z, t, ev, lam, k = _weibull_data(n=5000, seed=2, censor_q=0.5)
        kwargs = dict(n_trees=120, max_depth=3, learning_rate=0.1)
        with_cens = WeibullAFT(**kwargs).fit(Z, t, event=ev)
        ignore_cens = WeibullAFT(**kwargs).fit(Z, t)  # all treated as observed
        med_with = np.median(with_cens.predict(Z))
        med_ignore = np.median(ignore_cens.predict(Z))
        # Treating censored points as events underestimates the time.
        assert med_with > med_ignore

    def test_predict_quantile_and_survival_monotone(self):
        from openboost import WeibullAFT

        Z, t, ev, *_ = _weibull_data(n=1500, seed=4)
        m = WeibullAFT(n_trees=60, max_depth=3, learning_rate=0.1).fit(Z, t, event=ev)
        q10 = m.predict_quantile(Z, 0.1)
        q50 = m.predict_quantile(Z, 0.5)
        q90 = m.predict_quantile(Z, 0.9)
        assert np.all(q10 < q50) and np.all(q50 < q90)
        # Survival is a decreasing function of time.
        s_lo = m.predict_survival(Z, t=0.5)
        s_hi = m.predict_survival(Z, t=5.0)
        assert np.all(s_hi <= s_lo)
        assert np.all((s_lo >= 0) & (s_lo <= 1))

    def test_eval_set_and_early_stopping(self):
        from openboost import WeibullAFT

        Z, t, ev, *_ = _weibull_data(n=2000, seed=5)
        ntr = 1500
        m = WeibullAFT(n_trees=300, max_depth=3, learning_rate=0.2)
        m.fit(
            Z[:ntr], t[:ntr], event=ev[:ntr],
            eval_set=[(Z[ntr:], t[ntr:], ev[ntr:])],
            early_stopping_rounds=15,
        )
        assert "eval_0" in m.evals_result_
        assert len(m.evals_result_["eval_0"]["nll"]) < 300
        assert m.best_iteration_ < 300

    def test_positive_time_required(self):
        from openboost import WeibullAFT

        Z, t, ev, *_ = _weibull_data(n=200, seed=6)
        t = t.copy()
        t[0] = 0.0
        with pytest.raises(ValueError, match="positive"):
            WeibullAFT(n_trees=2).fit(Z, t, event=ev)

    def test_bad_event_length_raises(self):
        from openboost import WeibullAFT

        Z, t, ev, *_ = _weibull_data(n=200, seed=7)
        with pytest.raises(ValueError, match="event"):
            WeibullAFT(n_trees=2).fit(Z, t, event=ev[:100])
