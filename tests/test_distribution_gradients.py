"""Correctness tests for distributional families (_distributions.py).

Verifies, in the style of test_loss_correctness.py:
1. Analytic gradients w.r.t. RAW (link-space) parameters match central finite
   differences of each family's nll evaluated through the link functions.
2. nll values match scipy.stats reference log-densities where available.
3. Tweedie predictive correctness: proper series log-density, support-aware
   quantiles, vectorized compound Poisson-Gamma sampling.
4. CustomDistribution: link chain rule (gradient AND second-order hessian
   term) on numerical and JAX paths, and the empirical Fisher approximation.
5. Target-support validation at fit entry.
6. End-to-end: mean NLL decreases over boosting rounds for NaturalBoost.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# All distribution classes need scipy
scipy = pytest.importorskip("scipy")

from scipy import stats  # noqa: E402
from scipy.special import gammaln  # noqa: E402

from openboost._distributions import (  # noqa: E402
    CustomDistribution,
    Gamma,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
    Tweedie,
)

# =============================================================================
# Finite-difference helpers
# =============================================================================


def _params_from_raw(dist, raw_params):
    """Apply link functions: raw -> constrained parameters."""
    return {
        name: dist.link(name, np.asarray(raw_params[name], dtype=np.float64))
        for name in dist.param_names
    }


def _fd_grad_raw(dist, y, raw_params, name, objective=None, eps=1e-6):
    """Central finite difference of the objective w.r.t. one RAW parameter.

    objective(y, constrained_params) -> per-sample values. Defaults to
    dist.nll, i.e. the finite difference is taken through the link function.
    """
    if objective is None:
        objective = dist.nll

    def obj_at(raw_val):
        raws = dict(raw_params)
        raws[name] = raw_val
        return np.asarray(objective(y, _params_from_raw(dist, raws)), dtype=np.float64)

    raw = np.asarray(raw_params[name], dtype=np.float64)
    return (obj_at(raw + eps) - obj_at(raw - eps)) / (2 * eps)


def _check_family_gradients(dist, y, raw_params, objective=None, rtol=2e-3, atol=2e-3):
    """Assert analytic raw-space gradients match finite differences."""
    y = np.asarray(y, dtype=np.float64)
    params = _params_from_raw(dist, raw_params)
    grads = dist.nll_gradient(y, params)

    for name in dist.param_names:
        fd = _fd_grad_raw(dist, y, raw_params, name, objective=objective)
        assert_allclose(
            grads[name][0],
            fd,
            rtol=rtol,
            atol=atol,
            err_msg=f"{type(dist).__name__} gradient mismatch for parameter '{name}'",
        )
        # Hessians must be positive (used as tree weights)
        assert np.all(grads[name][1] > 0), f"non-positive hessian for '{name}'"


def _tweedie_surrogate_objective(dist):
    """The objective Tweedie's nll_gradient actually descends.

    Tweedie training gradients are deviance-based (standard GLM practice):
    grad_mu = d/d(log mu) of dev/(2*phi) and grad_phi = d/d(log phi) of
    dev/(2*phi) + log(phi), where the +log(phi) term is a saddle-point-style
    stand-in for the phi-dependence of the intractable normalizing constant.
    This is NOT the exact compound Poisson-Gamma NLL (which dist.nll now
    evaluates), so finite differences are taken against this surrogate.
    """

    def objective(y, params):
        mu = np.clip(params['mu'], 1e-10, None)
        deviance = dist._compute_deviance(np.clip(y, 0, None), mu)
        return deviance / (2 * params['phi']) + np.log(params['phi'])

    return objective


# =============================================================================
# Gradient vs finite differences, every registered family
# =============================================================================


class TestGradientsMatchFiniteDifferences:
    """Analytic d(NLL)/d(raw) must match FD of nll through the link."""

    def test_normal(self):
        y = np.array([-4.0, 0.0, 0.3, 6.0])
        raw = {
            'loc': np.array([1.0, -0.5, 0.0, 2.0]),
            'scale': np.array([-1.0, 0.0, 0.5, 1.5]),
        }
        _check_family_gradients(Normal(), y, raw)

    def test_lognormal(self):
        y = np.array([0.05, 1.0, 4.0, 50.0])
        raw = {
            'loc': np.array([0.0, 0.5, 1.0, 2.0]),
            'scale': np.array([-0.5, 0.0, 0.3, 1.0]),
        }
        _check_family_gradients(LogNormal(), y, raw)

    def test_gamma(self):
        y = np.array([0.1, 1.0, 4.2, 20.0])
        raw = {
            'concentration': np.array([0.0, 0.5, -0.5, 1.0]),
            'rate': np.array([0.0, -1.0, 0.5, 0.3]),
        }
        _check_family_gradients(Gamma(), y, raw)

    def test_poisson(self):
        y = np.array([0.0, 1.0, 3.0, 15.0])  # includes edge y=0
        raw = {'rate': np.array([-1.0, 0.0, 1.0, 2.5])}
        _check_family_gradients(Poisson(), y, raw)

    def test_studentt(self):
        y = np.array([-3.0, 0.0, 0.5, 5.0])
        raw = {
            'loc': np.array([0.0, 1.0, 0.0, -1.0]),
            'scale': np.array([0.0, -0.5, 0.7, 0.3]),
            'df': np.array([0.0, 1.0, 2.0, 3.0]),  # nu = 2 + softplus(raw)
        }
        _check_family_gradients(StudentT(), y, raw)

    @pytest.mark.parametrize("power", [1.3, 1.5, 1.7])
    def test_tweedie(self, power):
        # See _tweedie_surrogate_objective: training gradients descend the
        # deviance-based surrogate, not the exact series NLL.
        dist = Tweedie(power=power)
        y = np.array([0.0, 0.5, 1.0, 10.0])  # includes edge y=0 (point mass)
        raw = {
            'mu': np.array([0.7, 0.0, 0.7, 1.1]),
            'phi': np.array([0.0, -0.7, 0.0, 0.7]),
        }
        _check_family_gradients(
            dist, y, raw, objective=_tweedie_surrogate_objective(dist)
        )

    def test_negative_binomial(self):
        y = np.array([0.0, 1.0, 5.0, 12.0])  # includes edge y=0
        raw = {
            'mu': np.array([1.1, 0.0, 1.1, 2.0]),
            'r': np.array([0.7, 0.7, 0.0, -0.4]),
        }
        _check_family_gradients(NegativeBinomial(), y, raw)

    def test_negative_binomial_dispersion_gradient_analytic(self):
        """r-gradient: d(NLL)/d(log r) = -r*(psi(y+r)-psi(r)+log p+(1-p)-y/(r+mu))."""
        from scipy.special import digamma

        dist = NegativeBinomial()
        y = np.array([0.0, 5.0, 12.0, 1.0])
        mu = np.array([3.0, 3.0, 7.5, 0.5])
        r = np.array([2.0, 2.0, 0.7, 10.0])

        grads = dist.nll_gradient(y, {'mu': mu, 'r': r})
        p = r / (r + mu)
        expected = -r * (
            digamma(y + r) - digamma(r) + np.log(p) + (1 - p) - y / (r + mu)
        )
        assert_allclose(grads['r'][0], expected, rtol=1e-5, atol=1e-6)


# =============================================================================
# NLL vs scipy.stats reference densities
# =============================================================================


class TestNLLAgainstScipy:
    """dist.nll must equal the full negative log-density from scipy.stats."""

    def test_normal(self):
        y = np.array([-2.0, 0.0, 1.5, 4.0])
        params = {'loc': np.array([0.0, 1.0, 1.0, 3.0]),
                  'scale': np.array([1.0, 0.5, 2.0, 1.5])}
        expected = -stats.norm.logpdf(y, loc=params['loc'], scale=params['scale'])
        assert_allclose(Normal().nll(y, params), expected, rtol=1e-6)

    def test_lognormal(self):
        y = np.array([0.1, 1.0, 3.0, 20.0])
        params = {'loc': np.array([0.0, 0.5, 1.0, 2.0]),
                  'scale': np.array([1.0, 0.5, 0.8, 1.2])}
        expected = -stats.lognorm.logpdf(y, s=params['scale'], scale=np.exp(params['loc']))
        assert_allclose(LogNormal().nll(y, params), expected, rtol=1e-6)

    def test_gamma(self):
        y = np.array([0.2, 1.0, 4.0, 9.0])
        params = {'concentration': np.array([1.0, 2.0, 3.0, 0.5]),
                  'rate': np.array([1.0, 0.5, 2.0, 1.5])}
        expected = -stats.gamma.logpdf(y, a=params['concentration'],
                                       scale=1.0 / params['rate'])
        assert_allclose(Gamma().nll(y, params), expected, rtol=1e-6)

    def test_poisson(self):
        y = np.array([0.0, 1.0, 4.0, 10.0])
        params = {'rate': np.array([0.5, 1.0, 3.0, 12.0])}
        expected = -stats.poisson.logpmf(y.astype(int), mu=params['rate'])
        assert_allclose(Poisson().nll(y, params), expected, rtol=1e-6)

    def test_studentt(self):
        y = np.array([-2.0, 0.0, 1.0, 5.0])
        params = {'loc': np.array([0.0, 0.5, 1.0, 2.0]),
                  'scale': np.array([1.0, 0.5, 2.0, 1.0]),
                  'df': np.array([3.0, 5.0, 10.0, 2.5])}
        expected = -stats.t.logpdf(y, df=params['df'], loc=params['loc'],
                                   scale=params['scale'])
        assert_allclose(StudentT().nll(y, params), expected, rtol=1e-6)

    def test_negative_binomial(self):
        y = np.array([0.0, 1.0, 5.0, 12.0])
        params = {'mu': np.array([1.0, 2.0, 5.0, 8.0]),
                  'r': np.array([2.0, 1.5, 3.0, 0.8])}
        p = params['r'] / (params['r'] + params['mu'])
        expected = -stats.nbinom.logpmf(y.astype(int), n=params['r'], p=p)
        assert_allclose(NegativeBinomial().nll(y, params), expected, rtol=1e-5)


# =============================================================================
# Tweedie predictive correctness
# =============================================================================


def _tweedie_pdf_brute(y, mu, phi, power, nmax=400):
    """Brute-force compound Poisson-Gamma density for y > 0.

    f(y) = sum_n P(N=n) * GammaPDF(y; shape=n*alpha, scale) — explicit sum
    over the Poisson count of Gamma convolutions.
    """
    lam = mu ** (2 - power) / (phi * (2 - power))
    alpha = (2 - power) / (power - 1)
    scale = phi * (power - 1) * mu ** (power - 1)
    total = 0.0
    for n in range(1, nmax + 1):
        log_pn = -lam + n * np.log(lam) - gammaln(n + 1)
        log_fy = (
            (n * alpha - 1) * np.log(y)
            - y / scale
            - gammaln(n * alpha)
            - n * alpha * np.log(scale)
        )
        total += np.exp(log_pn + log_fy)
    return total


class TestTweediePredictive:
    """Proper likelihood, support-aware quantiles, vectorized sampling."""

    @pytest.mark.parametrize(
        "y,mu,phi,power",
        [
            (1.0, 2.0, 1.0, 1.5),
            (0.5, 1.0, 0.5, 1.3),
            (10.0, 3.0, 2.0, 1.7),
            (3.0, 0.8, 1.5, 1.5),
        ],
    )
    def test_nll_matches_brute_force_density(self, y, mu, phi, power):
        dist = Tweedie(power=power)
        nll = dist.nll(np.array([y]), {'mu': np.array([mu]), 'phi': np.array([phi])})
        expected = -np.log(_tweedie_pdf_brute(y, mu, phi, power))
        assert_allclose(nll[0], expected, rtol=1e-6)

    def test_nll_at_zero_is_poisson_mass(self):
        """Exact log P(Y=0) = -lambda with lambda = mu^(2-p)/(phi*(2-p))."""
        dist = Tweedie(power=1.5)
        mu = np.array([0.5, 2.0, 5.0])
        phi = np.array([1.0, 1.0, 2.0])
        lam = mu ** 0.5 / (phi * 0.5)
        nll = dist.nll(np.zeros(3), {'mu': mu, 'phi': phi})
        assert_allclose(nll, lam, rtol=1e-10)

    def test_quantiles_nonnegative_and_monotone(self):
        dist = Tweedie(power=1.5)
        params = {'mu': np.array([0.5, 2.0, 5.0]), 'phi': np.array([1.0, 1.0, 0.5])}
        prev = np.full(3, -np.inf)
        for q in [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]:
            val = dist.quantile(params, q)
            assert np.all(val >= 0)
            assert np.all(val >= prev), f"quantiles not monotone at q={q}"
            prev = val

    def test_quantile_zero_mass_matches_exp_lambda(self):
        """Fraction of exact-zero quantiles over a q-grid must equal P(Y=0)."""
        dist = Tweedie(power=1.5)
        mu, phi = 2.0, 1.0
        lam = mu ** 0.5 / (phi * 0.5)
        p_zero = np.exp(-lam)
        params = {'mu': np.array([mu]), 'phi': np.array([phi])}

        # Exact threshold behaviour around P(Y=0)
        assert dist.quantile(params, p_zero * 0.9)[0] == 0.0
        assert dist.quantile(params, min(p_zero * 1.1, 0.999))[0] > 0.0

        # Implied zero fraction over a fine grid of q values
        q_grid = np.linspace(0.0005, 0.9995, 2000)
        zero_frac = np.mean([dist.quantile(params, q)[0] == 0.0 for q in q_grid])
        assert abs(zero_frac - p_zero) < 0.01

    def test_sample_vectorized_zero_mass_and_mean(self):
        dist = Tweedie(power=1.5)
        n_obs = 500
        mu, phi = 2.0, 1.0
        lam = mu ** 0.5 / (phi * 0.5)
        params = {'mu': np.full(n_obs, mu), 'phi': np.full(n_obs, phi)}

        samples = dist.sample(params, n_samples=40, seed=42)
        assert samples.shape == (n_obs, 40)
        assert np.all(samples >= 0)
        # Zero mass and mean match the compound Poisson-Gamma
        assert abs(np.mean(samples == 0) - np.exp(-lam)) < 0.01
        assert abs(np.mean(samples) - mu) < 0.1


# =============================================================================
# CustomDistribution: link chain rule and empirical Fisher
# =============================================================================


class TestCustomDistributionChainRule:
    """Gradients/hessians must be w.r.t. RAW params through the link."""

    @staticmethod
    def _make_dist(use_jax):
        # Smooth pseudo-NLL exercising exp, sigmoid, and softplus links
        def nll_fn(y, p):
            a, b, c = p['a'], p['b'], p['c']
            return (y - a * b) ** 2 + c * y ** 2 + a
        return CustomDistribution(
            param_names=['a', 'b', 'c'],
            link_functions={'a': 'exp', 'b': 'sigmoid', 'c': 'softplus'},
            nll_fn=nll_fn,
            use_jax=use_jax,
        )

    def test_numerical_gradient_matches_fd_through_links(self):
        dist = self._make_dist(use_jax=False)
        y = np.array([0.5, -1.0, 2.0, 3.5])
        raw = {
            'a': np.array([0.0, 0.5, -0.5, 1.0]),
            'b': np.array([0.0, 1.0, -1.0, 0.3]),
            'c': np.array([0.0, -0.5, 0.5, 1.0]),
        }
        params = _params_from_raw(dist, raw)
        grads = dist.nll_gradient(y, params)

        for name in dist.param_names:
            fd = _fd_grad_raw(dist, y, raw, name, objective=dist.nll)
            assert_allclose(
                grads[name][0], fd, rtol=1e-3, atol=1e-4,
                err_msg=f"numerical-path gradient wrong for '{name}' link",
            )

    def test_numerical_hessian_includes_second_order_link_term(self):
        """h_raw = h_c*(link')^2 + g_c*link'' — the g_c*link'' term matters.

        For Normal NLL with sigma = exp(s): d2(NLL)/ds2 = 2*(y-mu)^2/sigma^2,
        while dropping the second-order term would give (3r^2/sigma^2 - 1).
        """
        def nll_fn(y, p):
            mu, sigma = p['loc'], p['scale']
            return 0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)

        dist = CustomDistribution(
            param_names=['loc', 'scale'],
            link_functions={'loc': 'identity', 'scale': 'exp'},
            nll_fn=nll_fn,
            use_jax=False,
        )
        y = np.array([1.5, -0.5, 3.0])
        params = {'loc': np.array([0.0, 0.0, 1.0]), 'scale': np.array([1.0, 2.0, 0.5])}
        grads = dist.nll_gradient(y, params)

        resid = y - params['loc']
        expected_hess = 2.0 * resid ** 2 / params['scale'] ** 2
        assert_allclose(grads['scale'][1], expected_hess, rtol=1e-2)

    def test_jax_gradient_matches_numerical_path(self):
        """JAX path must differentiate through the link, like the numerical path.

        Runs in CI (linux with jax); skipped locally when jax is missing.
        """
        pytest.importorskip('jax')
        import jax.numpy as jnp

        def nll_np(y, p):
            mu, sigma, w = p['loc'], p['scale'], p['w']
            return 0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu * w) ** 2 / (2 * sigma ** 2)

        def nll_jnp(y, p):
            mu, sigma, w = p['loc'], p['scale'], p['w']
            return 0.5 * jnp.log(2 * jnp.pi * sigma ** 2) + (y - mu * w) ** 2 / (2 * sigma ** 2)

        links = {'loc': 'identity', 'scale': 'exp', 'w': 'sigmoid'}
        d_num = CustomDistribution(['loc', 'scale', 'w'], links, nll_np, use_jax=False)
        d_jax = CustomDistribution(['loc', 'scale', 'w'], links, nll_jnp, use_jax=True)
        assert d_jax._jax_available

        y = np.array([0.5, -1.0, 2.0, 3.5])
        raw = {
            'loc': np.array([0.0, 0.5, -0.5, 1.0]),
            'scale': np.array([0.0, 0.5, -0.5, 0.3]),
            'w': np.array([0.0, 1.0, -1.0, 0.5]),
        }
        params = _params_from_raw(d_num, raw)

        # Call the JAX path directly so a silent numerical fallback can't pass
        res_jax = d_jax._jax_gradient(y, params)
        res_num = d_num.nll_gradient(y, params)

        for name in d_num.param_names:
            assert_allclose(
                res_jax[name][0], res_num[name][0], rtol=1e-2, atol=1e-3,
                err_msg=f"JAX vs numerical gradient mismatch for '{name}'",
            )
            # Hessians agree where not clipped at the positivity floor
            unclipped = res_num[name][1] > 1e-5
            if np.any(unclipped):
                assert_allclose(
                    res_jax[name][1][unclipped], res_num[name][1][unclipped],
                    rtol=5e-2, atol=1e-3,
                    err_msg=f"JAX vs numerical hessian mismatch for '{name}'",
                )


class TestCustomDistributionEmpiricalFisher:
    """Fisher must be the empirical g^2 diagonal, not identity."""

    @staticmethod
    def _make_dist():
        def nll_fn(y, p):
            mu, sigma = p['loc'], p['scale']
            return 0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)
        return CustomDistribution(
            param_names=['loc', 'scale'],
            link_functions={'loc': 'identity', 'scale': 'exp'},
            nll_fn=nll_fn,
            use_jax=False,
        )

    def test_fisher_non_identity_on_asymmetric_data(self):
        dist = self._make_dist()
        rng = np.random.default_rng(0)
        y = rng.exponential(3.0, size=100)  # asymmetric targets
        params = {'loc': np.zeros(100), 'scale': np.ones(100)}

        grads = dist.nll_gradient(y, params)
        F = dist.fisher_information(params)

        n_params = dist.n_params
        identity = np.broadcast_to(np.eye(n_params, dtype=np.float32), F.shape)
        assert not np.allclose(F, identity), "Fisher degenerated to identity"

        # Diagonal equals the batch mean of squared per-sample gradients
        for j, name in enumerate(dist.param_names):
            expected = max(np.mean(grads[name][0].astype(np.float64) ** 2), 1e-6)
            assert_allclose(F[:, j, j], expected, rtol=1e-4)
        # Off-diagonals stay zero (diagonal approximation)
        assert np.all(F[:, 0, 1] == 0) and np.all(F[:, 1, 0] == 0)

    def test_fisher_with_explicit_y_matches_cached_path(self):
        dist = self._make_dist()
        rng = np.random.default_rng(1)
        y = rng.exponential(2.0, size=50)
        params = {'loc': np.zeros(50), 'scale': np.ones(50)}

        F_explicit = dist.fisher_information(params, y=y)

        dist2 = self._make_dist()
        dist2.nll_gradient(y, params)
        F_cached = dist2.fisher_information(params)
        assert_allclose(F_explicit, F_cached, rtol=1e-6)

    def test_fisher_identity_fallback_without_gradients(self):
        dist = self._make_dist()
        params = {'loc': np.zeros(10), 'scale': np.ones(10)}
        F = dist.fisher_information(params)  # no gradient info available yet
        identity = np.broadcast_to(np.eye(2, dtype=np.float32), F.shape)
        assert np.allclose(F, identity)

    def test_natural_gradient_uses_empirical_fisher(self):
        """natural_gradient must rescale by 1/E[g^2], not return plain grads."""
        dist = self._make_dist()
        rng = np.random.default_rng(2)
        y = rng.exponential(3.0, size=80)
        params = {'loc': np.zeros(80), 'scale': np.ones(80)}

        nat = dist.natural_gradient(y, params)
        ord_grads = dist.nll_gradient(y, params)

        for name in dist.param_names:
            scale = np.mean(ord_grads[name][0].astype(np.float64) ** 2)
            expected = ord_grads[name][0] / scale
            assert_allclose(nat[name][0], expected, rtol=1e-3, atol=1e-4)


# =============================================================================
# Target-support validation at fit entry
# =============================================================================


class TestTargetValidation:
    """Out-of-support targets must raise at fit entry, not be clipped."""

    @pytest.mark.parametrize("dist_cls", [Gamma, LogNormal])
    @pytest.mark.parametrize("bad_value", [0.0, -1.0])
    def test_positive_support_rejects_nonpositive(self, dist_cls, bad_value):
        dist = dist_cls()
        y = np.array([1.0, 2.0, bad_value])
        with pytest.raises(ValueError, match="positive targets"):
            dist.init_params(y)

    @pytest.mark.parametrize("dist_cls", [Poisson, NegativeBinomial])
    def test_count_support_rejects_negative(self, dist_cls):
        dist = dist_cls()
        with pytest.raises(ValueError, match="non-negative"):
            dist.init_params(np.array([1.0, -2.0, 3.0]))

    @pytest.mark.parametrize("dist_cls", [Poisson, NegativeBinomial])
    def test_count_support_rejects_non_integer(self, dist_cls):
        dist = dist_cls()
        with pytest.raises(ValueError, match="integer count"):
            dist.init_params(np.array([1.0, 2.5, 3.0]))

    @pytest.mark.parametrize("dist_cls", [Poisson, NegativeBinomial])
    def test_count_support_accepts_float_noise(self, dist_cls):
        """Tiny float noise from float32 casts must not be rejected."""
        dist = dist_cls()
        y = np.array([0.0, 1.0 + 1e-7, 5.0 - 1e-7, 100.0])
        dist.init_params(y)  # must not raise

    def test_tweedie_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            Tweedie().init_params(np.array([1.0, -0.5, 0.0]))

    def test_tweedie_accepts_zeros(self):
        Tweedie().init_params(np.array([0.0, 0.0, 1.5, 3.0]))  # must not raise

    def test_valid_targets_pass(self):
        Gamma().init_params(np.array([0.5, 1.0, 2.0]))
        LogNormal().init_params(np.array([0.1, 1.0, 10.0]))
        Poisson().init_params(np.array([0.0, 1.0, 5.0]))
        NegativeBinomial().init_params(np.array([0.0, 2.0, 7.0]))

    def test_fit_entry_validation_end_to_end(self):
        """DistributionalGBDT.fit must reject out-of-support targets."""
        from openboost import DistributionalGBDT

        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3)).astype(np.float32)
        y = rng.gamma(2.0, 1.0, size=50).astype(np.float32)
        y[0] = 0.0  # out of Gamma support

        model = DistributionalGBDT(distribution='gamma', n_trees=2, max_depth=2)
        with pytest.raises(ValueError, match="positive targets"):
            model.fit(X, y)


# =============================================================================
# End-to-end: mean NLL decreases over boosting rounds
# =============================================================================


def _make_normal_y(rng, X):
    return (X[:, 0] * 2 + rng.normal(0, 1, len(X))).astype(np.float32)


def _make_gamma_y(rng, X):
    return rng.gamma(2.0, np.exp(0.3 * X[:, 0])).astype(np.float32) + 1e-3


def _make_poisson_y(rng, X):
    return rng.poisson(np.exp(0.5 + 0.4 * X[:, 0])).astype(np.float32)


def _make_negbin_y(rng, X):
    mu = np.exp(0.8 + 0.4 * X[:, 0])
    r = 2.0
    return rng.negative_binomial(r, r / (r + mu)).astype(np.float32)


class TestNLLDecreasesDuringBoosting:
    """Training must reduce mean NLL for NaturalBoost on each family."""

    @pytest.mark.parametrize(
        "dist_name,make_y",
        [
            ('normal', _make_normal_y),
            ('gamma', _make_gamma_y),
            ('poisson', _make_poisson_y),
            ('negbin', _make_negbin_y),
        ],
    )
    def test_naturalboost_mean_nll_decreases(self, dist_name, make_y):
        from openboost import HistoryCallback, NaturalBoost

        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4)).astype(np.float32)
        y = make_y(rng, X)

        history = HistoryCallback()
        model = NaturalBoost(
            distribution=dist_name, n_trees=25, max_depth=3, learning_rate=0.1
        )
        model.fit(X, y, callbacks=[history])

        losses = history.history['train_loss']
        assert len(losses) >= 10
        assert np.all(np.isfinite(losses))
        # NLL after training must beat the NLL at initialization,
        # and the tail of training must beat the head
        assert losses[-1] < losses[0], f"{dist_name}: NLL did not decrease"
        assert np.mean(losses[-5:]) < np.mean(losses[:5])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
