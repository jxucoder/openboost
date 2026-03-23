"""Loss function correctness tests for OpenBoost.

Verifies all loss functions using two approaches:
1. Analytical: compare gradient/hessian against independently computed formulas
2. Numerical differentiation: central differences vs returned gradient

This catches sign errors, missing factors, numerical instability.
"""

import numpy as np
import pytest

from openboost._loss import (
    compute_loss_value,
    gamma_gradient,
    get_loss_function,
    huber_gradient,  # noqa: F401
    logloss_gradient,
    mae_gradient,
    mse_gradient,
    poisson_gradient,
    quantile_gradient,  # noqa: F401
    tweedie_gradient,
)


def _numerical_gradient(loss_name, pred, y, eps=1e-5, **kwargs):
    """Compute gradient numerically via central differences.

    grad_i ≈ (L(pred+eps) - L(pred-eps)) / (2*eps)
    """
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(pred)
    num_grad = np.zeros(n, dtype=np.float64)

    for i in range(n):
        pred_plus = pred.copy()
        pred_minus = pred.copy()
        pred_plus[i] += eps
        pred_minus[i] -= eps
        loss_plus = compute_loss_value(loss_name, pred_plus, y, **kwargs) * n
        loss_minus = compute_loss_value(loss_name, pred_minus, y, **kwargs) * n
        num_grad[i] = (loss_plus - loss_minus) / (2 * eps)

    return num_grad


# =============================================================================
# Analytical gradient verification
# =============================================================================


class TestAnalyticalGradients:
    """Verify gradients against independently computed formulas."""

    def test_mse_gradient_analytical(self):
        """MSE: grad = 2*(pred - y), hess = 2."""
        pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([0.5, 2.5, 1.0], dtype=np.float32)

        grad, hess = mse_gradient(pred, y)

        expected_grad = 2.0 * (pred - y)
        np.testing.assert_allclose(grad, expected_grad, atol=1e-6)
        np.testing.assert_allclose(hess, 2.0, atol=1e-6)

    def test_mse_gradient_zero_at_match(self):
        """MSE gradient should be zero when pred == y."""
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grad, hess = mse_gradient(y.copy(), y)
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)

    def test_logloss_gradient_analytical(self):
        """LogLoss: grad = sigmoid(pred) - y, hess = p*(1-p)."""
        pred = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        y = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)

        grad, hess = logloss_gradient(pred, y)

        # Independently compute sigmoid
        p = 1.0 / (1.0 + np.exp(-pred.astype(np.float64)))
        expected_grad = (p - y).astype(np.float32)
        expected_hess = np.clip((p * (1 - p)).astype(np.float32), 1e-6, 1.0 - 1e-6)

        np.testing.assert_allclose(grad, expected_grad, atol=1e-5)
        np.testing.assert_allclose(hess, expected_hess, atol=1e-5)

    def test_mae_gradient_sign(self):
        """MAE: grad = sign(pred - y)."""
        pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([0.5, 3.0, 1.0], dtype=np.float32)

        grad, hess = mae_gradient(pred, y)

        expected_sign = np.sign(pred - y)
        np.testing.assert_allclose(grad, expected_sign, atol=1e-5)

    def test_poisson_gradient_analytical(self):
        """Poisson: grad = exp(pred) - y."""
        pred = np.array([0.0, 1.0, -0.5], dtype=np.float32)
        y = np.array([2.0, 1.0, 3.0], dtype=np.float32)

        grad, hess = poisson_gradient(pred, y)

        expected_grad = np.exp(pred) - y
        np.testing.assert_allclose(grad, expected_grad, atol=1e-4)

    def test_gamma_gradient_analytical(self):
        """Gamma: grad = 1 - y*exp(-pred), hess = y*exp(-pred)."""
        pred = np.array([1.0, 0.5, 2.0], dtype=np.float32)
        y = np.array([2.0, 1.0, 3.0], dtype=np.float32)

        grad, hess = gamma_gradient(pred, y)

        expected_grad = 1.0 - y * np.exp(-pred)
        np.testing.assert_allclose(grad, expected_grad, atol=1e-4)


# =============================================================================
# Numerical differentiation verification
# =============================================================================


class TestNumericalGradients:
    """Verify gradients match numerical differentiation (central differences)."""

    def _check_gradient(self, loss_name, pred, y, atol=1e-3, **kwargs):
        """Helper: compare analytical gradient against numerical gradient."""
        loss_fn = get_loss_function(loss_name, **kwargs)
        grad, _ = loss_fn(
            np.asarray(pred, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
        )
        grad = np.asarray(grad, dtype=np.float64)

        num_grad = _numerical_gradient(loss_name, pred, y, **kwargs)

        np.testing.assert_allclose(
            grad, num_grad, atol=atol,
            err_msg=f"Gradient mismatch for {loss_name}"
        )

    def test_mse_gradient_numerical(self):
        pred = np.array([1.0, 2.5, -0.3])
        y = np.array([0.5, 3.0, 1.0])
        self._check_gradient('mse', pred, y)

    def test_logloss_gradient_numerical(self):
        pred = np.array([0.5, -1.0, 2.0])
        y = np.array([1.0, 0.0, 1.0])
        self._check_gradient('logloss', pred, y)

    def test_huber_gradient_numerical(self):
        pred = np.array([1.0, 5.0, -2.0])
        y = np.array([0.5, 0.0, 1.0])
        self._check_gradient('huber', pred, y, huber_delta=1.0)

    def test_poisson_gradient_numerical(self):
        pred = np.array([0.5, 1.0, -0.5])
        y = np.array([2.0, 1.0, 3.0])
        self._check_gradient('poisson', pred, y, atol=1e-2)

    def test_gamma_gradient_numerical(self):
        pred = np.array([0.5, 1.0, 1.5])
        y = np.array([2.0, 1.0, 3.0])
        self._check_gradient('gamma', pred, y, atol=1e-2)

    @pytest.mark.parametrize("rho", [1.1, 1.5, 1.9])
    def test_tweedie_gradient_numerical(self, rho):
        pred = np.array([0.5, 1.0, 0.2])
        y = np.array([2.0, 1.0, 3.0])
        self._check_gradient('tweedie', pred, y, tweedie_rho=rho, atol=1e-2)

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
    def test_quantile_gradient_numerical(self, alpha):
        """Quantile loss gradient via numerical differentiation.

        Note: quantile loss has discontinuous gradient at pred=y,
        so we avoid exact match points.
        """
        pred = np.array([1.5, 0.3, -0.7])
        y = np.array([1.0, 2.0, 0.5])
        self._check_gradient('quantile', pred, y, quantile_alpha=alpha, atol=0.1)


# =============================================================================
# Edge cases and numerical stability
# =============================================================================


class TestLossEdgeCases:
    """Edge cases that can cause NaN, overflow, or incorrect behavior."""

    def test_logloss_extreme_negative_pred(self):
        """Very negative predictions should not produce NaN."""
        pred = np.array([-500.0, -100.0], dtype=np.float32)
        y = np.array([1.0, 0.0], dtype=np.float32)

        grad, hess = logloss_gradient(pred, y)

        assert np.all(np.isfinite(grad)), f"NaN in logloss grad: {grad}"
        assert np.all(np.isfinite(hess)), f"NaN in logloss hess: {hess}"

    def test_logloss_extreme_positive_pred(self):
        """Very positive predictions should not produce NaN."""
        pred = np.array([500.0, 100.0], dtype=np.float32)
        y = np.array([0.0, 1.0], dtype=np.float32)

        grad, hess = logloss_gradient(pred, y)

        assert np.all(np.isfinite(grad)), f"NaN in logloss grad: {grad}"
        assert np.all(np.isfinite(hess)), f"NaN in logloss hess: {hess}"

    def test_poisson_large_pred_no_overflow(self):
        """exp(pred) should not overflow for large predictions."""
        pred = np.array([15.0, 18.0], dtype=np.float32)
        y = np.array([1.0, 2.0], dtype=np.float32)

        grad, hess = poisson_gradient(pred, y)

        assert np.all(np.isfinite(grad)), f"Overflow in poisson grad: {grad}"
        assert np.all(np.isfinite(hess)), f"Overflow in poisson hess: {hess}"

    def test_tweedie_zero_y(self):
        """y=0 is valid for Tweedie — should not produce NaN."""
        pred = np.array([0.5, 1.0], dtype=np.float32)
        y = np.array([0.0, 0.0], dtype=np.float32)

        grad, hess = tweedie_gradient(pred, y, rho=1.5)

        assert np.all(np.isfinite(grad)), f"NaN in tweedie grad with y=0: {grad}"
        assert np.all(np.isfinite(hess)), f"NaN in tweedie hess with y=0: {hess}"

    def test_all_losses_finite_on_normal_input(self):
        """Every built-in loss should produce finite grad/hess on normal inputs."""
        rng = np.random.RandomState(42)
        pred = rng.randn(20).astype(np.float32)
        y_reg = rng.randn(20).astype(np.float32)
        y_bin = (rng.rand(20) > 0.5).astype(np.float32)
        y_pos = np.abs(y_reg) + 0.1  # Positive for Poisson/Gamma

        losses_and_data = [
            ('mse', pred, y_reg),
            ('mae', pred, y_reg),
            ('huber', pred, y_reg),
            ('logloss', pred, y_bin),
            ('poisson', pred * 0.5, y_pos),  # Smaller pred to avoid overflow
            ('gamma', np.abs(pred) + 0.1, y_pos),
        ]

        for loss_name, p, y in losses_and_data:
            loss_fn = get_loss_function(loss_name)
            grad, hess = loss_fn(p, y)
            assert np.all(np.isfinite(grad)), f"{loss_name}: NaN/inf in grad"
            assert np.all(np.isfinite(hess)), f"{loss_name}: NaN/inf in hess"


# =============================================================================
# Loss value computation
# =============================================================================


class TestLossValueComputation:
    """Verify compute_loss_value returns correct scalar losses."""

    def test_mse_loss_value(self):
        pred = np.array([1.0, 2.0, 3.0])
        y = np.array([1.5, 2.5, 2.0])
        loss = compute_loss_value('mse', pred, y)
        expected = np.mean((pred - y) ** 2)
        np.testing.assert_almost_equal(loss, expected, decimal=6)

    def test_mae_loss_value(self):
        pred = np.array([1.0, 2.0, 3.0])
        y = np.array([1.5, 2.5, 2.0])
        loss = compute_loss_value('mae', pred, y)
        expected = np.mean(np.abs(pred - y))
        np.testing.assert_almost_equal(loss, expected, decimal=6)

    def test_logloss_value(self):
        pred = np.array([2.0, -1.0])
        y = np.array([1.0, 0.0])
        loss = compute_loss_value('logloss', pred, y)
        # Manual: p = sigmoid(pred), -mean(y*log(p) + (1-y)*log(1-p))
        p = 1.0 / (1.0 + np.exp(-pred.astype(np.float64)))
        expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        np.testing.assert_almost_equal(loss, expected, decimal=6)

    def test_loss_zero_when_perfect(self):
        """MSE loss should be zero when predictions are perfect."""
        y = np.array([1.0, 2.0, 3.0])
        loss = compute_loss_value('mse', y, y)
        np.testing.assert_almost_equal(loss, 0.0, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
