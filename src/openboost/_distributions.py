"""Probability distributions for distributional GBDT.

Phase 15.1: Distribution classes for probabilistic prediction.

Each distribution defines:
- Parameters (e.g., μ, σ for Normal)
- Link functions (e.g., exp for scale parameters to ensure positivity)
- Negative log-likelihood gradient/hessian per parameter
- Fisher information matrix (for natural gradient / NGBoost)

Supported distributions:
- Normal (Gaussian): loc, scale
- LogNormal: loc, scale (of underlying normal)
- Gamma: concentration, rate
- Poisson: rate
- NegativeBinomial: mean, dispersion
- StudentT: loc, scale, df
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np
from numpy.typing import NDArray

from ._backends import is_cuda


# Type alias for gradient/hessian tuple
GradHess = Tuple[NDArray, NDArray]


@dataclass
class DistributionOutput:
    """Container for distribution parameter predictions.
    
    Attributes:
        params: Dictionary mapping parameter names to predicted values
        distribution: The Distribution instance used
    """
    params: dict[str, NDArray]
    distribution: "Distribution"
    
    def mean(self) -> NDArray:
        """Expected value E[Y|X]."""
        return self.distribution.mean(self.params)
    
    def variance(self) -> NDArray:
        """Variance Var[Y|X]."""
        return self.distribution.variance(self.params)
    
    def std(self) -> NDArray:
        """Standard deviation."""
        return np.sqrt(self.variance())
    
    def quantile(self, q: float) -> NDArray:
        """q-th quantile (0 < q < 1)."""
        return self.distribution.quantile(self.params, q)
    
    def interval(self, alpha: float = 0.1) -> Tuple[NDArray, NDArray]:
        """(1-alpha) prediction interval.
        
        Args:
            alpha: Significance level (0.1 = 90% interval)
            
        Returns:
            (lower, upper) bounds
        """
        lower = self.quantile(alpha / 2)
        upper = self.quantile(1 - alpha / 2)
        return lower, upper
    
    def sample(self, n_samples: int = 1, seed: int | None = None) -> NDArray:
        """Draw samples from the predicted distribution.
        
        Args:
            n_samples: Number of samples per observation
            seed: Random seed for reproducibility
            
        Returns:
            samples: Shape (n_observations, n_samples)
        """
        return self.distribution.sample(self.params, n_samples, seed)
    
    def nll(self, y: NDArray) -> NDArray:
        """Negative log-likelihood for observed values.
        
        Args:
            y: Observed values
            
        Returns:
            nll: Per-sample negative log-likelihood
        """
        return self.distribution.nll(y, self.params)


class Distribution(ABC):
    """Base class for probability distributions.
    
    Subclasses must implement:
    - n_params: Number of distributional parameters
    - param_names: Names of parameters
    - link: Transform raw -> constrained parameter space
    - link_inv: Transform constrained -> raw
    - nll_gradient: Gradient and hessian of NLL w.r.t. raw parameters
    - fisher_information: Fisher information matrix (for NGBoost)
    """
    
    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of distributional parameters."""
        pass
    
    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Names of parameters, e.g., ['loc', 'scale']."""
        pass
    
    @abstractmethod
    def link(self, param_name: str, raw: NDArray) -> NDArray:
        """Apply link function: raw -> constrained parameter space.
        
        E.g., for scale: exp(raw) to ensure positivity.
        
        Args:
            param_name: Name of the parameter
            raw: Raw (unbounded) values
            
        Returns:
            Constrained parameter values
        """
        pass
    
    @abstractmethod
    def link_inv(self, param_name: str, param: NDArray) -> NDArray:
        """Inverse link: constrained -> raw (for initialization).
        
        Args:
            param_name: Name of the parameter
            param: Constrained parameter values
            
        Returns:
            Raw (unbounded) values
        """
        pass
    
    @abstractmethod
    def nll_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Compute gradient and hessian of NLL w.r.t. each RAW parameter.
        
        The gradient is d(NLL)/d(raw), accounting for the link function.
        
        Args:
            y: Observed target values
            params: Dictionary of constrained parameter values
            
        Returns:
            Dictionary mapping param_name -> (gradient, hessian)
        """
        pass
    
    @abstractmethod
    def fisher_information(
        self,
        params: dict[str, NDArray],
    ) -> NDArray:
        """Fisher information matrix at given parameters.
        
        Shape: (n_samples, n_params, n_params)
        Used for natural gradient computation in NGBoost.
        
        Args:
            params: Dictionary of constrained parameter values
            
        Returns:
            Fisher information matrix
        """
        pass
    
    def _is_diagonal_fisher(self, F: NDArray) -> bool:
        """Check if Fisher matrix is diagonal (common for many distributions)."""
        n_params = F.shape[1]
        if n_params == 1:
            return True
        # Check if off-diagonal elements are near zero (sample a few)
        off_diag_sum = np.sum(np.abs(F[0])) - np.sum(np.abs(np.diag(F[0])))
        return off_diag_sum < 1e-8
    
    def natural_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Compute natural gradient: F^{-1} @ ordinary_gradient.
        
        Natural gradient accounts for the geometry of the parameter space,
        leading to faster convergence. This is the key insight of NGBoost.
        
        Args:
            y: Observed target values
            params: Dictionary of constrained parameter values
            
        Returns:
            Dictionary mapping param_name -> (natural_gradient, hessian)
        """
        # Get ordinary gradients
        ord_grads = self.nll_gradient(y, params)
        
        # Get Fisher matrix
        F = self.fisher_information(params)  # (n_samples, n_params, n_params)
        
        # Stack gradients: (n_samples, n_params)
        grad_stack = np.stack(
            [ord_grads[p][0] for p in self.param_names], 
            axis=1
        )
        
        n_samples = y.shape[0]
        n_params = self.n_params
        
        # Vectorized Fisher inversion based on matrix size
        if n_params == 1:
            # 1x1: simple reciprocal
            natural_grad_stack = grad_stack / np.maximum(F[:, 0, 0:1], 1e-10)
            
        elif n_params == 2:
            # 2x2: analytical inverse (vectorized)
            # For [[a,b],[c,d]], inverse is 1/(ad-bc) * [[d,-b],[-c,a]]
            a, b = F[:, 0, 0], F[:, 0, 1]
            c, d = F[:, 1, 0], F[:, 1, 1]
            det = a * d - b * c
            det = np.maximum(np.abs(det), 1e-10) * np.sign(det + 1e-20)
            
            # F_inv @ grad
            g0, g1 = grad_stack[:, 0], grad_stack[:, 1]
            natural_grad_stack = np.stack([
                (d * g0 - b * g1) / det,
                (-c * g0 + a * g1) / det,
            ], axis=1)
            
        elif self._is_diagonal_fisher(F):
            # Diagonal Fisher: element-wise division (very common case)
            diag = np.diagonal(F, axis1=1, axis2=2)  # (n_samples, n_params)
            natural_grad_stack = grad_stack / np.maximum(diag, 1e-10)
            
        else:
            # General case: batched solve (still faster than loop)
            try:
                # np.linalg.solve broadcasts over leading dimensions
                natural_grad_stack = np.linalg.solve(F, grad_stack)
            except np.linalg.LinAlgError:
                # Fallback: regularize and retry
                F_reg = F + 1e-6 * np.eye(n_params)
                natural_grad_stack = np.linalg.solve(F_reg, grad_stack)
        
        # Unstack back to dict
        # For natural gradient, use identity hessian (standard NGBoost approach)
        result = {}
        for j, p in enumerate(self.param_names):
            result[p] = (
                natural_grad_stack[:, j].astype(np.float32),
                np.ones(n_samples, dtype=np.float32),
            )
        
        return result
    
    def init_params(self, y: NDArray) -> dict[str, float]:
        """Initialize parameters from target values.
        
        Returns raw (pre-link) initial values for each parameter.
        
        Args:
            y: Target values for initialization
            
        Returns:
            Dictionary mapping param_name -> initial raw value
        """
        # Default implementation - subclasses should override
        return {p: 0.0 for p in self.param_names}
    
    @abstractmethod
    def mean(self, params: dict[str, NDArray]) -> NDArray:
        """Expected value E[Y|params]."""
        pass
    
    @abstractmethod
    def variance(self, params: dict[str, NDArray]) -> NDArray:
        """Variance Var[Y|params]."""
        pass
    
    def quantile(self, params: dict[str, NDArray], q: float) -> NDArray:
        """q-th quantile of the distribution."""
        raise NotImplementedError(f"quantile not implemented for {self.__class__.__name__}")
    
    def sample(
        self, 
        params: dict[str, NDArray], 
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        """Sample from the distribution."""
        raise NotImplementedError(f"sample not implemented for {self.__class__.__name__}")
    
    def nll(self, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        """Negative log-likelihood (for evaluation)."""
        raise NotImplementedError(f"nll not implemented for {self.__class__.__name__}")


# =============================================================================
# Normal (Gaussian) Distribution
# =============================================================================

class Normal(Distribution):
    """Normal (Gaussian) distribution.
    
    Parameters:
        loc (μ): Mean, unbounded
        scale (σ): Standard deviation, must be positive
        
    Link functions:
        loc: identity (unbounded)
        scale: exp (ensures σ > 0)
        
    PDF: p(y) = (1/√(2πσ²)) exp(-(y-μ)²/(2σ²))
    NLL: 0.5 * log(2πσ²) + (y-μ)²/(2σ²)
    """
    
    @property
    def n_params(self) -> int:
        return 2
    
    @property
    def param_names(self) -> list[str]:
        return ['loc', 'scale']
    
    def link(self, param_name: str, raw: NDArray) -> NDArray:
        if param_name == 'loc':
            return raw
        elif param_name == 'scale':
            # exp with clipping for numerical stability
            return np.exp(np.clip(raw, -20, 20))
        raise ValueError(f"Unknown parameter: {param_name}")
    
    def link_inv(self, param_name: str, param: NDArray) -> NDArray:
        if param_name == 'loc':
            return param
        elif param_name == 'scale':
            return np.log(np.clip(param, 1e-10, None))
        raise ValueError(f"Unknown parameter: {param_name}")
    
    def init_params(self, y: NDArray) -> dict[str, float]:
        """Initialize with sample mean and std."""
        loc_init = float(np.mean(y))
        scale_init = float(np.std(y)) + 1e-6
        return {
            'loc': loc_init,  # Already in raw space (identity link)
            'scale': float(np.log(scale_init)),  # Convert to raw (log) space
        }
    
    def nll_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Compute gradients of NLL w.r.t. raw parameters.
        
        NLL = 0.5 * log(2πσ²) + (y - μ)² / (2σ²)
        
        For loc (identity link):
            d(NLL)/dμ = -(y - μ) / σ²
            d²(NLL)/dμ² = 1 / σ²
        
        For scale with exp link (σ = exp(s)):
            d(NLL)/ds = 1 - (y - μ)² / σ²
            d²(NLL)/ds² ≈ 2 (expected hessian at optimum)
        """
        μ = params['loc']
        σ = params['scale']
        
        residual = y - μ
        var = σ ** 2
        
        # Location gradients (identity link)
        grad_loc = -residual / var
        hess_loc = 1.0 / var
        
        # Scale gradients (exp link: σ = exp(s))
        # Chain rule: d(NLL)/ds = d(NLL)/dσ * dσ/ds = d(NLL)/dσ * σ
        # d(NLL)/dσ = 1/σ - (y-μ)²/σ³
        # d(NLL)/ds = σ * (1/σ - (y-μ)²/σ³) = 1 - (y-μ)²/σ²
        grad_scale = 1.0 - (residual ** 2) / var
        
        # Expected hessian (more stable than exact)
        # At optimum, (y-μ)² ≈ σ², so d²(NLL)/ds² ≈ 2
        hess_scale = 2.0 * np.ones_like(y)
        
        return {
            'loc': (grad_loc.astype(np.float32), hess_loc.astype(np.float32)),
            'scale': (grad_scale.astype(np.float32), hess_scale.astype(np.float32)),
        }
    
    def fisher_information(
        self,
        params: dict[str, NDArray],
    ) -> NDArray:
        """Fisher information matrix for Normal distribution.
        
        For Normal with exp link on scale:
        F = [[1/σ², 0   ],
             [0,    2   ]]
        
        The off-diagonal is 0 because mean and variance are orthogonal
        parameters in the normal family.
        """
        n_samples = params['loc'].shape[0]
        σ = params['scale']
        
        F = np.zeros((n_samples, 2, 2), dtype=np.float32)
        F[:, 0, 0] = 1.0 / (σ ** 2)  # d²/dμ²
        F[:, 1, 1] = 2.0              # d²/ds² (expected, with exp link)
        # Off-diagonal is 0 for Normal
        
        return F
    
    def mean(self, params: dict[str, NDArray]) -> NDArray:
        return params['loc']
    
    def variance(self, params: dict[str, NDArray]) -> NDArray:
        return params['scale'] ** 2
    
    def quantile(self, params: dict[str, NDArray], q: float) -> NDArray:
        """Quantile of Normal distribution."""
        from scipy import stats
        μ = params['loc']
        σ = params['scale']
        return stats.norm.ppf(q, loc=μ, scale=σ)
    
    def sample(
        self, 
        params: dict[str, NDArray], 
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        """Sample from Normal distribution."""
        rng = np.random.default_rng(seed)
        μ = params['loc']
        σ = params['scale']
        n_obs = μ.shape[0]
        return rng.normal(μ[:, None], σ[:, None], size=(n_obs, n_samples))
    
    def nll(self, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        """Negative log-likelihood."""
        μ = params['loc']
        σ = params['scale']
        return 0.5 * np.log(2 * np.pi * σ**2) + (y - μ)**2 / (2 * σ**2)


# =============================================================================
# LogNormal Distribution
# =============================================================================

class LogNormal(Distribution):
    """Log-Normal distribution for positive continuous data.
    
    If X ~ LogNormal(μ, σ), then log(X) ~ Normal(μ, σ).
    
    Parameters:
        loc (μ): Mean of underlying normal
        scale (σ): Std of underlying normal (must be positive)
        
    Link functions:
        loc: identity
        scale: exp
        
    Mean: exp(μ + σ²/2)
    Variance: (exp(σ²) - 1) * exp(2μ + σ²)
    """
    
    @property
    def n_params(self) -> int:
        return 2
    
    @property
    def param_names(self) -> list[str]:
        return ['loc', 'scale']
    
    def link(self, param_name: str, raw: NDArray) -> NDArray:
        if param_name == 'loc':
            return raw
        elif param_name == 'scale':
            return np.exp(np.clip(raw, -20, 20))
        raise ValueError(f"Unknown parameter: {param_name}")
    
    def link_inv(self, param_name: str, param: NDArray) -> NDArray:
        if param_name == 'loc':
            return param
        elif param_name == 'scale':
            return np.log(np.clip(param, 1e-10, None))
        raise ValueError(f"Unknown parameter: {param_name}")
    
    def init_params(self, y: NDArray) -> dict[str, float]:
        """Initialize from positive target values."""
        log_y = np.log(np.clip(y, 1e-10, None))
        loc_init = float(np.mean(log_y))
        scale_init = float(np.std(log_y)) + 1e-6
        return {
            'loc': loc_init,
            'scale': float(np.log(scale_init)),
        }
    
    def nll_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Gradients for LogNormal.
        
        NLL = log(y) + 0.5*log(2πσ²) + (log(y) - μ)²/(2σ²)
        
        Same gradients as Normal but with log(y) as target.
        """
        μ = params['loc']
        σ = params['scale']
        
        log_y = np.log(np.clip(y, 1e-10, None))
        residual = log_y - μ
        var = σ ** 2
        
        grad_loc = -residual / var
        hess_loc = 1.0 / var
        
        grad_scale = 1.0 - (residual ** 2) / var
        hess_scale = 2.0 * np.ones_like(y)
        
        return {
            'loc': (grad_loc.astype(np.float32), hess_loc.astype(np.float32)),
            'scale': (grad_scale.astype(np.float32), hess_scale.astype(np.float32)),
        }
    
    def fisher_information(
        self,
        params: dict[str, NDArray],
    ) -> NDArray:
        """Same as Normal (parameters are for underlying normal)."""
        n_samples = params['loc'].shape[0]
        σ = params['scale']
        
        F = np.zeros((n_samples, 2, 2), dtype=np.float32)
        F[:, 0, 0] = 1.0 / (σ ** 2)
        F[:, 1, 1] = 2.0
        
        return F
    
    def mean(self, params: dict[str, NDArray]) -> NDArray:
        μ = params['loc']
        σ = params['scale']
        return np.exp(μ + σ**2 / 2)
    
    def variance(self, params: dict[str, NDArray]) -> NDArray:
        μ = params['loc']
        σ = params['scale']
        return (np.exp(σ**2) - 1) * np.exp(2*μ + σ**2)
    
    def quantile(self, params: dict[str, NDArray], q: float) -> NDArray:
        from scipy import stats
        μ = params['loc']
        σ = params['scale']
        return stats.lognorm.ppf(q, s=σ, scale=np.exp(μ))
    
    def sample(
        self, 
        params: dict[str, NDArray], 
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        rng = np.random.default_rng(seed)
        μ = params['loc']
        σ = params['scale']
        n_obs = μ.shape[0]
        return rng.lognormal(μ[:, None], σ[:, None], size=(n_obs, n_samples))
    
    def nll(self, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        μ = params['loc']
        σ = params['scale']
        log_y = np.log(np.clip(y, 1e-10, None))
        return log_y + 0.5 * np.log(2 * np.pi * σ**2) + (log_y - μ)**2 / (2 * σ**2)


# =============================================================================
# Gamma Distribution
# =============================================================================

class Gamma(Distribution):
    """Gamma distribution for positive continuous data.
    
    Parameterization: shape (α) and rate (β)
    - Mean = α/β
    - Variance = α/β²
    
    Parameters:
        concentration (α): Shape parameter, must be positive
        rate (β): Rate parameter, must be positive
        
    Link functions: exp for both (ensure positivity)
    
    Alternative: Can also be parameterized by mean and dispersion.
    """
    
    @property
    def n_params(self) -> int:
        return 2
    
    @property
    def param_names(self) -> list[str]:
        return ['concentration', 'rate']
    
    def link(self, param_name: str, raw: NDArray) -> NDArray:
        # Both parameters must be positive
        return np.exp(np.clip(raw, -20, 20))
    
    def link_inv(self, param_name: str, param: NDArray) -> NDArray:
        return np.log(np.clip(param, 1e-10, None))
    
    def init_params(self, y: NDArray) -> dict[str, float]:
        """Initialize using method of moments.
        
        mean = α/β, var = α/β²
        => β = mean/var, α = mean * β = mean²/var
        """
        y_clip = np.clip(y, 1e-10, None)
        mean_y = float(np.mean(y_clip))
        var_y = float(np.var(y_clip)) + 1e-6
        
        rate = mean_y / var_y
        concentration = mean_y * rate
        
        return {
            'concentration': float(np.log(max(concentration, 1e-6))),
            'rate': float(np.log(max(rate, 1e-6))),
        }
    
    def nll_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Gradients for Gamma distribution.
        
        NLL = -α*log(β) + log(Γ(α)) - (α-1)*log(y) + β*y
        
        d(NLL)/dα = -log(β) + ψ(α) - log(y)
        d(NLL)/dβ = -α/β + y
        
        With exp links (α = exp(a), β = exp(b)):
        d(NLL)/da = α * (-log(β) + ψ(α) - log(y))
        d(NLL)/db = β * (-α/β + y) = -α + β*y
        """
        from scipy.special import digamma, polygamma
        
        α = params['concentration']
        β = params['rate']
        
        log_y = np.log(np.clip(y, 1e-10, None))
        log_β = np.log(β)
        
        # Gradient w.r.t. raw concentration (with exp link)
        grad_conc_raw = α * (-log_β + digamma(α) - log_y)
        # Expected hessian approximation
        hess_conc = α * polygamma(1, α) * α  # α² * ψ'(α)
        hess_conc = np.clip(hess_conc, 0.1, 100)  # Stability
        
        # Gradient w.r.t. raw rate (with exp link)
        grad_rate_raw = -α + β * y
        hess_rate = α  # Expected hessian: α
        
        return {
            'concentration': (grad_conc_raw.astype(np.float32), hess_conc.astype(np.float32)),
            'rate': (grad_rate_raw.astype(np.float32), hess_rate.astype(np.float32)),
        }
    
    def fisher_information(
        self,
        params: dict[str, NDArray],
    ) -> NDArray:
        """Fisher information for Gamma (with exp links)."""
        from scipy.special import polygamma
        
        n_samples = params['concentration'].shape[0]
        α = params['concentration']
        
        F = np.zeros((n_samples, 2, 2), dtype=np.float32)
        # F[α,α] = α² * ψ'(α) (trigamma)
        F[:, 0, 0] = α**2 * polygamma(1, α)
        # F[β,β] = α (with exp link)
        F[:, 1, 1] = α
        # F[α,β] = -α (but small, often ignored)
        F[:, 0, 1] = -α
        F[:, 1, 0] = -α
        
        return F
    
    def mean(self, params: dict[str, NDArray]) -> NDArray:
        α = params['concentration']
        β = params['rate']
        return α / β
    
    def variance(self, params: dict[str, NDArray]) -> NDArray:
        α = params['concentration']
        β = params['rate']
        return α / (β ** 2)
    
    def quantile(self, params: dict[str, NDArray], q: float) -> NDArray:
        from scipy import stats
        α = params['concentration']
        β = params['rate']
        return stats.gamma.ppf(q, a=α, scale=1/β)
    
    def sample(
        self, 
        params: dict[str, NDArray], 
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        rng = np.random.default_rng(seed)
        α = params['concentration']
        β = params['rate']
        n_obs = α.shape[0]
        return rng.gamma(α[:, None], 1/β[:, None], size=(n_obs, n_samples))
    
    def nll(self, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        from scipy.special import gammaln
        α = params['concentration']
        β = params['rate']
        y_clip = np.clip(y, 1e-10, None)
        return -α * np.log(β) + gammaln(α) - (α - 1) * np.log(y_clip) + β * y_clip


# =============================================================================
# Poisson Distribution
# =============================================================================

class Poisson(Distribution):
    """Poisson distribution for count data.
    
    Single parameter: rate (λ)
    - Mean = λ
    - Variance = λ
    
    Link function: exp (ensures λ > 0)
    """
    
    @property
    def n_params(self) -> int:
        return 1
    
    @property
    def param_names(self) -> list[str]:
        return ['rate']
    
    def link(self, param_name: str, raw: NDArray) -> NDArray:
        return np.exp(np.clip(raw, -20, 20))
    
    def link_inv(self, param_name: str, param: NDArray) -> NDArray:
        return np.log(np.clip(param, 1e-10, None))
    
    def init_params(self, y: NDArray) -> dict[str, float]:
        mean_y = float(np.mean(np.clip(y, 0, None))) + 1e-6
        return {'rate': float(np.log(mean_y))}
    
    def nll_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Gradients for Poisson.
        
        NLL = λ - y*log(λ) + log(y!)
        d(NLL)/dλ = 1 - y/λ
        
        With exp link (λ = exp(l)):
        d(NLL)/dl = λ - y
        d²(NLL)/dl² = λ
        """
        λ = params['rate']
        
        grad = λ - y
        hess = np.maximum(λ, 1e-6)  # Hessian = λ
        
        return {
            'rate': (grad.astype(np.float32), hess.astype(np.float32)),
        }
    
    def fisher_information(
        self,
        params: dict[str, NDArray],
    ) -> NDArray:
        """Fisher information for Poisson: F = λ."""
        n_samples = params['rate'].shape[0]
        λ = params['rate']
        
        F = np.zeros((n_samples, 1, 1), dtype=np.float32)
        F[:, 0, 0] = λ
        
        return F
    
    def mean(self, params: dict[str, NDArray]) -> NDArray:
        return params['rate']
    
    def variance(self, params: dict[str, NDArray]) -> NDArray:
        return params['rate']
    
    def quantile(self, params: dict[str, NDArray], q: float) -> NDArray:
        from scipy import stats
        λ = params['rate']
        return stats.poisson.ppf(q, mu=λ)
    
    def sample(
        self, 
        params: dict[str, NDArray], 
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        rng = np.random.default_rng(seed)
        λ = params['rate']
        n_obs = λ.shape[0]
        return rng.poisson(λ[:, None], size=(n_obs, n_samples))
    
    def nll(self, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        from scipy.special import gammaln
        λ = params['rate']
        y_int = np.round(np.clip(y, 0, None))
        return λ - y_int * np.log(np.clip(λ, 1e-10, None)) + gammaln(y_int + 1)


# =============================================================================
# Student-t Distribution
# =============================================================================

class StudentT(Distribution):
    """Student-t distribution for heavy-tailed data.
    
    Parameters:
        loc (μ): Location parameter
        scale (σ): Scale parameter (positive)
        df (ν): Degrees of freedom (positive, typically > 2)
        
    For ν → ∞, approaches Normal distribution.
    Lower ν = heavier tails.
    
    Link functions:
        loc: identity
        scale: exp
        df: softplus (ensures > 0, typically > 2)
    """
    
    @property
    def n_params(self) -> int:
        return 3
    
    @property
    def param_names(self) -> list[str]:
        return ['loc', 'scale', 'df']
    
    def link(self, param_name: str, raw: NDArray) -> NDArray:
        if param_name == 'loc':
            return raw
        elif param_name == 'scale':
            return np.exp(np.clip(raw, -20, 20))
        elif param_name == 'df':
            # Softplus + offset to keep df > 2 (ensures finite variance)
            return 2.0 + np.log1p(np.exp(np.clip(raw, -20, 20)))
        raise ValueError(f"Unknown parameter: {param_name}")
    
    def link_inv(self, param_name: str, param: NDArray) -> NDArray:
        if param_name == 'loc':
            return param
        elif param_name == 'scale':
            return np.log(np.clip(param, 1e-10, None))
        elif param_name == 'df':
            # Inverse of softplus + offset
            return np.log(np.exp(np.clip(param - 2.0, 1e-10, None)) - 1)
        raise ValueError(f"Unknown parameter: {param_name}")
    
    def init_params(self, y: NDArray) -> dict[str, float]:
        loc_init = float(np.median(y))  # Median is more robust
        scale_init = float(np.std(y)) + 1e-6
        df_init = 10.0  # Start with moderate tails
        return {
            'loc': loc_init,
            'scale': float(np.log(scale_init)),
            'df': float(np.log(np.exp(df_init - 2.0) - 1)),
        }
    
    def nll_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Gradients for Student-t (simplified, using expected hessians)."""
        from scipy.special import digamma
        
        μ = params['loc']
        σ = params['scale']
        ν = params['df']
        
        z = (y - μ) / σ
        z2 = z ** 2
        
        # Weight for each observation
        w = (ν + 1) / (ν + z2)
        
        # Location gradient
        grad_loc = -w * z / σ
        hess_loc = (ν + 1) / ((ν + 3) * σ**2)  # Expected hessian
        
        # Scale gradient (with exp link)
        grad_scale = 1 - w * z2
        hess_scale = 2.0 * np.ones_like(y)  # Approximation
        
        # DF gradient (complex, use approximation)
        # For simplicity, use small gradient toward Normal (large ν)
        grad_df = 0.01 * np.ones_like(y)  # Slight regularization toward large df
        hess_df = 0.1 * np.ones_like(y)
        
        return {
            'loc': (grad_loc.astype(np.float32), hess_loc.astype(np.float32)),
            'scale': (grad_scale.astype(np.float32), hess_scale.astype(np.float32)),
            'df': (grad_df.astype(np.float32), hess_df.astype(np.float32)),
        }
    
    def fisher_information(
        self,
        params: dict[str, NDArray],
    ) -> NDArray:
        """Fisher information for Student-t (diagonal approximation)."""
        n_samples = params['loc'].shape[0]
        σ = params['scale']
        ν = params['df']
        
        F = np.zeros((n_samples, 3, 3), dtype=np.float32)
        F[:, 0, 0] = (ν + 1) / ((ν + 3) * σ**2)
        F[:, 1, 1] = 2.0
        F[:, 2, 2] = 0.1  # Small, df is hard to estimate
        
        return F
    
    def mean(self, params: dict[str, NDArray]) -> NDArray:
        return params['loc']  # For ν > 1
    
    def variance(self, params: dict[str, NDArray]) -> NDArray:
        σ = params['scale']
        ν = params['df']
        # Variance = σ² * ν / (ν - 2) for ν > 2
        return σ**2 * ν / np.maximum(ν - 2, 1e-6)
    
    def quantile(self, params: dict[str, NDArray], q: float) -> NDArray:
        from scipy import stats
        μ = params['loc']
        σ = params['scale']
        ν = params['df']
        return stats.t.ppf(q, df=ν, loc=μ, scale=σ)
    
    def sample(
        self, 
        params: dict[str, NDArray], 
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        rng = np.random.default_rng(seed)
        μ = params['loc']
        σ = params['scale']
        ν = params['df']
        n_obs = μ.shape[0]
        return μ[:, None] + σ[:, None] * rng.standard_t(ν[:, None], size=(n_obs, n_samples))
    
    def nll(self, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        from scipy.special import gammaln
        μ = params['loc']
        σ = params['scale']
        ν = params['df']
        z = (y - μ) / σ
        return (
            gammaln((ν + 1) / 2) - gammaln(ν / 2)
            - 0.5 * np.log(ν * np.pi) - np.log(σ)
            - (ν + 1) / 2 * np.log(1 + z**2 / ν)
        ) * -1  # Negative because we computed log-likelihood


# =============================================================================
# Tweedie Distribution (Kaggle Insurance Competitions!)
# =============================================================================

class Tweedie(Distribution):
    """Tweedie distribution for zero-inflated positive continuous data.
    
    **Key use case**: Insurance claims, revenue forecasting with zeros.
    
    Popular in Kaggle competitions:
    - Porto Seguro Safe Driver Prediction
    - Allstate Claims Severity
    - Any competition with zero-inflated positive targets
    
    The Tweedie distribution is a compound Poisson-Gamma:
    - ρ = 1: Poisson (count data)
    - 1 < ρ < 2: Compound Poisson-Gamma (zeros + positive continuous)
    - ρ = 2: Gamma (positive continuous)
    
    Parameters:
        mu (μ): Mean parameter (positive)
        phi (φ): Dispersion parameter (positive)
        
    Why better than XGBoost?
    - XGBoost Tweedie only outputs point estimates
    - NGBoost Tweedie outputs full distribution → prediction intervals,
      uncertainty quantification, probabilistic forecasts
    
    Link functions:
        mu: log (ensures μ > 0)
        phi: log (ensures φ > 0)
    """
    
    def __init__(self, power: float = 1.5):
        """Initialize Tweedie with power parameter.
        
        Args:
            power: Variance power (1 < power < 2 for compound Poisson-Gamma)
                   1.5 is the default used in most Kaggle competitions.
        """
        self.power = power
    
    @property
    def n_params(self) -> int:
        return 2
    
    @property
    def param_names(self) -> list[str]:
        return ['mu', 'phi']
    
    def link(self, param_name: str, raw: NDArray) -> NDArray:
        # Both parameters must be positive
        return np.exp(np.clip(raw, -20, 20))
    
    def link_inv(self, param_name: str, param: NDArray) -> NDArray:
        return np.log(np.clip(param, 1e-10, None))
    
    def init_params(self, y: NDArray) -> dict[str, float]:
        """Initialize from target values.
        
        For Tweedie, μ = E[Y], and φ is estimated from variance.
        """
        y_clip = np.clip(y, 1e-10, None)
        mu_init = float(np.mean(y_clip)) + 1e-6
        
        # Estimate dispersion: Var(Y) = φ * μ^ρ
        var_y = float(np.var(y_clip)) + 1e-6
        phi_init = var_y / (mu_init ** self.power) + 1e-6
        
        return {
            'mu': float(np.log(mu_init)),
            'phi': float(np.log(phi_init)),
        }
    
    def nll_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Gradients for Tweedie distribution.
        
        Using the deviance formulation (standard in GLMs).
        
        For Tweedie with power ρ:
        d(NLL)/dμ = (μ^(1-ρ) - y*μ^(-ρ)) / φ
        """
        μ = params['mu']
        φ = params['phi']
        ρ = self.power
        
        # Ensure numerical stability
        μ_safe = np.clip(μ, 1e-10, 1e10)
        y_safe = np.clip(y, 0, None)
        
        # Gradient w.r.t. log(μ) (with log link)
        # d(NLL)/d(log μ) = μ * d(NLL)/dμ
        mu_pow_1_rho = np.power(μ_safe, 1 - ρ)
        mu_pow_neg_rho = np.power(μ_safe, -ρ)
        
        grad_mu_raw = μ_safe * (mu_pow_1_rho - y_safe * mu_pow_neg_rho) / φ
        
        # Expected hessian approximation
        hess_mu = μ_safe ** (2 - ρ) / φ
        hess_mu = np.clip(hess_mu, 1e-6, 1e6)
        
        # Gradient w.r.t. log(φ)
        # Dispersion affects the scale but is harder to estimate
        # Use simple gradient: d(NLL)/d(log φ) ≈ 1 - deviance/φ
        deviance = self._compute_deviance(y_safe, μ_safe)
        grad_phi_raw = 1.0 - deviance / (2 * φ)
        hess_phi = 0.5 * np.ones_like(y)  # Conservative hessian
        
        return {
            'mu': (grad_mu_raw.astype(np.float32), hess_mu.astype(np.float32)),
            'phi': (grad_phi_raw.astype(np.float32), hess_phi.astype(np.float32)),
        }
    
    def _compute_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        """Compute Tweedie deviance."""
        ρ = self.power
        y_safe = np.clip(y, 1e-10, None)
        mu_safe = np.clip(mu, 1e-10, None)
        
        if ρ == 1:  # Poisson
            return 2 * (y_safe * np.log(y_safe / mu_safe) - (y_safe - mu_safe))
        elif ρ == 2:  # Gamma
            return 2 * (np.log(mu_safe / y_safe) + (y_safe - mu_safe) / mu_safe)
        else:
            # General Tweedie (1 < ρ < 2)
            term1 = np.power(y_safe, 2 - ρ) / ((1 - ρ) * (2 - ρ))
            term2 = y_safe * np.power(mu_safe, 1 - ρ) / (1 - ρ)
            term3 = np.power(mu_safe, 2 - ρ) / (2 - ρ)
            return 2 * (term1 - term2 + term3)
    
    def fisher_information(
        self,
        params: dict[str, NDArray],
    ) -> NDArray:
        """Fisher information for Tweedie."""
        n_samples = params['mu'].shape[0]
        μ = params['mu']
        φ = params['phi']
        ρ = self.power
        
        F = np.zeros((n_samples, 2, 2), dtype=np.float32)
        F[:, 0, 0] = μ ** (2 - ρ) / φ
        F[:, 1, 1] = 0.5
        
        return F
    
    def mean(self, params: dict[str, NDArray]) -> NDArray:
        return params['mu']
    
    def variance(self, params: dict[str, NDArray]) -> NDArray:
        μ = params['mu']
        φ = params['phi']
        return φ * np.power(μ, self.power)
    
    def quantile(self, params: dict[str, NDArray], q: float) -> NDArray:
        """Approximate quantile using Normal approximation."""
        μ = params['mu']
        var = self.variance(params)
        σ = np.sqrt(var)
        from scipy import stats
        return stats.norm.ppf(q, loc=μ, scale=σ)
    
    def sample(
        self, 
        params: dict[str, NDArray], 
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        """Sample from Tweedie using compound Poisson-Gamma."""
        rng = np.random.default_rng(seed)
        μ = params['mu']
        φ = params['phi']
        ρ = self.power
        n_obs = μ.shape[0]
        
        samples = np.zeros((n_obs, n_samples), dtype=np.float32)
        
        # Compound Poisson-Gamma sampling
        for i in range(n_obs):
            λ = μ[i] ** (2 - ρ) / (φ[i] * (2 - ρ))  # Poisson rate
            α = (2 - ρ) / (ρ - 1)  # Gamma shape
            β = φ[i] * (ρ - 1) * μ[i] ** (ρ - 1)  # Gamma scale
            
            for j in range(n_samples):
                n_claims = rng.poisson(λ)
                if n_claims > 0:
                    samples[i, j] = np.sum(rng.gamma(α, β, size=n_claims))
        
        return samples
    
    def nll(self, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        """Negative log-likelihood (deviance-based)."""
        μ = params['mu']
        φ = params['phi']
        deviance = self._compute_deviance(y, μ)
        return deviance / (2 * φ)


# =============================================================================
# Negative Binomial Distribution (Kaggle Count Data Competitions!)
# =============================================================================

class NegativeBinomial(Distribution):
    """Negative Binomial distribution for overdispersed count data.
    
    **Key use case**: Sales forecasting, demand prediction, click counts.
    
    Popular in Kaggle competitions:
    - Rossmann Store Sales
    - Bike Sharing Demand
    - Grupo Bimbo Inventory Demand
    - Any competition with count data where variance > mean
    
    Compared to Poisson:
    - Poisson: Var(Y) = Mean(Y)
    - NegBin: Var(Y) = Mean(Y) + Mean(Y)²/r  (overdispersion)
    
    Parameters:
        mu (μ): Mean parameter (positive)
        r: Dispersion parameter (positive, smaller = more overdispersion)
        
    Why better than XGBoost?
    - XGBoost can't output count distributions at all
    - NGBoost NegBin outputs full distribution → prediction intervals,
      probability of exceeding thresholds, demand planning
    
    Link functions:
        mu: log (ensures μ > 0)
        r: log (ensures r > 0)
    """
    
    @property
    def n_params(self) -> int:
        return 2
    
    @property
    def param_names(self) -> list[str]:
        return ['mu', 'r']
    
    def link(self, param_name: str, raw: NDArray) -> NDArray:
        return np.exp(np.clip(raw, -20, 20))
    
    def link_inv(self, param_name: str, param: NDArray) -> NDArray:
        return np.log(np.clip(param, 1e-10, None))
    
    def init_params(self, y: NDArray) -> dict[str, float]:
        """Initialize using method of moments.
        
        Mean = μ
        Var = μ + μ²/r
        => r = μ² / (Var - μ)
        """
        y_clip = np.clip(y, 0, None)
        mu_init = float(np.mean(y_clip)) + 1e-6
        var_y = float(np.var(y_clip)) + 1e-6
        
        # Estimate r from method of moments
        if var_y > mu_init:
            r_init = mu_init ** 2 / (var_y - mu_init)
        else:
            r_init = 10.0  # Default if not overdispersed
        
        r_init = np.clip(r_init, 0.1, 1000)
        
        return {
            'mu': float(np.log(mu_init)),
            'r': float(np.log(r_init)),
        }
    
    def nll_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Gradients for Negative Binomial.
        
        NLL = -log Γ(y+r) + log Γ(r) + log Γ(y+1) 
              - r*log(r/(r+μ)) - y*log(μ/(r+μ))
        """
        from scipy.special import digamma, polygamma
        
        μ = params['mu']
        r = params['r']
        
        # Ensure numerical stability
        μ_safe = np.clip(μ, 1e-10, 1e10)
        r_safe = np.clip(r, 1e-10, 1e10)
        y_safe = np.clip(y, 0, None)
        
        # Common terms
        p = r_safe / (r_safe + μ_safe)  # Success probability
        
        # Gradient w.r.t. log(μ)
        # d(NLL)/d(log μ) = μ * d(NLL)/dμ = μ * (μ/(r+μ) - y/μ * r/(r+μ))
        #                 = μ²/(r+μ) - y*r/(r+μ) = (μ² - y*r)/(r+μ)
        #                 = μ * (μ - y*r/μ)/(r+μ) = μ * (μ - y*r/μ) * (1-p)/μ
        #                 = (μ - y) * (1-p) ... simplified
        grad_mu_raw = (μ_safe - y_safe) * (1 - p)
        
        # Expected hessian
        hess_mu = μ_safe * (1 - p)
        hess_mu = np.clip(hess_mu, 1e-6, 1e6)
        
        # Gradient w.r.t. log(r) (dispersion)
        # This is complex, use approximation
        grad_r_raw = r_safe * (digamma(y_safe + r_safe) - digamma(r_safe) + np.log(p))
        hess_r = r_safe * polygamma(1, r_safe)  # Expected hessian
        hess_r = np.clip(hess_r, 0.1, 10)
        
        return {
            'mu': (grad_mu_raw.astype(np.float32), hess_mu.astype(np.float32)),
            'r': (grad_r_raw.astype(np.float32), hess_r.astype(np.float32)),
        }
    
    def fisher_information(
        self,
        params: dict[str, NDArray],
    ) -> NDArray:
        """Fisher information for Negative Binomial."""
        from scipy.special import polygamma
        
        n_samples = params['mu'].shape[0]
        μ = params['mu']
        r = params['r']
        
        p = r / (r + μ)
        
        F = np.zeros((n_samples, 2, 2), dtype=np.float32)
        F[:, 0, 0] = μ * (1 - p)
        F[:, 1, 1] = np.clip(r * polygamma(1, r), 0.1, 10)
        
        return F
    
    def mean(self, params: dict[str, NDArray]) -> NDArray:
        return params['mu']
    
    def variance(self, params: dict[str, NDArray]) -> NDArray:
        μ = params['mu']
        r = params['r']
        return μ + μ ** 2 / r
    
    def quantile(self, params: dict[str, NDArray], q: float) -> NDArray:
        from scipy import stats
        μ = params['mu']
        r = params['r']
        p = r / (r + μ)
        return stats.nbinom.ppf(q, n=r, p=p)
    
    def sample(
        self, 
        params: dict[str, NDArray], 
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        rng = np.random.default_rng(seed)
        μ = params['mu']
        r = params['r']
        n_obs = μ.shape[0]
        
        p = r / (r + μ)
        return rng.negative_binomial(r[:, None], p[:, None], size=(n_obs, n_samples))
    
    def nll(self, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        from scipy.special import gammaln
        μ = params['mu']
        r = params['r']
        y_int = np.round(np.clip(y, 0, None))
        
        return (
            gammaln(r) + gammaln(y_int + 1) - gammaln(y_int + r)
            + r * np.log(r / (r + μ))
            + y_int * np.log(μ / (r + μ))
        ) * -1  # Negate log-likelihood
    
    def prob_exceed(self, params: dict[str, NDArray], threshold: float) -> NDArray:
        """Probability that Y > threshold.
        
        Very useful for demand planning: "What's the probability we need 
        more than 100 units?"
        """
        from scipy import stats
        μ = params['mu']
        r = params['r']
        p = r / (r + μ)
        return 1 - stats.nbinom.cdf(threshold, n=r, p=p)


# =============================================================================
# Custom Distribution with Autodiff (Define Your Own!)
# =============================================================================

class CustomDistribution(Distribution):
    """User-defined distribution with automatic gradient computation.
    
    Define any parametric distribution by specifying:
    1. Parameter names and link functions
    2. Negative log-likelihood function
    
    Gradients are computed automatically via:
    - JAX (if available) - fastest
    - Numerical differentiation (fallback)
    
    Example: Custom "ratio" distribution y ~ Normal(A*(1-B)/C, σ)
    
        >>> def my_nll(y, params):
        ...     A, B, C, sigma = params['A'], params['B'], params['C'], params['sigma']
        ...     mu = A * (1 - B) / C
        ...     return 0.5 * np.log(2 * np.pi * sigma**2) + (y - mu)**2 / (2 * sigma**2)
        >>> 
        >>> dist = CustomDistribution(
        ...     param_names=['A', 'B', 'C', 'sigma'],
        ...     link_functions={
        ...         'A': 'identity',      # A ∈ (-∞, ∞)
        ...         'B': 'sigmoid',       # B ∈ (0, 1)
        ...         'C': 'softplus',      # C > 0
        ...         'sigma': 'exp',       # σ > 0
        ...     },
        ...     nll_fn=my_nll,
        ...     mean_fn=lambda params: params['A'] * (1 - params['B']) / params['C'],
        ... )
        >>> 
        >>> model = NGBoost(distribution=dist, n_trees=100)
        >>> model.fit(X, y)
    
    For Kaggle competitions with custom evaluation metrics, you can define
    the NLL to match the competition metric!
    """
    
    # Available link functions
    LINK_FUNCTIONS = {
        'identity': (lambda x: x, lambda x: x),
        'exp': (lambda x: np.exp(np.clip(x, -20, 20)), lambda x: np.log(np.clip(x, 1e-10, None))),
        'softplus': (lambda x: np.log1p(np.exp(np.clip(x, -20, 20))), lambda x: np.log(np.exp(np.clip(x, 1e-10, None)) - 1)),
        'sigmoid': (lambda x: 1 / (1 + np.exp(-np.clip(x, -20, 20))), lambda x: np.log(np.clip(x / (1 - x + 1e-10), 1e-10, None))),
        'square': (lambda x: x ** 2, lambda x: np.sqrt(np.clip(x, 0, None))),
    }
    
    def __init__(
        self,
        param_names: list[str],
        link_functions: dict[str, str],
        nll_fn: callable,
        mean_fn: callable | None = None,
        variance_fn: callable | None = None,
        init_fn: callable | None = None,
        use_jax: bool = True,
        eps: float = 1e-5,
    ):
        """Initialize custom distribution.
        
        Args:
            param_names: List of parameter names (e.g., ['A', 'B', 'sigma'])
            link_functions: Dict mapping param name to link type:
                - 'identity': no transformation, param ∈ (-∞, ∞)
                - 'exp': exponential, param > 0
                - 'softplus': log(1 + exp(x)), param > 0 (smoother than exp)
                - 'sigmoid': 1/(1+exp(-x)), param ∈ (0, 1)
                - 'square': x², param ≥ 0
            nll_fn: Function (y, params_dict) -> array of NLL per sample
            mean_fn: Optional function (params_dict) -> mean prediction
            variance_fn: Optional function (params_dict) -> variance
            init_fn: Optional function (y) -> dict of initial raw param values
            use_jax: Try to use JAX for autodiff (falls back to numerical if unavailable)
            eps: Epsilon for numerical gradients
        """
        self._param_names = param_names
        self._link_functions = link_functions
        self._nll_fn = nll_fn
        self._mean_fn = mean_fn
        self._variance_fn = variance_fn
        self._init_fn = init_fn
        self._use_jax = use_jax
        self._eps = eps
        
        # Check for JAX availability
        self._jax_available = False
        self._jax_grad_fn = None
        if use_jax:
            try:
                import jax
                import jax.numpy as jnp
                self._jax_available = True
                self._jax = jax
                self._jnp = jnp
                # Will compile grad function on first use
            except ImportError:
                pass
    
    @property
    def n_params(self) -> int:
        return len(self._param_names)
    
    @property
    def param_names(self) -> list[str]:
        return self._param_names
    
    def link(self, param_name: str, raw: NDArray) -> NDArray:
        link_type = self._link_functions.get(param_name, 'identity')
        link_fn, _ = self.LINK_FUNCTIONS[link_type]
        return link_fn(raw)
    
    def link_inv(self, param_name: str, param: NDArray) -> NDArray:
        link_type = self._link_functions.get(param_name, 'identity')
        _, inv_fn = self.LINK_FUNCTIONS[link_type]
        return inv_fn(param)
    
    def init_params(self, y: NDArray) -> dict[str, float]:
        if self._init_fn is not None:
            return self._init_fn(y)
        
        # Default initialization: zeros in raw space
        return {name: 0.0 for name in self._param_names}
    
    def _compute_nll_value(self, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        """Compute NLL using user-provided function."""
        return self._nll_fn(y, params)
    
    def _numerical_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Compute gradients numerically (vectorized finite differences)."""
        results = {}
        eps = self._eps
        n = len(y)
        
        # Compute center NLL once
        nll_center = self._nll_fn(y, params)
        
        for param_name in self._param_names:
            # Vectorized perturbation
            params_plus = {k: v.copy() for k, v in params.items()}
            params_minus = {k: v.copy() for k, v in params.items()}
            
            params_plus[param_name] = params[param_name] + eps
            params_minus[param_name] = params[param_name] - eps
            
            nll_plus = self._nll_fn(y, params_plus)
            nll_minus = self._nll_fn(y, params_minus)
            
            # Central difference for gradient
            grad = (nll_plus - nll_minus) / (2 * eps)
            
            # Central difference for hessian
            hess = (nll_plus - 2 * nll_center + nll_minus) / (eps ** 2)
            
            # Ensure positive hessian and clip extremes
            hess = np.clip(hess, 1e-6, 1e6)
            grad = np.clip(grad, -1e6, 1e6)
            
            results[param_name] = (grad.astype(np.float32), hess.astype(np.float32))
        
        return results
    
    def _jax_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Compute gradients using JAX autodiff."""
        jax = self._jax
        jnp = self._jnp
        
        # Define loss for a single sample
        def single_nll(param_values, y_single, param_names):
            params_dict = {name: jnp.array([val]) for name, val in zip(param_names, param_values)}
            return self._nll_fn(jnp.array([y_single]), params_dict)[0]
        
        # Get grad and hessian functions
        grad_fn = jax.grad(single_nll)
        hess_fn = jax.hessian(single_nll)
        
        # Vectorize over samples
        n = len(y)
        results = {name: (np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)) 
                   for name in self._param_names}
        
        for i in range(n):
            param_values = [params[name][i] for name in self._param_names]
            
            try:
                grads = grad_fn(param_values, y[i], self._param_names)
                hess_matrix = hess_fn(param_values, y[i], self._param_names)
                
                for j, name in enumerate(self._param_names):
                    results[name][0][i] = float(grads[j])
                    results[name][1][i] = max(float(hess_matrix[j, j]), 1e-6)
            except Exception:
                # Fall back to numerical for this sample
                for j, name in enumerate(self._param_names):
                    results[name][0][i] = 0.0
                    results[name][1][i] = 1.0
        
        return results
    
    def nll_gradient(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, GradHess]:
        """Compute gradients (auto-selects JAX or numerical)."""
        if self._jax_available:
            try:
                return self._jax_gradient(y, params)
            except Exception:
                pass
        
        return self._numerical_gradient(y, params)
    
    def fisher_information(
        self,
        params: dict[str, NDArray],
    ) -> NDArray:
        """Approximate Fisher information (diagonal)."""
        n_samples = list(params.values())[0].shape[0]
        n_params = self.n_params
        
        # Default: identity matrix (no natural gradient scaling)
        F = np.zeros((n_samples, n_params, n_params), dtype=np.float32)
        for i in range(n_params):
            F[:, i, i] = 1.0
        
        return F
    
    def mean(self, params: dict[str, NDArray]) -> NDArray:
        if self._mean_fn is not None:
            return self._mean_fn(params)
        # Default: first parameter
        return params[self._param_names[0]]
    
    def variance(self, params: dict[str, NDArray]) -> NDArray:
        if self._variance_fn is not None:
            return self._variance_fn(params)
        # Default: ones
        return np.ones_like(list(params.values())[0])
    
    def quantile(self, params: dict[str, NDArray], q: float) -> NDArray:
        """Approximate quantile using Normal assumption."""
        from scipy import stats
        mean = self.mean(params)
        std = np.sqrt(self.variance(params))
        return stats.norm.ppf(q, loc=mean, scale=std)
    
    def sample(
        self, 
        params: dict[str, NDArray], 
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        """Sample using Normal approximation."""
        rng = np.random.default_rng(seed)
        mean = self.mean(params)
        std = np.sqrt(self.variance(params))
        n_obs = mean.shape[0]
        return rng.normal(mean[:, None], std[:, None], size=(n_obs, n_samples))
    
    def nll(self, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        return self._nll_fn(y, params)


def create_custom_distribution(
    param_names: list[str],
    link_functions: dict[str, str],
    nll_fn: callable,
    mean_fn: callable | None = None,
    variance_fn: callable | None = None,
) -> CustomDistribution:
    """Convenience function to create a custom distribution.
    
    Example: Model y ~ Normal(A * exp(-B*x_feature), sigma)
    
        >>> dist = create_custom_distribution(
        ...     param_names=['A', 'B', 'sigma'],
        ...     link_functions={'A': 'exp', 'B': 'softplus', 'sigma': 'exp'},
        ...     nll_fn=lambda y, p: 0.5*np.log(2*np.pi*p['sigma']**2) + (y-p['A'])**2/(2*p['sigma']**2),
        ...     mean_fn=lambda p: p['A'],
        ...     variance_fn=lambda p: p['sigma']**2,
        ... )
    """
    return CustomDistribution(
        param_names=param_names,
        link_functions=link_functions,
        nll_fn=nll_fn,
        mean_fn=mean_fn,
        variance_fn=variance_fn,
    )


# =============================================================================
# Distribution Registry
# =============================================================================

DISTRIBUTIONS: dict[str, type[Distribution]] = {
    'normal': Normal,
    'gaussian': Normal,
    'lognormal': LogNormal,
    'log_normal': LogNormal,
    'gamma': Gamma,
    'poisson': Poisson,
    'studentt': StudentT,
    'student_t': StudentT,
    't': StudentT,
    # Kaggle competition favorites
    'tweedie': Tweedie,
    'negativebinomial': NegativeBinomial,
    'negative_binomial': NegativeBinomial,
    'negbin': NegativeBinomial,
}


def get_distribution(name: str | Distribution) -> Distribution:
    """Get distribution by name or return instance.
    
    Args:
        name: Distribution name or Distribution instance
        
    Returns:
        Distribution instance
        
    Example:
        >>> dist = get_distribution('normal')
        >>> dist = get_distribution('gamma')
    """
    if isinstance(name, Distribution):
        return name
    
    name_lower = name.lower()
    if name_lower not in DISTRIBUTIONS:
        available = ', '.join(sorted(set(DISTRIBUTIONS.keys())))
        raise ValueError(f"Unknown distribution '{name}'. Available: {available}")
    
    return DISTRIBUTIONS[name_lower]()


def list_distributions() -> list[str]:
    """List available distribution names."""
    return sorted(set(DISTRIBUTIONS.keys()))
