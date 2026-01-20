"""Distributional Gradient Boosting and NGBoost.

Phase 15.2-15.3: Probabilistic prediction via distributional regression.

Predicts full probability distributions instead of point estimates.
Trains separate tree ensembles for each distribution parameter.

Classes:
- DistributionalGBDT: Uses ordinary gradient descent
- NGBoost: Uses natural gradient descent (faster convergence)

Example:
    ```python
    import openboost as ob
    
    # Standard distributional GBDT
    model = ob.DistributionalGBDT(distribution='normal', n_trees=100)
    model.fit(X_train, y_train)
    
    # Get distribution parameters
    output = model.predict_distribution(X_test)
    mu, sigma = output.params['loc'], output.params['scale']
    
    # Get prediction intervals
    lower, upper = output.interval(alpha=0.1)  # 90% interval
    
    # Sample from predicted distribution
    samples = output.sample(n_samples=100)
    
    # NaturalBoost (recommended)
    model = ob.NaturalBoostNormal(n_trees=500)
    model.fit(X_train, y_train)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .._array import BinnedArray, array
from .._backends import is_cuda
from .._distributions import (
    Distribution,
    DistributionOutput,
    get_distribution,
)
from .._core._tree import fit_tree
from .._core._growth import TreeStructure
from .._persistence import PersistenceMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class DistributionalGBDT(PersistenceMixin):
    """Distributional Gradient Boosting for probabilistic prediction.
    
    Trains K tree ensembles, where K = number of distribution parameters.
    Each ensemble predicts one parameter (e.g., mean, variance).
    Uses ordinary gradient descent.
    
    For faster convergence, consider using NGBoost (natural gradient).
    
    Args:
        distribution: Distribution name ('normal', 'gamma', 'poisson', etc.)
                     or Distribution instance
        n_trees: Number of boosting rounds
        max_depth: Maximum depth of each tree
        learning_rate: Shrinkage factor applied to each tree
        min_child_weight: Minimum sum of hessian in a leaf
        reg_lambda: L2 regularization on leaf values
        reg_alpha: L1 regularization on leaf values
        subsample: Row sampling ratio (0.0-1.0)
        colsample_bytree: Column sampling ratio (0.0-1.0)
        n_bins: Number of bins for histogram building
        
    Attributes:
        trees_: Dict mapping param_name -> list of trees
        distribution_: Fitted Distribution instance
        
    Example:
        ```python
        model = DistributionalGBDT(distribution='normal', n_trees=100)
        model.fit(X_train, y_train)
        
        # Point prediction (mean)
        y_pred = model.predict(X_test)
        
        # Full distribution
        output = model.predict_distribution(X_test)
        lower, upper = output.interval(alpha=0.1)
        ```
    """
    
    distribution: (
        Literal['normal', 'lognormal', 'gamma', 'poisson', 'studentt', 'tweedie', 'negbin']
        | Distribution
    ) = 'normal'
    n_trees: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_bins: int = 256
    
    # Fitted attributes (not init)
    trees_: dict[str, list[TreeStructure]] = field(default_factory=dict, init=False, repr=False)
    distribution_: Distribution | None = field(default=None, init=False, repr=False)
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _base_scores: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    n_features_in_: int = field(default=0, init=False, repr=False)
    
    def fit(self, X: NDArray, y: NDArray) -> "DistributionalGBDT":
        """Fit the distributional gradient boosting model.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
            
        Returns:
            self: Fitted model
        """
        # Get distribution instance
        self.distribution_ = get_distribution(self.distribution)
        
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples = len(y)
        
        # Bin features
        if isinstance(X, BinnedArray):
            self.X_binned_ = X
        else:
            self.X_binned_ = array(X, n_bins=self.n_bins)
        
        self.n_features_in_ = self.X_binned_.n_features
        
        # Initialize tree storage
        self.trees_ = {}
        for param_name in self.distribution_.param_names:
            self.trees_[param_name] = []
        
        # Initialize raw predictions (in link space) using data statistics
        init_params = self.distribution_.init_params(y)
        raw_preds = {}
        params = {}
        
        for param_name in self.distribution_.param_names:
            raw_init = init_params[param_name]
            self._base_scores[param_name] = float(raw_init)
            raw_preds[param_name] = np.full(n_samples, raw_init, dtype=np.float32)
            params[param_name] = self.distribution_.link(param_name, raw_preds[param_name])
        
        # Training loop
        for round_idx in range(self.n_trees):
            # Update params from raw predictions (apply link functions)
            for param_name in self.distribution_.param_names:
                params[param_name] = self.distribution_.link(
                    param_name, raw_preds[param_name]
                )
            
            # Get gradients for each parameter (ordinary gradient)
            grads_dict = self._compute_gradients(y, params)
            
            # Train one tree per parameter
            for param_name in self.distribution_.param_names:
                grad, hess = grads_dict[param_name]
                
                tree = fit_tree(
                    self.X_binned_,
                    grad,
                    hess,
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                )
                
                self.trees_[param_name].append(tree)
                
                # Update raw predictions (add tree prediction, as in standard GBDT)
                # Tree is trained on gradients, so it outputs the negative gradient direction
                tree_pred = tree(self.X_binned_)
                if hasattr(tree_pred, 'copy_to_host'):
                    tree_pred = tree_pred.copy_to_host()
                
                raw_preds[param_name] += self.learning_rate * tree_pred
        
        return self
    
    def _compute_gradients(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, tuple[NDArray, NDArray]]:
        """Compute gradients (ordinary gradient descent).
        
        Subclasses can override for different gradient computation.
        """
        return self.distribution_.nll_gradient(y, params)
    
    def _predict_raw(self, X: NDArray | BinnedArray) -> dict[str, NDArray]:
        """Predict raw (link-space) parameters.
        
        Args:
            X: Features to predict on
            
        Returns:
            Dictionary mapping param_name -> raw predictions
        """
        if not self.trees_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Bin the data if needed, using training bin edges for consistency
        if isinstance(X, BinnedArray):
            X_binned = X
        elif self.X_binned_ is not None:
            # Use transform to apply training bin edges to new data
            X_binned = self.X_binned_.transform(X)
        else:
            X_binned = array(X, n_bins=self.n_bins)
        
        n_samples = X_binned.n_samples
        raw_preds = {}
        
        for param_name in self.distribution_.param_names:
            # Start with base score
            pred = np.full(n_samples, self._base_scores[param_name], dtype=np.float32)
            
            # Accumulate tree predictions
            for tree in self.trees_[param_name]:
                tree_pred = tree(X_binned)
                if hasattr(tree_pred, 'copy_to_host'):
                    tree_pred = tree_pred.copy_to_host()
                pred += self.learning_rate * tree_pred
            
            raw_preds[param_name] = pred
        
        return raw_preds
    
    def predict_params(self, X: NDArray | BinnedArray) -> dict[str, NDArray]:
        """Predict distribution parameters.
        
        Args:
            X: Features to predict on
            
        Returns:
            Dictionary mapping param_name -> predicted values
            (in constrained parameter space)
        """
        raw_preds = self._predict_raw(X)
        
        params = {}
        for param_name in self.distribution_.param_names:
            params[param_name] = self.distribution_.link(
                param_name, raw_preds[param_name]
            )
        
        return params
    
    def predict_distribution(self, X: NDArray | BinnedArray) -> DistributionOutput:
        """Predict full distribution.
        
        Args:
            X: Features to predict on
            
        Returns:
            DistributionOutput with params, mean(), variance(), 
            quantile(), interval(), sample() methods
        """
        params = self.predict_params(X)
        return DistributionOutput(params=params, distribution=self.distribution_)
    
    def predict(self, X: NDArray | BinnedArray) -> NDArray:
        """Predict mean (expected value).
        
        This provides a point prediction for compatibility with standard GBDT.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted mean values
        """
        params = self.predict_params(X)
        return self.distribution_.mean(params)
    
    def predict_interval(
        self,
        X: NDArray | BinnedArray,
        alpha: float = 0.1,
    ) -> tuple[NDArray, NDArray]:
        """Predict (1-alpha) prediction interval.
        
        Args:
            X: Features to predict on
            alpha: Significance level (0.1 = 90% interval)
            
        Returns:
            (lower, upper) bounds
        """
        output = self.predict_distribution(X)
        return output.interval(alpha)
    
    def predict_quantile(self, X: NDArray | BinnedArray, q: float) -> NDArray:
        """Predict q-th quantile.
        
        Args:
            X: Features to predict on
            q: Quantile level (0 < q < 1)
            
        Returns:
            Predicted quantiles
        """
        output = self.predict_distribution(X)
        return output.quantile(q)
    
    def sample(
        self, 
        X: NDArray | BinnedArray, 
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        """Sample from predicted distribution.
        
        Args:
            X: Features, shape (n, n_features)
            n_samples: Number of samples per observation
            seed: Random seed for reproducibility
            
        Returns:
            samples: Shape (n, n_samples)
        """
        output = self.predict_distribution(X)
        return output.sample(n_samples, seed)
    
    def score(self, X: NDArray | BinnedArray, y: NDArray) -> float:
        """Compute negative log-likelihood (lower is better).
        
        Args:
            X: Features
            y: True target values
            
        Returns:
            Mean negative log-likelihood
        """
        output = self.predict_distribution(X)
        nll = output.nll(np.asarray(y, dtype=np.float32))
        return float(np.mean(nll))
    
    def nll(self, X: NDArray | BinnedArray, y: NDArray) -> float:
        """Alias for score() - compute mean NLL."""
        return self.score(X, y)

    def _post_load(self) -> None:
        """Recreate distribution instance after loading from file."""
        if self.distribution_ is None and self.distribution is not None:
            self.distribution_ = get_distribution(self.distribution)


@dataclass
class NaturalBoost(DistributionalGBDT):
    """Natural Gradient Boosting for probabilistic prediction.
    
    OpenBoost's implementation of natural gradient boosting, inspired by NGBoost.
    Uses natural gradient instead of ordinary gradient, leading to faster
    convergence by accounting for the geometry of the parameter space.
    
    Natural gradient: F^{-1} @ ordinary_gradient
    where F is the Fisher information matrix.
    
    Key advantages over standard GBDT:
    - Full probability distributions, not just point estimates
    - Prediction intervals and uncertainty quantification
    - Faster convergence than ordinary gradient descent
    
    Key advantages over official NGBoost:
    - GPU acceleration via histogram-based trees
    - Faster on large datasets (>10k samples)
    - Custom distributions with autodiff support
    
    Reference:
        Duan et al. "NGBoost: Natural Gradient Boosting for Probabilistic
        Prediction." ICML 2020.
    
    Args:
        distribution: Distribution name or instance
        n_trees: Number of boosting rounds (often needs fewer than ordinary)
        max_depth: Maximum depth of each tree (default 4, often smaller is better)
        learning_rate: Shrinkage factor
        min_child_weight: Minimum sum of hessian in a leaf
        reg_lambda: L2 regularization
        n_bins: Number of bins for histogram building
        
    Example:
        ```python
        model = NaturalBoost(distribution='normal', n_trees=500)
        model.fit(X_train, y_train)
        
        # Get prediction intervals
        lower, upper = model.predict_interval(X_test, alpha=0.1)
        
        # Get full distribution
        output = model.predict_distribution(X_test)
        samples = output.sample(n_samples=1000)
        ```
    """
    
    # Override defaults for NaturalBoost
    max_depth: int = 4  # Shallower trees often work better
    learning_rate: float = 0.1
    
    def _compute_gradients(
        self,
        y: NDArray,
        params: dict[str, NDArray],
    ) -> dict[str, tuple[NDArray, NDArray]]:
        """Compute natural gradients.
        
        Natural gradient = F^{-1} @ ordinary_gradient
        where F is the Fisher information matrix.
        """
        return self.distribution_.natural_gradient(y, params)


# =============================================================================
# Convenience aliases
# =============================================================================

# NaturalBoost with specific distributions
def NaturalBoostNormal(**kwargs) -> NaturalBoost:
    """NaturalBoost with Normal distribution."""
    return NaturalBoost(distribution='normal', **kwargs)


def NaturalBoostLogNormal(**kwargs) -> NaturalBoost:
    """NaturalBoost with LogNormal distribution (for positive data)."""
    return NaturalBoost(distribution='lognormal', **kwargs)


def NaturalBoostGamma(**kwargs) -> NaturalBoost:
    """NaturalBoost with Gamma distribution (for positive data)."""
    return NaturalBoost(distribution='gamma', **kwargs)


def NaturalBoostPoisson(**kwargs) -> NaturalBoost:
    """NaturalBoost with Poisson distribution (for count data)."""
    return NaturalBoost(distribution='poisson', **kwargs)


def NaturalBoostStudentT(**kwargs) -> NaturalBoost:
    """NaturalBoost with Student-t distribution (for heavy-tailed data)."""
    return NaturalBoost(distribution='studentt', **kwargs)


# =============================================================================
# Kaggle Competition Favorites
# =============================================================================

def NaturalBoostTweedie(power: float = 1.5, **kwargs) -> NaturalBoost:
    """NaturalBoost with Tweedie distribution (for insurance claims, zero-inflated data).
    
    **Kaggle Use Cases**:
    - Porto Seguro Safe Driver Prediction
    - Allstate Claims Severity
    - Any zero-inflated positive target
    
    Args:
        power: Tweedie power parameter (1 < power < 2).
               1.5 is the default for insurance claims.
        **kwargs: Other NaturalBoost parameters (n_trees, learning_rate, etc.)
        
    Example:
        ```python
        model = NaturalBoostTweedie(power=1.5, n_trees=500)
        model.fit(X_train, y_train)  # y has zeros and positive values
        
        # Get prediction intervals (XGBoost can't do this!)
        lower, upper = model.predict_interval(X_test, alpha=0.1)
        ```
    """
    from .._distributions import Tweedie
    return NaturalBoost(distribution=Tweedie(power=power), **kwargs)


def NaturalBoostNegBin(**kwargs) -> NaturalBoost:
    """NaturalBoost with Negative Binomial distribution (for overdispersed count data).
    
    **Kaggle Use Cases**:
    - Rossmann Store Sales
    - Bike Sharing Demand
    - Grupo Bimbo Inventory Demand
    - Any count prediction where variance > mean
    
    Args:
        **kwargs: NaturalBoost parameters (n_trees, learning_rate, etc.)
        
    Example:
        ```python
        model = NaturalBoostNegBin(n_trees=500)
        model.fit(X_train, y_train)  # y is count data
        
        # Probability of exceeding threshold (demand planning!)
        output = model.predict_distribution(X_test)
        prob_high_demand = output.distribution.prob_exceed(output.params, 100)
        ```
    """
    return NaturalBoost(distribution='negativebinomial', **kwargs)


# =============================================================================
# Backward compatibility aliases (deprecated)
# =============================================================================

# Keep old names working but mark as deprecated
NGBoost = NaturalBoost  # Alias for backward compatibility
NGBoostNormal = NaturalBoostNormal
NGBoostLogNormal = NaturalBoostLogNormal
NGBoostGamma = NaturalBoostGamma
NGBoostPoisson = NaturalBoostPoisson
NGBoostStudentT = NaturalBoostStudentT
NGBoostTweedie = NaturalBoostTweedie
NGBoostNegBin = NaturalBoostNegBin
