"""sklearn-compatible wrappers for OpenBoost models.

Phase 13: Thin adapters that provide sklearn compatibility
(GridSearchCV, cross_val_score, Pipeline, etc.).

These wrappers delegate to the core OpenBoost models while providing:
- sklearn BaseEstimator interface (get_params, set_params)
- RegressorMixin / ClassifierMixin (score method)
- Proper input validation (check_X_y, check_array)
- Feature importance as a property

Example:
    ```python
    from openboost import OpenBoostRegressor, OpenBoostClassifier
    from sklearn.model_selection import GridSearchCV
    
    # Regression
    reg = OpenBoostRegressor(n_estimators=100, max_depth=6)
    reg.fit(X_train, y_train)
    reg.score(X_test, y_test)  # R² score
    
    # Classification
    clf = OpenBoostClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    clf.predict_proba(X_test)
    ```
    >>> clf.classes_
    >>> 
    >>> # GridSearchCV
    >>> param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7]}
    >>> search = GridSearchCV(OpenBoostRegressor(), param_grid, cv=5)
    >>> search.fit(X, y)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

try:
    from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Provide stubs if sklearn not available
    class BaseEstimator:
        pass
    class RegressorMixin:
        pass
    class ClassifierMixin:
        pass

from ._boosting import GradientBoosting, MultiClassGradientBoosting
from .._callbacks import EarlyStopping, Logger, Callback
from .._importance import compute_feature_importances

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _check_sklearn():
    """Raise error if sklearn not available."""
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "sklearn is required for OpenBoostRegressor/OpenBoostClassifier. "
            "Install with: pip install scikit-learn"
        )


class OpenBoostRegressor(BaseEstimator, RegressorMixin):
    """Gradient Boosting Regressor with sklearn-compatible interface.
    
    This is a thin wrapper around OpenBoost's GradientBoosting that provides
    full compatibility with sklearn's ecosystem (GridSearchCV, Pipeline, etc.).
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds (trees).
    max_depth : int, default=6
        Maximum depth of each tree.
    learning_rate : float, default=0.1
        Shrinkage factor applied to each tree's contribution.
    loss : {'squared_error', 'absolute_error', 'huber', 'quantile'}, default='squared_error'
        Loss function to optimize.
    min_child_weight : float, default=1.0
        Minimum sum of hessian in a leaf node.
    reg_lambda : float, default=1.0
        L2 regularization on leaf values.
    reg_alpha : float, default=0.0
        L1 regularization on leaf values.
    gamma : float, default=0.0
        Minimum gain required to make a split.
    subsample : float, default=1.0
        Fraction of samples to use for each tree.
    colsample_bytree : float, default=1.0
        Fraction of features to use for each tree.
    n_bins : int, default=256
        Number of bins for histogram building.
    quantile_alpha : float, default=0.5
        Quantile level for 'quantile' loss.
    subsample_strategy : {'none', 'random', 'goss'}, default='none'
        Sampling strategy for large-scale training (Phase 17).
        - 'none': No sampling (default)
        - 'random': Random subsampling
        - 'goss': Gradient-based One-Side Sampling (LightGBM-style)
    goss_top_rate : float, default=0.2
        Fraction of top-gradient samples to keep (for GOSS).
    goss_other_rate : float, default=0.1
        Fraction of remaining samples to sample (for GOSS).
    batch_size : int, optional
        Mini-batch size for large datasets. If None, process all at once.
    early_stopping_rounds : int, optional
        Stop training if validation score doesn't improve for this many rounds.
        Requires eval_set to be passed to fit().
    verbose : int, default=0
        Verbosity level (0=silent, N=log every N rounds).
    random_state : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit (if X is a DataFrame).
    feature_importances_ : ndarray of shape (n_features_in_,)
        Feature importances (based on split frequency).
    booster_ : GradientBoosting
        The underlying fitted OpenBoost model.
    best_iteration_ : int
        Iteration with best validation score (if early stopping used).
    best_score_ : float
        Best validation score achieved (if early stopping used).
        
    Examples
    --------
    >>> from openboost import OpenBoostRegressor
    >>> reg = OpenBoostRegressor(n_estimators=100, max_depth=6)
    >>> reg.fit(X_train, y_train)
    >>> reg.predict(X_test)
    >>> reg.score(X_test, y_test)  # R² score
    
    >>> # With early stopping
    >>> reg = OpenBoostRegressor(n_estimators=1000, early_stopping_rounds=50)
    >>> reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    >>> print(f"Best iteration: {reg.best_iteration_}")
    
    >>> # GridSearchCV
    >>> from sklearn.model_selection import GridSearchCV
    >>> param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
    >>> search = GridSearchCV(OpenBoostRegressor(), param_grid, cv=5)
    >>> search.fit(X, y)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        loss: Literal['squared_error', 'absolute_error', 'huber', 'quantile'] = 'squared_error',
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        n_bins: int = 256,
        quantile_alpha: float = 0.5,
        subsample_strategy: Literal['none', 'random', 'goss'] = 'none',
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        batch_size: int | None = None,
        early_stopping_rounds: int | None = None,
        verbose: int = 0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.loss = loss
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_bins = n_bins
        self.quantile_alpha = quantile_alpha
        self.subsample_strategy = subsample_strategy
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state
    
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        sample_weight: NDArray | None = None,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
    ) -> "OpenBoostRegressor":
        """Fit the gradient boosting regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping.
            
        Returns
        -------
        self : OpenBoostRegressor
            Fitted estimator.
        """
        _check_sklearn()
        
        # Validate input
        X, y = check_X_y(X, y, dtype=np.float32, y_numeric=True)
        
        # Store sklearn attributes
        self.n_features_in_ = X.shape[1]
        
        # Map sklearn loss names to OpenBoost names
        loss_map = {
            'squared_error': 'mse',
            'absolute_error': 'mae',
            'huber': 'huber',
            'quantile': 'quantile',
        }
        internal_loss = loss_map.get(self.loss, self.loss)
        
        # Build callback list
        callbacks = []
        if self.early_stopping_rounds is not None and eval_set is not None:
            callbacks.append(EarlyStopping(
                patience=self.early_stopping_rounds,
                restore_best=True,
                verbose=self.verbose > 0,
            ))
        
        # Create core model
        self.booster_ = GradientBoosting(
            n_trees=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            loss=internal_loss,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_bins=self.n_bins,
            quantile_alpha=self.quantile_alpha,
            # Phase 17: Large-scale training
            subsample_strategy=self.subsample_strategy,
            goss_top_rate=self.goss_top_rate,
            goss_other_rate=self.goss_other_rate,
            batch_size=self.batch_size,
        )
        
        # Fit with callbacks
        self.booster_.fit(
            X, y,
            callbacks=callbacks if callbacks else None,
            eval_set=eval_set,
            sample_weight=sample_weight,
        )
        
        # Copy early stopping attributes
        if hasattr(self.booster_, 'best_iteration_'):
            self.best_iteration_ = self.booster_.best_iteration_
        if hasattr(self.booster_, 'best_score_'):
            self.best_score_ = self.booster_.best_score_
        
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict target values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        _check_sklearn()
        check_is_fitted(self, 'booster_')
        X = check_array(X, dtype=np.float32)
        
        return self.booster_.predict(X)
    
    @property
    def feature_importances_(self) -> NDArray:
        """Feature importances based on split frequency."""
        check_is_fitted(self, 'booster_')
        return compute_feature_importances(self.booster_, importance_type='frequency')
    
    # score() is inherited from RegressorMixin (R² score)


class OpenBoostClassifier(BaseEstimator, ClassifierMixin):
    """Gradient Boosting Classifier with sklearn-compatible interface.
    
    Automatically handles binary and multi-class classification.
    Uses logloss for binary, softmax for multi-class.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=6
        Maximum depth of each tree.
    learning_rate : float, default=0.1
        Shrinkage factor.
    min_child_weight : float, default=1.0
        Minimum sum of hessian in a leaf.
    reg_lambda : float, default=1.0
        L2 regularization on leaf values.
    reg_alpha : float, default=0.0
        L1 regularization on leaf values.
    gamma : float, default=0.0
        Minimum gain required to make a split.
    subsample : float, default=1.0
        Fraction of samples per tree.
    colsample_bytree : float, default=1.0
        Fraction of features per tree.
    n_bins : int, default=256
        Number of bins for histogram building.
    subsample_strategy : {'none', 'random', 'goss'}, default='none'
        Sampling strategy for large-scale training (Phase 17).
    goss_top_rate : float, default=0.2
        Fraction of top-gradient samples to keep (for GOSS).
    goss_other_rate : float, default=0.1
        Fraction of remaining samples to sample (for GOSS).
    batch_size : int, optional
        Mini-batch size for large datasets.
    early_stopping_rounds : int, optional
        Stop if validation doesn't improve.
    verbose : int, default=0
        Verbosity level.
    random_state : int, optional
        Random seed.
        
    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features.
    feature_importances_ : ndarray
        Feature importances.
    booster_ : GradientBoosting or MultiClassGradientBoosting
        Underlying model.
        
    Examples
    --------
    >>> from openboost import OpenBoostClassifier
    >>> clf = OpenBoostClassifier(n_estimators=100)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict(X_test)
    >>> clf.predict_proba(X_test)
    >>> clf.classes_
    array([0, 1])
    
    >>> # Multi-class
    >>> clf.fit(X_train, y_train)  # y_train has 3+ classes
    >>> clf.predict_proba(X_test).shape
    (n_samples, n_classes)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        n_bins: int = 256,
        subsample_strategy: Literal['none', 'random', 'goss'] = 'none',
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        batch_size: int | None = None,
        early_stopping_rounds: int | None = None,
        verbose: int = 0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_bins = n_bins
        self.subsample_strategy = subsample_strategy
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state
    
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        sample_weight: NDArray | None = None,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
    ) -> "OpenBoostClassifier":
        """Fit the gradient boosting classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping.
            
        Returns
        -------
        self : OpenBoostClassifier
            Fitted estimator.
        """
        _check_sklearn()
        
        # Validate input
        X, y = check_X_y(X, y, dtype=np.float32)
        
        # Store sklearn attributes
        self.n_features_in_ = X.shape[1]
        
        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        
        # Transform eval_set labels if provided
        if eval_set is not None:
            eval_set_encoded = []
            for X_val, y_val in eval_set:
                y_val_encoded = self._label_encoder.transform(y_val)
                eval_set_encoded.append((X_val, y_val_encoded))
            eval_set = eval_set_encoded
        
        # Build callbacks
        callbacks = []
        if self.early_stopping_rounds is not None and eval_set is not None:
            callbacks.append(EarlyStopping(
                patience=self.early_stopping_rounds,
                restore_best=True,
                verbose=self.verbose > 0,
            ))
        
        # Choose model based on number of classes
        if self.n_classes_ == 2:
            # Binary classification
            self.booster_ = GradientBoosting(
                n_trees=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                loss='logloss',
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                gamma=self.gamma,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                n_bins=self.n_bins,
                # Phase 17: Large-scale training
                subsample_strategy=self.subsample_strategy,
                goss_top_rate=self.goss_top_rate,
                goss_other_rate=self.goss_other_rate,
                batch_size=self.batch_size,
            )
            self.booster_.fit(
                X, y_encoded.astype(np.float32),
                callbacks=callbacks if callbacks else None,
                eval_set=eval_set,
                sample_weight=sample_weight,
            )
        else:
            # Multi-class classification
            self.booster_ = MultiClassGradientBoosting(
                n_classes=self.n_classes_,
                n_trees=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                gamma=self.gamma,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                n_bins=self.n_bins,
                # Phase 17: Large-scale training
                subsample_strategy=self.subsample_strategy,
                goss_top_rate=self.goss_top_rate,
                goss_other_rate=self.goss_other_rate,
            )
            # Note: MultiClass doesn't support callbacks yet
            self.booster_.fit(X, y_encoded)
        
        # Copy early stopping attributes
        if hasattr(self.booster_, 'best_iteration_'):
            self.best_iteration_ = self.booster_.best_iteration_
        if hasattr(self.booster_, 'best_score_'):
            self.best_score_ = self.booster_.best_score_
        
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        _check_sklearn()
        check_is_fitted(self, 'booster_')
        X = check_array(X, dtype=np.float32)
        
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
    
    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        _check_sklearn()
        check_is_fitted(self, 'booster_')
        X = check_array(X, dtype=np.float32)
        
        return self.booster_.predict_proba(X)
    
    @property
    def feature_importances_(self) -> NDArray:
        """Feature importances based on split frequency."""
        check_is_fitted(self, 'booster_')
        return compute_feature_importances(self.booster_, importance_type='frequency')
    
    # score() is inherited from ClassifierMixin (accuracy)


# =============================================================================
# Phase 15: Distributional Regressor (NGBoost-style)
# =============================================================================

class OpenBoostDistributionalRegressor(BaseEstimator, RegressorMixin):
    """Distributional regression with sklearn-compatible interface.
    
    Predicts full probability distributions instead of point estimates.
    Uses natural gradient boosting (NGBoost) by default for faster convergence.
    
    Parameters
    ----------
    distribution : str, default='normal'
        Distribution family. Options: 'normal', 'lognormal', 'gamma', 
        'poisson', 'studentt'.
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=4
        Maximum depth of each tree. Typically shallower than standard GBDT.
    learning_rate : float, default=0.1
        Shrinkage factor.
    min_child_weight : float, default=1.0
        Minimum sum of hessian in a leaf.
    reg_lambda : float, default=1.0
        L2 regularization on leaf values.
    n_bins : int, default=256
        Number of bins for histogram building.
    use_natural_gradient : bool, default=True
        If True, use NGBoost (natural gradient). Recommended for faster
        convergence and better uncertainty calibration.
    verbose : int, default=0
        Verbosity level.
        
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    booster_ : NGBoost or DistributionalGBDT
        The underlying fitted model.
        
    Examples
    --------
    >>> from openboost import OpenBoostDistributionalRegressor
    >>> model = OpenBoostDistributionalRegressor(distribution='normal')
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Point prediction (mean)
    >>> y_pred = model.predict(X_test)
    >>> 
    >>> # Prediction intervals (90%)
    >>> lower, upper = model.predict_interval(X_test, alpha=0.1)
    >>> 
    >>> # Full distribution parameters
    >>> params = model.predict_distribution(X_test)
    >>> mu, sigma = params['loc'], params['scale']
    >>> 
    >>> # Sample from predicted distribution
    >>> samples = model.sample(X_test, n_samples=100)
    """
    
    def __init__(
        self,
        distribution: Literal['normal', 'lognormal', 'gamma', 'poisson', 'studentt'] = 'normal',
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        n_bins: int = 256,
        use_natural_gradient: bool = True,
        verbose: int = 0,
    ) -> None:
        self.distribution = distribution
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.n_bins = n_bins
        self.use_natural_gradient = use_natural_gradient
        self.verbose = verbose
    
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        **kwargs,
    ) -> "OpenBoostDistributionalRegressor":
        """Fit the distributional regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : OpenBoostDistributionalRegressor
            Fitted estimator.
        """
        _check_sklearn()
        X, y = check_X_y(X, y, dtype=np.float32, y_numeric=True)
        
        self.n_features_in_ = X.shape[1]
        
        # Import here to avoid circular imports
        from ._distributional import NGBoost, DistributionalGBDT
        
        ModelClass = NGBoost if self.use_natural_gradient else DistributionalGBDT
        
        self.booster_ = ModelClass(
            distribution=self.distribution,
            n_trees=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            n_bins=self.n_bins,
        )
        
        self.booster_.fit(X, y)
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict mean (expected value).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted mean values.
        """
        _check_sklearn()
        check_is_fitted(self, 'booster_')
        X = check_array(X, dtype=np.float32)
        
        return self.booster_.predict(X)
    
    def predict_interval(
        self,
        X: NDArray,
        alpha: float = 0.1,
    ) -> tuple[NDArray, NDArray]:
        """Predict (1-alpha) prediction interval.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on.
        alpha : float, default=0.1
            Significance level. 0.1 gives a 90% prediction interval.
            
        Returns
        -------
        lower : ndarray of shape (n_samples,)
            Lower bounds of the interval.
        upper : ndarray of shape (n_samples,)
            Upper bounds of the interval.
        """
        _check_sklearn()
        check_is_fitted(self, 'booster_')
        X = check_array(X, dtype=np.float32)
        
        return self.booster_.predict_interval(X, alpha=alpha)
    
    def predict_distribution(self, X: NDArray) -> dict[str, NDArray]:
        """Predict all distribution parameters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on.
            
        Returns
        -------
        params : dict
            Dictionary mapping parameter names to predicted values.
            For Normal: {'loc': mean, 'scale': std}
        """
        _check_sklearn()
        check_is_fitted(self, 'booster_')
        X = check_array(X, dtype=np.float32)
        
        return self.booster_.predict_params(X)
    
    def predict_quantile(self, X: NDArray, q: float) -> NDArray:
        """Predict q-th quantile.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on.
        q : float
            Quantile level (0 < q < 1).
            
        Returns
        -------
        quantiles : ndarray of shape (n_samples,)
            Predicted quantiles.
        """
        _check_sklearn()
        check_is_fitted(self, 'booster_')
        X = check_array(X, dtype=np.float32)
        
        return self.booster_.predict_quantile(X, q)
    
    def sample(
        self,
        X: NDArray,
        n_samples: int = 1,
        seed: int | None = None,
    ) -> NDArray:
        """Sample from predicted distribution.
        
        Parameters
        ----------
        X : array-like of shape (n_obs, n_features)
            Features to predict on.
        n_samples : int, default=1
            Number of samples per observation.
        seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        samples : ndarray of shape (n_obs, n_samples)
            Samples from the predicted distribution.
        """
        _check_sklearn()
        check_is_fitted(self, 'booster_')
        X = check_array(X, dtype=np.float32)
        
        return self.booster_.sample(X, n_samples, seed)
    
    def nll_score(self, X: NDArray, y: NDArray) -> float:
        """Compute negative log-likelihood (lower is better).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.
        y : array-like of shape (n_samples,)
            True target values.
            
        Returns
        -------
        nll : float
            Mean negative log-likelihood.
        """
        _check_sklearn()
        check_is_fitted(self, 'booster_')
        X = check_array(X, dtype=np.float32)
        
        return self.booster_.nll(X, y)
    
    # score() inherited from RegressorMixin uses R² on mean predictions


# =============================================================================
# Phase 15: Linear Leaf Regressor
# =============================================================================

class OpenBoostLinearLeafRegressor(BaseEstimator, RegressorMixin):
    """Linear Leaf Gradient Boosting with sklearn-compatible interface.
    
    Uses trees with linear models in leaves instead of constant values.
    This provides better extrapolation beyond the training data range.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=4
        Maximum tree depth. Typically shallower than standard GBDT since
        linear models in leaves add flexibility.
    learning_rate : float, default=0.1
        Shrinkage factor.
    loss : str, default='squared_error'
        Loss function: 'squared_error', 'absolute_error', 'huber'.
    min_samples_leaf : int, default=20
        Minimum samples in a leaf to fit linear model.
    reg_lambda : float, default=1.0
        L2 regularization for tree splits.
    reg_lambda_linear : float, default=0.1
        L2 regularization for linear models in leaves (ridge).
    max_features_linear : int, str, or None, default='sqrt'
        Max features for linear model in each leaf:
        - None: Use all features
        - 'sqrt': Use sqrt(n_features)
        - 'log2': Use log2(n_features)
        - int: Use exactly this many features
    n_bins : int, default=256
        Number of bins for histogram building.
    verbose : int, default=0
        Verbosity level.
        
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    booster_ : LinearLeafGBDT
        The underlying fitted model.
        
    Examples
    --------
    >>> from openboost import OpenBoostLinearLeafRegressor
    >>> model = OpenBoostLinearLeafRegressor(n_estimators=100, max_depth=4)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> 
    >>> # Compare with standard GBDT on extrapolation tasks
    >>> # LinearLeafRegressor typically performs better when the
    >>> # underlying relationship has linear components
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        loss: Literal['squared_error', 'absolute_error', 'huber'] = 'squared_error',
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        reg_lambda_linear: float = 0.1,
        max_features_linear: int | Literal['sqrt', 'log2'] | None = 'sqrt',
        n_bins: int = 256,
        verbose: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.loss = loss
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.reg_lambda_linear = reg_lambda_linear
        self.max_features_linear = max_features_linear
        self.n_bins = n_bins
        self.verbose = verbose
    
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        **kwargs,
    ) -> "OpenBoostLinearLeafRegressor":
        """Fit the linear leaf regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : OpenBoostLinearLeafRegressor
            Fitted estimator.
        """
        _check_sklearn()
        X, y = check_X_y(X, y, dtype=np.float32, y_numeric=True)
        
        self.n_features_in_ = X.shape[1]
        
        # Map sklearn loss names to internal names
        loss_map = {
            'squared_error': 'mse',
            'absolute_error': 'mae',
            'huber': 'huber',
        }
        internal_loss = loss_map.get(self.loss, self.loss)
        
        from ._linear_leaf import LinearLeafGBDT
        
        self.booster_ = LinearLeafGBDT(
            n_trees=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            loss=internal_loss,
            min_samples_leaf=self.min_samples_leaf,
            reg_lambda_tree=self.reg_lambda,
            reg_lambda_linear=self.reg_lambda_linear,
            max_features_linear=self.max_features_linear,
            n_bins=self.n_bins,
        )
        
        self.booster_.fit(X, y)
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict target values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        _check_sklearn()
        check_is_fitted(self, 'booster_')
        X = check_array(X, dtype=np.float32)
        
        return self.booster_.predict(X)
    
    # score() inherited from RegressorMixin (R² score)
