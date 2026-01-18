"""sklearn-compatible wrappers for OpenBoost models.

Phase 13: Thin adapters that provide sklearn compatibility
(GridSearchCV, cross_val_score, Pipeline, etc.).

These wrappers delegate to the core OpenBoost models while providing:
- sklearn BaseEstimator interface (get_params, set_params)
- RegressorMixin / ClassifierMixin (score method)
- Proper input validation (check_X_y, check_array)
- Feature importance as a property

Example:
    >>> from openboost import OpenBoostRegressor, OpenBoostClassifier
    >>> from sklearn.model_selection import GridSearchCV
    >>> 
    >>> # Regression
    >>> reg = OpenBoostRegressor(n_estimators=100, max_depth=6)
    >>> reg.fit(X_train, y_train)
    >>> reg.score(X_test, y_test)  # R² score
    >>> 
    >>> # Classification
    >>> clf = OpenBoostClassifier(n_estimators=100)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict_proba(X_test)
    >>> clf.classes_
    >>> 
    >>> # GridSearchCV
    >>> param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7]}
    >>> search = GridSearchCV(OpenBoostRegressor(), param_grid, cv=5)
    >>> search.fit(X, y)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        loss: str = 'squared_error',
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        n_bins: int = 256,
        quantile_alpha: float = 0.5,
        early_stopping_rounds: int | None = None,
        verbose: int = 0,
        random_state: int | None = None,
    ):
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
        early_stopping_rounds: int | None = None,
        verbose: int = 0,
        random_state: int | None = None,
    ):
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
