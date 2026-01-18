"""Feature importance utilities for OpenBoost.

Phase 13: Compute feature importances from any tree-based model.

This module provides functions to compute feature importances that work
with any model containing trees: GradientBoosting, DART, MultiClass,
OpenBoostGAM, etc.

Example:
    >>> import openboost as ob
    >>> from openboost import compute_feature_importances
    >>> 
    >>> model = ob.GradientBoosting(n_trees=100).fit(X, y)
    >>> importances = compute_feature_importances(model)
    >>> 
    >>> # Top features
    >>> top_features = np.argsort(importances)[::-1][:10]
    >>> for i in top_features:
    ...     print(f"Feature {i}: {importances[i]:.4f}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class HasTrees(Protocol):
    """Protocol for models with trees (duck typing).
    
    Any model with a `trees_` attribute containing TreeStructure objects
    can use the importance functions.
    """
    trees_: list


def compute_feature_importances(
    model: Any,
    importance_type: str = 'frequency',
    normalize: bool = True,
) -> NDArray:
    """Compute feature importances from any tree-based model.
    
    Works with: GradientBoosting, DART, OpenBoostGAM, MultiClassGradientBoosting,
    and any model with a `trees_` attribute.
    
    Args:
        model: A fitted model with `trees_` attribute.
        importance_type: Type of importance calculation:
            - 'frequency': Number of times feature is used for splits (default)
            - 'gain': Sum of gain from splits on each feature (if available)
            - 'cover': Sum of samples covered by splits (if available)
        normalize: If True, normalize importances to sum to 1.
        
    Returns:
        importances: Array of shape (n_features,) with importance scores.
        
    Raises:
        ValueError: If model has no trees or unknown importance_type.
        
    Example:
        >>> model = ob.GradientBoosting(n_trees=100).fit(X, y)
        >>> importances = compute_feature_importances(model)
        >>> 
        >>> # Use with sklearn-style attribute
        >>> model.feature_importances_ = compute_feature_importances(model)
    """
    if not hasattr(model, 'trees_'):
        raise ValueError("Model must have 'trees_' attribute")
    
    trees = _get_trees_flat(model)
    
    if not trees:
        raise ValueError("Model has no fitted trees")
    
    # Get number of features
    n_features = _get_n_features(model, trees)
    
    # Compute importances
    importances = np.zeros(n_features, dtype=np.float64)
    
    for tree in trees:
        _accumulate_importance(tree, importances, importance_type)
    
    # Normalize if requested
    if normalize:
        total = importances.sum()
        if total > 0:
            importances /= total
    
    return importances.astype(np.float32)


def _get_trees_flat(model: Any) -> list:
    """Extract a flat list of trees from various model types.
    
    Handles:
    - GradientBoosting, DART: trees_ is list of TreeStructure
    - MultiClassGradientBoosting: trees_ is list of lists (one per class per round)
    - OpenBoostGAM: trees_ is dict of feature -> list of trees
    """
    trees = model.trees_
    
    if not trees:
        return []
    
    # Check if it's a list of lists (MultiClass)
    if isinstance(trees, list) and trees and isinstance(trees[0], list):
        return [t for round_trees in trees for t in round_trees]
    
    # Check if it's a dict (GAM)
    if isinstance(trees, dict):
        return [t for feature_trees in trees.values() for t in feature_trees]
    
    # Regular flat list
    return list(trees)


def _get_n_features(model: Any, trees: list) -> int:
    """Get number of features from model or trees."""
    # Try model attributes first
    if hasattr(model, 'n_features_in_'):
        return model.n_features_in_
    
    if hasattr(model, 'X_binned_') and model.X_binned_ is not None:
        return model.X_binned_.n_features
    
    # Infer from trees
    if trees:
        return trees[0].n_features
    
    raise ValueError("Cannot determine number of features")


def _accumulate_importance(tree, importances: NDArray, importance_type: str) -> None:
    """Accumulate importance scores from a single tree.
    
    Args:
        tree: TreeStructure object
        importances: Array to accumulate into (modified in place)
        importance_type: 'frequency', 'gain', or 'cover'
    """
    n_nodes = tree.n_nodes
    features = tree.features
    left_children = tree.left_children
    
    for node_idx in range(n_nodes):
        # Check if this is a split node (not a leaf)
        if left_children[node_idx] != -1:
            feature = features[node_idx]
            
            if feature < 0 or feature >= len(importances):
                continue
            
            if importance_type == 'frequency':
                # Count splits
                importances[feature] += 1.0
            
            elif importance_type == 'gain':
                # Use gain if stored, otherwise fall back to frequency
                if hasattr(tree, 'split_gains') and tree.split_gains is not None:
                    importances[feature] += tree.split_gains[node_idx]
                else:
                    # Fallback: use 1.0 (equivalent to frequency)
                    importances[feature] += 1.0
            
            elif importance_type == 'cover':
                # Use sample counts if stored, otherwise fall back to frequency
                if hasattr(tree, 'node_counts') and tree.node_counts is not None:
                    importances[feature] += tree.node_counts[node_idx]
                else:
                    # Fallback: use 1.0
                    importances[feature] += 1.0
            
            else:
                raise ValueError(f"Unknown importance_type: '{importance_type}'. "
                               "Use 'frequency', 'gain', or 'cover'.")


def get_feature_importance_dict(
    model: Any,
    feature_names: list[str] | None = None,
    importance_type: str = 'frequency',
    top_n: int | None = None,
) -> dict[str, float]:
    """Get feature importances as a sorted dictionary.
    
    Convenience function that returns importances as a dict, optionally
    with feature names and limited to top N features.
    
    Args:
        model: Fitted tree-based model.
        feature_names: Optional list of feature names.
        importance_type: Type of importance ('frequency', 'gain', 'cover').
        top_n: If provided, return only top N features.
        
    Returns:
        Dict mapping feature name/index to importance, sorted by importance.
        
    Example:
        >>> importance_dict = get_feature_importance_dict(
        ...     model, 
        ...     feature_names=['age', 'income', 'score'],
        ...     top_n=2
        ... )
        >>> # {'income': 0.45, 'age': 0.32}
    """
    importances = compute_feature_importances(model, importance_type, normalize=True)
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Create dict and sort by importance
    importance_dict = dict(zip(feature_names, importances))
    sorted_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Limit to top N if requested
    if top_n is not None:
        sorted_dict = dict(list(sorted_dict.items())[:top_n])
    
    return sorted_dict


def plot_feature_importances(
    model: Any,
    feature_names: list[str] | None = None,
    importance_type: str = 'frequency',
    top_n: int = 20,
    ax=None,
    **kwargs,
):
    """Plot feature importances as a horizontal bar chart.
    
    Args:
        model: Fitted tree-based model.
        feature_names: Optional list of feature names.
        importance_type: Type of importance ('frequency', 'gain', 'cover').
        top_n: Number of top features to show.
        ax: Matplotlib axes to plot on (creates new if None).
        **kwargs: Additional arguments passed to barh().
        
    Returns:
        Matplotlib axes object.
        
    Example:
        >>> plot_feature_importances(model, top_n=10)
        >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    importances = compute_feature_importances(model, importance_type, normalize=True)
    
    # Get indices of top features
    top_indices = np.argsort(importances)[::-1][:top_n]
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Get names and values for top features
    top_names = [feature_names[i] for i in top_indices]
    top_values = importances[top_indices]
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_values, **kwargs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel(f'Importance ({importance_type})')
    ax.set_title('Feature Importances')
    
    return ax
