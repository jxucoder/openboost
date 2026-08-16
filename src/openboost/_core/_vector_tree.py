"""CPU tree fitting for shared-structure, vector-valued boosting trees.

This module is intentionally separate from the scalar tree builder.  A vector
tree chooses one split structure for all outputs, sums the per-output Newton
gain when comparing splits, and stores one vector in every leaf.  The first
consumer is :class:`openboost.HistogramBoost`.

The implementation is CPU-only.  Keeping that boundary explicit avoids
silently routing vector gradients through CUDA kernels whose histogram and
prediction layouts are scalar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._array import MISSING_BIN, BinnedArray
from ._growth import TreeStructure, VectorLeaves


@dataclass(frozen=True)
class VectorSplit:
    """Best split found for one vector-gradient node."""

    feature: int = -1
    threshold: int = -1
    gain: float = 0.0
    missing_go_left: bool = True


def _soft_threshold(values: NDArray, reg_alpha: float) -> NDArray:
    if reg_alpha <= 0.0:
        return values
    return np.sign(values) * np.maximum(np.abs(values) - reg_alpha, 0.0)


def _node_score(
    sum_grad: NDArray,
    sum_hess: NDArray,
    reg_lambda: float,
    reg_alpha: float,
) -> NDArray:
    """Per-candidate vector Newton score, averaged across outputs."""
    shrunk = _soft_threshold(sum_grad, reg_alpha)
    return np.mean(shrunk * shrunk / (sum_hess + reg_lambda), axis=-1)


def _leaf_value(
    sum_grad: NDArray,
    sum_hess: NDArray,
    reg_lambda: float,
    reg_alpha: float,
) -> NDArray:
    shrunk = _soft_threshold(sum_grad, reg_alpha)
    return -shrunk / (sum_hess + reg_lambda)


def _feature_histogram(
    bins: NDArray,
    grad: NDArray,
    hess: NDArray,
) -> tuple[NDArray, NDArray]:
    """Aggregate ``(n_samples, n_outputs)`` statistics into 256 bins."""
    n_outputs = grad.shape[1]
    hist_grad = np.zeros((256, n_outputs), dtype=np.float64)
    hist_hess = np.zeros((256, n_outputs), dtype=np.float64)
    np.add.at(hist_grad, bins, grad)
    np.add.at(hist_hess, bins, hess)
    return hist_grad, hist_hess


def _find_best_vector_split(
    binned: NDArray,
    grad: NDArray,
    hess: NDArray,
    *,
    min_child_weight: float,
    reg_lambda: float,
    reg_alpha: float,
    min_gain: float,
) -> VectorSplit:
    """Find the best numeric split, including both missing-value directions."""
    total_grad = np.sum(grad, axis=0, dtype=np.float64)
    total_hess = np.sum(hess, axis=0, dtype=np.float64)
    parent_score = float(_node_score(total_grad, total_hess, reg_lambda, reg_alpha))
    best = VectorSplit()

    for feature in range(binned.shape[0]):
        hist_grad, hist_hess = _feature_histogram(binned[feature], grad, hess)
        missing_grad = hist_grad[MISSING_BIN]
        missing_hess = hist_hess[MISSING_BIN]
        nonmissing_grad = total_grad - missing_grad
        nonmissing_hess = total_hess - missing_hess

        # Threshold 254 cannot leave a non-missing sample on the right, so the
        # useful numeric thresholds are 0..253.  Empty sides are rejected by
        # the child-weight check below.
        left_grad = np.cumsum(hist_grad[:MISSING_BIN], axis=0)[:-1]
        left_hess = np.cumsum(hist_hess[:MISSING_BIN], axis=0)[:-1]
        right_grad = nonmissing_grad - left_grad
        right_hess = nonmissing_hess - left_hess

        for missing_go_left in (True, False):
            candidate_left_grad = left_grad + (missing_grad if missing_go_left else 0.0)
            candidate_left_hess = left_hess + (missing_hess if missing_go_left else 0.0)
            candidate_right_grad = right_grad + (0.0 if missing_go_left else missing_grad)
            candidate_right_hess = right_hess + (0.0 if missing_go_left else missing_hess)

            # Mean curvature keeps min_child_weight invariant to n_outputs.
            valid = (np.mean(candidate_left_hess, axis=1) >= min_child_weight) & (
                np.mean(candidate_right_hess, axis=1) >= min_child_weight
            )
            if not np.any(valid):
                continue

            gains = (
                _node_score(
                    candidate_left_grad,
                    candidate_left_hess,
                    reg_lambda,
                    reg_alpha,
                )
                + _node_score(
                    candidate_right_grad,
                    candidate_right_hess,
                    reg_lambda,
                    reg_alpha,
                )
                - parent_score
            )
            gains = np.where(valid, gains, -np.inf)
            threshold = int(np.argmax(gains))
            gain = float(gains[threshold])
            if gain > best.gain and gain > min_gain:
                best = VectorSplit(feature, threshold, gain, missing_go_left)

    return best


def fit_vector_tree(
    X: BinnedArray | NDArray,
    grad: NDArray,
    hess: NDArray,
    *,
    max_depth: int = 6,
    min_child_weight: float = 1e-3,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_gain: float = 0.0,
) -> TreeStructure:
    """Fit one CPU, level-wise tree with shared structure and vector leaves.

    Parameters
    ----------
    X:
        CPU ``BinnedArray`` or feature-major uint8 matrix.
    grad, hess:
        Arrays with shape ``(n_samples, n_outputs)``.
    """
    if isinstance(X, BinnedArray):
        if X.device != "cpu" or hasattr(X.data, "__cuda_array_interface__"):
            raise NotImplementedError("fit_vector_tree currently supports CPU data only")
        if X.any_categorical:
            raise NotImplementedError(
                "fit_vector_tree does not yet support categorical feature splits"
            )
        binned = np.asarray(X.data, dtype=np.uint8)
        n_features = X.n_features
        n_samples = X.n_samples
    else:
        binned = np.asarray(X, dtype=np.uint8)
        if binned.ndim != 2:
            raise ValueError("binned X must have shape (n_features, n_samples)")
        n_features, n_samples = binned.shape

    grad = np.asarray(grad, dtype=np.float64)
    hess = np.asarray(hess, dtype=np.float64)
    if grad.ndim != 2 or hess.shape != grad.shape:
        raise ValueError("grad and hess must have matching (n_samples, n_outputs) shapes")
    if grad.shape[0] != n_samples:
        raise ValueError(f"grad has {grad.shape[0]} samples, expected {n_samples}")
    if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(hess)):
        raise ValueError("grad and hess must contain only finite values")
    if np.any(hess < 0.0):
        raise ValueError("hess must be non-negative for vector Newton trees")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if reg_lambda <= 0.0:
        raise ValueError("reg_lambda must be strictly positive")

    n_outputs = grad.shape[1]
    max_nodes = 2 ** (max_depth + 1) - 1
    features = np.full(max_nodes, -1, dtype=np.int32)
    thresholds = np.zeros(max_nodes, dtype=np.uint8)
    left_children = np.full(max_nodes, -1, dtype=np.int32)
    right_children = np.full(max_nodes, -1, dtype=np.int32)
    missing_go_left = np.ones(max_nodes, dtype=np.bool_)
    values = np.zeros((max_nodes, n_outputs), dtype=np.float32)

    node_samples: dict[int, NDArray] = {0: np.arange(n_samples, dtype=np.int32)}
    leaves: dict[int, NDArray] = {}
    active = [0]
    deepest = 0

    for depth in range(max_depth):
        next_active: list[int] = []
        for node_id in active:
            sample_idx = node_samples[node_id]
            split = _find_best_vector_split(
                binned[:, sample_idx],
                grad[sample_idx],
                hess[sample_idx],
                min_child_weight=min_child_weight,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                min_gain=min_gain,
            )
            if split.feature < 0:
                leaves[node_id] = sample_idx
                continue

            bins = binned[split.feature, sample_idx]
            goes_left = bins <= split.threshold
            if split.missing_go_left:
                goes_left |= bins == MISSING_BIN
            else:
                goes_left &= bins != MISSING_BIN

            left_idx = sample_idx[goes_left]
            right_idx = sample_idx[~goes_left]
            if left_idx.size == 0 or right_idx.size == 0:
                leaves[node_id] = sample_idx
                continue

            left_id = 2 * node_id + 1
            right_id = left_id + 1
            features[node_id] = split.feature
            thresholds[node_id] = split.threshold
            left_children[node_id] = left_id
            right_children[node_id] = right_id
            missing_go_left[node_id] = split.missing_go_left
            node_samples[left_id] = left_idx
            node_samples[right_id] = right_idx
            next_active.extend((left_id, right_id))
            deepest = max(deepest, depth + 1)
        active = next_active
        if not active:
            break

    for node_id in active:
        leaves[node_id] = node_samples[node_id]

    for node_id, sample_idx in leaves.items():
        sum_grad = np.sum(grad[sample_idx], axis=0, dtype=np.float64)
        sum_hess = np.sum(hess[sample_idx], axis=0, dtype=np.float64)
        values[node_id] = _leaf_value(
            sum_grad,
            sum_hess,
            reg_lambda,
            reg_alpha,
        ).astype(np.float32)

    n_nodes = max([0, *node_samples]) + 1
    return TreeStructure(
        features=features[:n_nodes],
        thresholds=thresholds[:n_nodes],
        left_children=left_children[:n_nodes],
        right_children=right_children[:n_nodes],
        values=VectorLeaves(values[:n_nodes], n_outputs=n_outputs),
        n_nodes=n_nodes,
        depth=deepest,
        n_features=n_features,
        missing_go_left=missing_go_left[:n_nodes],
    )
