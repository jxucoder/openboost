"""Unified multi-channel boosting trainer.

One loop for every model: bin once, maintain raw scores F (n, K), ask the
objective for per-channel (grad, hess), fit one tree per channel, update F
and every eval set incrementally.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._array import BinnedArray, array
from ._backends import is_cuda
from ._callbacks import (
    Callback,
    CallbackManager,
    EarlyStopping,
    TrainingState,
    warn_if_early_stopping_without_eval_set,
)
from ._core._growth import TreeStructure
from ._core._tree import fit_tree, fit_tree_gpu_native
from ._objectives import Objective, RawScores
from ._validation import validate_eval_set, validate_sample_weight


@dataclass
class TrainerConfig:
    """Tree-building knobs shared by every facade."""

    n_trees: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_bins: int = 254
    random_state: int | None = None
    min_gain: float = 0.0


def _to_host(pred: NDArray) -> NDArray:
    if hasattr(pred, "copy_to_host"):
        return pred.copy_to_host()
    if hasattr(pred, "get"):
        return pred.get()
    return np.asarray(pred)


def _is_device(arr) -> bool:
    return hasattr(arr, "__cuda_array_interface__")


def _as_host_raw(raw: RawScores) -> RawScores:
    return {name: _to_host(score) for name, score in raw.items()}


def _gpu_native_eligible(X_binned: BinnedArray, config: TrainerConfig) -> bool:
    if not is_cuda():
        return False
    if config.reg_alpha != 0.0 or config.colsample_bytree < 1.0 or config.subsample < 1.0:
        return False
    has_missing = (
        hasattr(X_binned, "has_missing")
        and len(X_binned.has_missing) > 0
        and np.any(X_binned.has_missing)
    )
    has_categorical = (
        hasattr(X_binned, "is_categorical")
        and len(X_binned.is_categorical) > 0
        and np.any(X_binned.is_categorical)
    )
    return not has_missing and not has_categorical


def _legacy_to_structure(legacy, n_features: int, max_depth: int) -> TreeStructure:
    features, thresholds, values, left, right = legacy.to_arrays()
    return TreeStructure(
        features=features,
        thresholds=thresholds,
        left_children=left,
        right_children=right,
        values=values,
        n_nodes=len(features),
        depth=max_depth,
        n_features=n_features,
    )


def _empty_raw(n: int, base_scores: dict[str, float]) -> RawScores:
    return {
        name: np.full(n, score, dtype=np.float32) for name, score in base_scores.items()
    }


def _bin_features(X, reference: BinnedArray | None, n_bins: int) -> BinnedArray:
    if isinstance(X, BinnedArray):
        return X
    if reference is not None:
        return reference.transform(X)
    return array(X, n_bins=n_bins)


def predict_raw(model: Any, X) -> RawScores:
    """Accumulate raw scores from ``model.trees_`` / ``_base_scores``."""
    if not getattr(model, "trees_", None):
        raise RuntimeError("Model not fitted. Call fit() first.")

    X_binned = _bin_features(X, getattr(model, "X_binned_", None), model.n_bins)
    n = X_binned.n_samples
    raw = _empty_raw(n, model._base_scores)
    lr = model.learning_rate
    for name, trees in model.trees_.items():
        pred = raw[name]
        for tree in trees:
            pred = pred + lr * _to_host(tree(X_binned))
        raw[name] = pred
    return raw


def _restore_on_failure(fit):
    @wraps(fit)
    def wrapped(model, *args, **kwargs):
        # Training assigns new containers; restore trainer-owned state if
        # objective execution or validation fails, including on the first step.
        previous = model.__dict__.copy()
        try:
            return fit(model, *args, **kwargs)
        except Exception:
            model.__dict__.clear()
            model.__dict__.update(previous)
            raise
    return wrapped


@_restore_on_failure
def fit_boosting(
    model: Any,
    objective: Objective,
    X,
    y: NDArray,
    *,
    config: TrainerConfig,
    sample_weight: NDArray | None = None,
    extra: dict[str, Any] | None = None,
    callbacks: list[Callback] | None = None,
    early_stopping_rounds: int | None = None,
    eval_sets: list[dict[str, Any]] | None = None,
    eval_fn: Callable[[NDArray, RawScores, dict[str, Any] | None], float] | None = None,
    eval_metric_name: str = "loss",
    rng: np.random.Generator | None = None,
) -> Any:
    """Fit ``model`` in place. Returns ``model``.

    ``eval_sets`` entries are dicts with keys ``X``, ``y``, optional ``extra``,
    optional ``name`` (defaults to ``eval_0``, ``eval_1``, ...).
    """
    y = np.asarray(y).ravel()
    n_samples = len(y)
    sample_weight = validate_sample_weight(sample_weight, n_samples)
    extra = extra or {}

    use_gpu = is_cuda()
    if not (0 < config.subsample <= 1 and 0 < config.colsample_bytree <= 1):
        raise ValueError("sampling ratios must be in (0, 1]")
    if use_gpu and (config.subsample < 1 or config.colsample_bytree < 1):
        raise ValueError("GPU sampling is not supported by the unified trainer; use CPU or ratios=1")
    rng = rng if rng is not None else np.random.default_rng(config.random_state)
    device_state = (
        use_gpu
        and bool(getattr(objective, "device_capable", False))
        and extra.get("log_offset") is None
    )
    if use_gpu and not device_state:
        reason = "exposure offsets" if extra.get("log_offset") is not None else "objective capability"
        warnings.warn(f"CUDA objective fallback to CPU: {reason}; tree execution may still use CUDA",
                      RuntimeWarning, stacklevel=2)

    model.X_binned_ = _bin_features(X, None, config.n_bins)
    model.n_features_in_ = model.X_binned_.n_features
    model.learning_rate = config.learning_rate
    model.n_bins = config.n_bins

    base = objective.init_raw(y, sample_weight, extra)
    model._base_scores = dict(base)
    model.trees_ = {name: [] for name in objective.channel_names}

    use_native = _gpu_native_eligible(model.X_binned_, config)
    if use_gpu and not use_native:
        warnings.warn("CUDA native tree fallback to generic tree path (constraints or feature metadata)",
                      RuntimeWarning, stacklevel=2)
    # Weighting turns an unweighted unit Hessian into sample_weight. Even
    # uniform weights use the actual array rather than an inferred constant.
    unit_hess = bool(getattr(objective, "unit_hessian", False)) and sample_weight is None

    if use_gpu:
        from numba import cuda

        from ._core._predict import _add_inplace_cuda

        binned_gpu = model.X_binned_.data
        if not _is_device(binned_gpu):
            binned_gpu = cuda.to_device(binned_gpu)
    else:
        cuda = None  # type: ignore[assignment]
        _add_inplace_cuda = None  # type: ignore[assignment]
        binned_gpu = model.X_binned_.data

    if device_state:
        raw = {
            name: cuda.to_device(np.full(n_samples, score, dtype=np.float32))
            for name, score in model._base_scores.items()
        }
        y_step = cuda.to_device(np.ascontiguousarray(y, dtype=np.float32))
        sw_step = (
            cuda.to_device(np.ascontiguousarray(sample_weight, dtype=np.float32))
            if sample_weight is not None
            else None
        )
    else:
        raw = _empty_raw(n_samples, model._base_scores)
        y_step = y
        sw_step = sample_weight

    cb_list = list(callbacks) if callbacks else []
    if early_stopping_rounds is not None:
        cb_list.append(EarlyStopping(patience=early_stopping_rounds, restore_best=True))
    cb_manager = CallbackManager(cb_list)
    state = TrainingState(model=model, n_rounds=config.n_trees)
    cb_manager.on_train_begin(state)

    prepared_eval: list[tuple[str, BinnedArray, NDArray, dict[str, Any], RawScores]] = []
    if eval_sets:
        pairs = [(item["X"], item["y"]) for item in eval_sets]
        validate_eval_set(pairs, model.X_binned_.n_features)
        for i, item in enumerate(eval_sets):
            name = item.get("name", f"eval_{i}")
            X_e = _bin_features(item["X"], model.X_binned_, config.n_bins)
            y_e = np.asarray(item["y"]).ravel()
            extra_e = item.get("extra") or {}
            raw_e = _empty_raw(X_e.n_samples, model._base_scores)
            prepared_eval.append((name, X_e, y_e, extra_e, raw_e))

    warn_if_early_stopping_without_eval_set(cb_list, prepared_eval or None)
    model.evals_result_ = {name: {eval_metric_name: []} for name, *_ in prepared_eval}

    score_eval = eval_fn or (
        lambda y_e, raw_e, extra_e: objective.loss_value(raw_e, y_e, None, extra_e)
    )

    for round_idx in range(config.n_trees):
        if device_state:
            grads = objective.step(raw, y_step, sw_step, extra)
        else:
            grads = objective.step(_as_host_raw(raw), y, sample_weight, extra)

        for name in objective.channel_names:
            grad, hess = grads[name]
            if use_gpu and not _is_device(grad):
                grad = cuda.to_device(np.ascontiguousarray(grad, dtype=np.float32))
                hess = cuda.to_device(np.ascontiguousarray(hess, dtype=np.float32))
            elif not use_gpu:
                grad = np.ascontiguousarray(grad, dtype=np.float32)
                hess = np.ascontiguousarray(hess, dtype=np.float32)

            if use_native:
                pred_buf = raw[name] if device_state else None
                legacy = fit_tree_gpu_native(
                    binned_gpu,
                    grad,
                    hess,
                    max_depth=config.max_depth,
                    min_child_weight=config.min_child_weight,
                    reg_lambda=config.reg_lambda,
                    min_gain=config.min_gain,
                    pred_gpu=pred_buf,
                    learning_rate=config.learning_rate,
                    const_hess=1.0 if unit_hess else 0.0,
                )
                tree = _legacy_to_structure(
                    legacy, model.n_features_in_, config.max_depth
                )
                if not device_state:
                    raw[name] = raw[name] + config.learning_rate * _to_host(
                        tree(model.X_binned_)
                    )
            else:
                tree = fit_tree(
                    model.X_binned_,
                    grad,
                    hess,
                    max_depth=config.max_depth,
                    min_child_weight=config.min_child_weight,
                    reg_lambda=config.reg_lambda,
                    min_gain=config.min_gain,
                    reg_alpha=config.reg_alpha,
                    subsample=config.subsample,
                    colsample_bytree=config.colsample_bytree,
                    rng=rng,
                )
                update = tree(model.X_binned_)
                if device_state:
                    if not _is_device(update):
                        update = cuda.to_device(
                            np.ascontiguousarray(_to_host(update), dtype=np.float32)
                        )
                    _add_inplace_cuda(raw[name], update, config.learning_rate)
                else:
                    raw[name] = _to_host(raw[name]) + config.learning_rate * _to_host(
                        update
                    )

            model.trees_[name].append(tree)
            for _n, X_e, _y_e, _extra_e, raw_e in prepared_eval:
                raw_e[name] = raw_e[name] + config.learning_rate * _to_host(tree(X_e))

        last_metric = None
        for name, _X_e, y_e, extra_e, raw_e in prepared_eval:
            last_metric = float(score_eval(y_e, raw_e, extra_e))
            model.evals_result_[name][eval_metric_name].append(last_metric)

        state.round_idx = round_idx
        if cb_manager.callbacks:
            state.train_loss = objective.loss_value(
                _as_host_raw(raw), y, sample_weight, extra
            )
            if last_metric is not None:
                state.val_loss = last_metric
            if not cb_manager.on_round_end(state):
                break

    cb_manager.on_train_end(state)
    return model
