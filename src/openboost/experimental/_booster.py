"""Thin CPU facade over the unified trainer."""

import warnings
from dataclasses import fields, replace

import numpy as np

from .._array import BinnedArray
from .._backends import backend_context
from .._trainer import TrainerConfig, fit_boosting, predict_raw
from .._validation import validate_sample_weight
from ._builders import ConstantSchedule, CPUHistogramBuilder, ExtensionSession
from ._contracts import ObjectiveBridge, channels_of


def validate_config(config):
    if not isinstance(config, TrainerConfig):
        raise TypeError('config must be TrainerConfig')
    for name in ('n_trees', 'max_depth', 'n_bins'):
        v = getattr(config, name)
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f'{name} must be an integer')
    if config.n_trees < 1 or not 0 <= config.max_depth <= 8 or not 2 <= config.n_bins <= 254:
        raise ValueError('Require n_trees>=1, max_depth in [0,8], n_bins in [2,254]')
    for f in fields(config):
        if f.name in ('n_trees', 'max_depth', 'n_bins', 'random_state'):
            continue
        v = getattr(config, f.name)
        if not np.isscalar(v) or not np.isfinite(v) or v < 0:
            raise ValueError(f'{f.name} must be finite and nonnegative')
    if not 0 < config.subsample <= 1 or not 0 < config.colsample_bytree <= 1:
        raise ValueError('sampling ratios must be in (0,1]')


def target(y):
    if hasattr(y, '__cuda_array_interface__'):
        raise ValueError('P3 requires CPU input arrays')
    y = np.asarray(y)
    if y.ndim != 1 or not len(y):
        raise ValueError('y must be a nonempty 1D array')
    y = np.ascontiguousarray(y, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        raise ValueError('y must be finite')
    return y


def features(X, n=None):
    if isinstance(X, BinnedArray):
        if X.device != 'cpu' or not isinstance(X.data, np.ndarray):
            raise ValueError('P3 requires CPU binned arrays')
        shape = (X.n_samples, X.n_features)
    else:
        if hasattr(X, '__cuda_array_interface__'):
            raise ValueError('P3 requires CPU feature arrays')
        X = np.asarray(X)
        shape = X.shape
        if X.dtype.kind not in 'fiu' or np.any(np.isinf(X)):
            raise ValueError('X must be numeric, finite or NaN')
    if len(shape) != 2 or min(shape) < 1 or (n is not None and shape[0] != n):
        raise ValueError('X must be nonempty 2D with one row per target')
    return X


class Booster:
    """Experimental CPU boosting. Plugin objects are training-time dependencies."""

    def __init__(self, *, objective, tree_builder=None, step_schedule=None, config=None, device='cpu', fallback='error'):
        self.objective = objective
        self.tree_builder = tree_builder if tree_builder is not None else CPUHistogramBuilder()
        self.step_schedule = step_schedule if step_schedule is not None else ConstantSchedule()
        self.config = replace(config) if config is not None else TrainerConfig()
        self.device = device
        self.fallback = fallback
        self.trees_ = {}

    def fit(self, X, y, sample_weight=None, *, eval_sets=None, callbacks=None,
            early_stopping_rounds=None):
        validate_config(self.config)
        channels = channels_of(self.objective)
        if (not isinstance(getattr(self.tree_builder, 'supported_devices', None), frozenset)
                or 'cpu' not in self.tree_builder.supported_devices
                or not callable(getattr(self.tree_builder, 'build', None))):
            raise ValueError('P3 builder must declare supported_devices including CPU and implement build')
        if not callable(getattr(self.step_schedule, 'coefficients', None)):
            raise ValueError('StepSchedule must implement coefficients')
        if type(self.tree_builder) is CPUHistogramBuilder and self.config.reg_lambda == 0 and self.config.min_child_weight == 0:
            raise ValueError('CPUHistogramBuilder requires positive min_child_weight when reg_lambda=0')
        from .._callbacks import LearningRateScheduler
        if any(isinstance(cb, LearningRateScheduler) for cb in callbacks or []):
            raise ValueError('Use StepSchedule instead of a learning-rate-mutating callback')
        if self.device not in ('cpu', 'cuda') or self.fallback not in ('error', 'warn'):
            raise ValueError('Invalid device or fallback policy')
        if 'cpu' not in self.objective.supported_devices:
            raise ValueError('P3 requires an objective supporting CPU')
        reason = None
        if self.device != 'cpu':
            reason = 'Experimental P3 supports CPU execution only'
            if self.fallback == 'error':
                raise ValueError(reason)
            warnings.warn(reason, RuntimeWarning, stacklevel=2)
        y = target(y)
        X = features(X, len(y))
        if hasattr(sample_weight, "__cuda_array_interface__"):
            raise ValueError("P3 requires CPU weights")
        sample_weight = validate_sample_weight(sample_weight, len(y))
        if sample_weight is not None and (not np.all(np.isfinite(sample_weight)) or not np.any(sample_weight > 0)):
            raise ValueError("Weights must be finite with positive total weight")
        prepared = []
        for item in eval_sets or []:
            if not isinstance(item, dict) or set(item) - {'X', 'y', 'name'}:
                raise ValueError('eval_sets entries accept X, y, and optional name only')
            y_e = target(item['y'])
            prepared.append({**item, 'X': features(item['X'], len(y_e)), 'y': y_e})
        rng = np.random.default_rng(self.config.random_state)
        bridge = ObjectiveBridge(self.objective, rng)
        with backend_context('cpu'):
            fit_boosting(self, bridge, X, y, config=replace(self.config), sample_weight=sample_weight,
                         eval_sets=prepared, callbacks=callbacks, early_stopping_rounds=early_stopping_rounds,
                         rng=rng, extension=ExtensionSession(bridge, self.tree_builder, self.step_schedule, replace(self.config)))
        self.channel_names_ = channels
        self.fit_report_ = dict(requested_device=self.device, actual_device='cpu',
                                objective_device='cpu', tree_device='cpu', update_device='cpu',
                                eval_device='cpu' if prepared else None, builder_path=type(self.tree_builder).__name__,
                                fallback_reason=reason, random_state=self.config.random_state,
                                tree_counts={k: len(v) for k, v in self.trees_.items()},
                                timing_scope='No performance timing collected; synchronous CPU execution')
        return self

    def predict_raw(self, X):
        with backend_context('cpu'):
            return predict_raw(self, features(X))
