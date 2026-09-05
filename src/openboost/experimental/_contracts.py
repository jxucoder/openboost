"""Small extension contracts; deliberately separate from the stable API."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .._distributions import get_distribution
from .._objectives import DistributionObjective


@dataclass(frozen=True)
class ExecutionContext:
    device: str
    xp: Any
    rng: np.random.Generator
    round_idx: int
    channel: str | None = None


def readonly(value):
    view = value.view()
    view.flags.writeable = False
    return view


def mapping(values):
    return MappingProxyType({k: readonly(v) for k, v in values.items()})


def vector(value, n, label):
    if not isinstance(value, np.ndarray) or hasattr(value, '__cuda_array_interface__'):
        raise TypeError(f'{label} must be a CPU numpy array')
    if value.dtype != np.float32 or value.shape != (n,) or not value.flags.c_contiguous:
        raise ValueError(f'{label} must be contiguous float32 ({n},)')
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{label} contains non-finite values')
    return value


def channels_of(objective):
    channels = getattr(objective, 'channel_names', None)
    if (not isinstance(channels, tuple) or not channels
            or any(not isinstance(k, str) or not k for k in channels)
            or len(set(channels)) != len(channels)):
        raise ValueError('channel_names must be a nonempty tuple of unique strings')
    devices = getattr(objective, 'supported_devices', None)
    if not isinstance(devices, frozenset) or not devices or not devices <= {'cpu', 'cuda'}:
        raise ValueError('supported_devices must be an explicit nonempty frozenset')
    for name in ('init_raw', 'step', 'loss_value', 'constrain'):
        if not callable(getattr(objective, name, None)):
            raise ValueError(f'Objective must implement {name}')
    return channels


def exact_keys(values, channels, label):
    if not hasattr(values, 'keys') or set(values) != set(channels):
        raise ValueError(f'{label} must have exactly the declared channels')


class DistributionObjectiveAdapter:
    """Expose existing distribution math with exact-type CUDA capability."""

    supported_devices = frozenset({'cpu'})

    def __init__(self, distribution='normal', *, natural=False):
        self._objective = DistributionObjective(get_distribution(distribution), natural=natural)
        self.channel_names = tuple(self._objective.channel_names)
        self.supported_devices = frozenset({'cpu', 'cuda'} if self._objective.device_capable else {'cpu'})

    def init_raw(self, y, sample_weight=None, extra=None):
        return self._objective.init_raw(y, sample_weight, extra)

    def step(self, raw, y, sample_weight=None, extra=None, *, context):
        out = self._objective.step(raw, y, sample_weight, extra)
        if context.device == 'cuda':
            return {k: tuple(context.xp.asarray(a) for a in pair) for k, pair in out.items()}
        return out

    def loss_value(self, raw, y, sample_weight=None, extra=None, *, context):
        return self._objective.loss_value(raw, y, sample_weight, extra)

    def constrain(self, raw, extra=None):
        return self._objective.constrain(raw, extra)


class ObjectiveBridge:
    """Validate plugin boundaries while letting the existing trainer own the loop."""

    device_capable = False
    unit_hessian = False

    def __init__(self, objective, rng):
        self.objective = objective
        self.channel_names = channels_of(objective)
        self.rng = rng
        self.round_idx = -1

    @property
    def context(self):
        return ExecutionContext('cpu', np, self.rng, self.round_idx)

    def init_raw(self, y, sample_weight=None, extra=None):
        base = self.objective.init_raw(readonly(y), None if sample_weight is None else readonly(sample_weight), extra)
        exact_keys(base, self.channel_names, 'init_raw')
        result = {}
        for k, value in base.items():
            if not np.isscalar(value) or not np.isfinite(value) or abs(value) > np.finfo(np.float32).max:
                raise ValueError('init_raw values must be finite float32-representable scalars')
            result[k] = float(value)
        return result

    def step(self, raw, y, sample_weight=None, extra=None):
        self.round_idx += 1
        for k in self.channel_names:
            vector(raw[k], len(y), f'raw[{k}]')
        out = self.objective.step(mapping(raw), readonly(y), None if sample_weight is None else readonly(sample_weight), extra, context=self.context)
        exact_keys(out, self.channel_names, 'step')
        seen = list(raw.values()) + [y] + ([] if sample_weight is None else [sample_weight])
        result = {}
        for k in self.channel_names:
            pair = out[k]
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError('Each channel must return a (grad, hess) tuple')
            for label, a in zip(('grad', 'hess'), pair, strict=True):
                vector(a, len(y), f'{k}.{label}')
                if label == 'hess' and np.any(a < 0):
                    raise ValueError('Effective Hessian must be nonnegative')
                if any(np.shares_memory(a, old) for old in seen):
                    raise ValueError('Objective buffers must not alias inputs or other channel statistics')
                seen.append(a)
            result[k] = pair
        return result

    def loss_value(self, raw, y, sample_weight=None, extra=None):
        value = self.objective.loss_value(mapping(raw), readonly(y), None if sample_weight is None else readonly(sample_weight), extra, context=self.context)
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError('loss_value must return a finite scalar')
        return float(value)
