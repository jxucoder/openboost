"""Fixed-slot leaf reduction and an explicit, device-preserving leaf rule."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from numba import njit

from ._batch_split import _array, _namespace


@dataclass(frozen=True)
class LeafStatistics:
    grad: object
    hess: object
    counts: object
    active: object


@njit(cache=True)
def _reduce_cpu(g, h, ids, active, G, H, counts):
    for i in range(len(ids)):
        node = ids[i]
        if node >= 0 and active[node]:
            G[node] += g[i]
            H[node] += h[i]
            counts[node] += 1


@lru_cache(maxsize=1)
def _reduce_kernel():
    import cupy as cp

    return cp.RawKernel(
        r"""
    extern "C" __global__ void reduce_leaf(
      const float* g, const float* h, const int* ids, const bool* active,
      int samples, float* G, float* H, int* counts) {
      for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; i<samples;
           i+=(long long)blockDim.x*gridDim.x) {
        int node=ids[i];
        if (node<0 || !active[node]) continue;
        atomicAdd(G+node, g[i]); atomicAdd(H+node, h[i]); atomicAdd(counts+node, 1);
      }
    }
    """,
        "reduce_leaf",
    )


def reduce_leaves(grad, hess, sample_node_ids, active):
    """Reduce already-weighted G/H and physical row counts into 1..511 slots.

    Contiguous float32 statistics, int32 IDs and bool active mask must be NumPy
    or same-current-device CuPy arrays. -1 and inactive assignments are ignored.
    Empty/zero-weight rows are legal; zero-weight rows count. Results are owned
    compact arrays on the input device; only validation scalars reach the host.
    """
    xp = _namespace(grad)
    if grad.ndim != 1 or active.ndim != 1 or not 1 <= len(active) <= 511:
        raise ValueError("Require 1D statistics and 1..511 leaf slots")
    samples, slots = len(grad), len(active)
    if samples > np.iinfo(np.int32).max:
        raise ValueError("Row count exceeds int32 capacity")
    for a, dtype, shape in (
        (grad, np.float32, (samples,)),
        (hess, np.float32, (samples,)),
        (sample_node_ids, np.int32, (samples,)),
        (active, np.bool_, (slots,)),
    ):
        _array(a, xp, dtype, shape)
    if (
        not bool(xp.all(xp.isfinite(grad)))
        or not bool(xp.all(xp.isfinite(hess)))
        or bool(xp.any(hess < 0))
    ):
        raise ValueError("Require finite statistics and nonnegative curvature")
    if bool(xp.any(sample_node_ids < -1)) or bool(xp.any(sample_node_ids >= slots)):
        raise ValueError("Invalid leaf node IDs")
    G, H = xp.zeros(slots, xp.float32), xp.zeros(slots, xp.float32)
    counts, owned_active = xp.zeros(slots, xp.int32), active.copy()
    if xp is np:
        _reduce_cpu(grad, hess, sample_node_ids, owned_active, G, H, counts)
    elif samples:
        _reduce_kernel()(
            (min(65535, (samples + 255) // 256),),
            (256,),
            (grad, hess, sample_node_ids, owned_active, np.int32(samples), G, H, counts),
        )
    if not bool(xp.all(xp.isfinite(G))) or not bool(xp.all(xp.isfinite(H))):
        raise ValueError("Leaf reduction overflowed float32")
    return LeafStatistics(G, H, counts, owned_active)


class NewtonLeafRule:
    """L2 Newton leaf -G/(H+lambda); zero/zero is zero, nonzero/zero fails."""

    supported_devices = frozenset({"cpu", "cuda"})

    def values(self, grad, hess, *, config, context):
        xp = context.xp
        if config.reg_alpha != 0:
            raise ValueError("NewtonLeafRule supports L2 only")
        if (
            not np.isscalar(config.reg_lambda)
            or not np.isfinite(config.reg_lambda)
            or config.reg_lambda < 0
        ):
            raise ValueError("reg_lambda must be finite and nonnegative")
        denominator = hess.astype(xp.float64) + float(config.reg_lambda)
        if bool(xp.any((denominator == 0) & (grad != 0))):
            raise ValueError("Zero curvature denominator with nonzero gradient")
        values = (-grad.astype(xp.float64) / xp.where(denominator == 0, 1.0, denominator)).astype(
            xp.float32
        )
        if not bool(xp.all(xp.isfinite(values))):
            raise ValueError("Newton leaf values overflowed float32")
        return values


def leaf_values(grad, hess, sample_node_ids, active, *, leaf_rule=None, config=None, context=None):
    """Reduce real rows, then apply a declared CPU/CUDA leaf rule.

    Rule.values(G,H,config=...,context=...) must return contiguous finite float32
    (slots,) on the same device. G/H are read-only by contract: private compact
    copies protect stored statistics, and mutation is rejected. Empty/inactive
    and zero-G/zero-H slots must return zero. Result ownership is detached from
    plugin scratch arrays. The caller supplies a shared ExecutionContext during
    fitting; a standalone call creates one from the config seed.
    """
    from .._trainer import TrainerConfig
    from ..experimental._contracts import ExecutionContext

    xp = _namespace(grad)
    device = "cpu" if xp is np else "cuda"
    if config is None:
        config = TrainerConfig()
    if not isinstance(config, TrainerConfig):
        raise TypeError("config must be TrainerConfig")
    if context is None:
        context = ExecutionContext(device, xp, np.random.default_rng(config.random_state), 0)
    if (
        not isinstance(context, ExecutionContext)
        or context.device != device
        or context.xp is not xp
    ):
        raise ValueError("Leaf context must match the input device")
    rule = NewtonLeafRule() if leaf_rule is None else leaf_rule
    if (
        not isinstance(getattr(rule, "supported_devices", None), frozenset)
        or device not in rule.supported_devices
        or not callable(getattr(rule, "values", None))
    ):
        raise ValueError("Leaf rule must declare this device and implement values")
    stats = reduce_leaves(grad, hess, sample_node_ids, active)
    G, H = stats.grad.copy(), stats.hess.copy()
    if xp is np:
        G.flags.writeable = H.flags.writeable = False
    values = rule.values(G, H, config=replace(config), context=context)
    if not bool(xp.array_equal(G, stats.grad)) or not bool(xp.array_equal(H, stats.hess)):
        raise ValueError("Leaf rule mutated input statistics")
    _array(values, xp, np.float32, stats.grad.shape)
    if not bool(xp.all(xp.isfinite(values))):
        raise ValueError("Leaf rule must return finite values")
    empty = ~stats.active | (stats.counts == 0) | ((stats.grad == 0) & (stats.hess == 0))
    if bool(xp.any(empty & (values != 0))):
        raise ValueError("Empty/inactive/zero-statistic leaves must be zero")
    return values.copy()
