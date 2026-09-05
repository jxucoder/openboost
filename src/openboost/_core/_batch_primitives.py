"""Experimental fixed-slot histograms; arrays remain on the input device."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit


@dataclass(frozen=True)
class HistogramBatch:
    grad: object
    hess: object
    counts: object
    active: object

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.grad, self.hess, self.counts, self.active))


@njit(cache=True)
def _accumulate_cpu(bins, grad, hess, ids, active, out_g, out_h, counts):
    for i in range(len(ids)):
        node = ids[i]
        if node >= 0 and active[node]:
            counts[node] += 1
            for f in range(bins.shape[0]):
                b = bins[f, i]
                out_g[node, f, b] += grad[i]
                out_h[node, f, b] += hess[i]


@lru_cache(maxsize=1)
def _cuda_kernel():
    import cupy as cp

    # Sample-centric scatter, following the existing native histogram strategy.
    # Counts are per node (not Hessian-derived), written only by feature zero.
    return cp.RawKernel(r'''
    extern "C" __global__ void histogram(
        const unsigned char* bins, const float* g, const float* h,
        const int* ids, const bool* active, const int samples, const int features,
        float* out_g, float* out_h, int* counts) {
        const long long total = (long long)samples * features;
        for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             j < total; j += (long long)blockDim.x * gridDim.x) {
            const int i = j % samples;
            const int f = j / samples;
            const int node = ids[i];
            if (node < 0 || !active[node]) continue;
            const long long offset = ((long long)node * features + f) * 256 + bins[j];
            atomicAdd(out_g + offset, g[i]);
            atomicAdd(out_h + offset, h[i]);
            if (f == 0) atomicAdd(counts + node, 1);
        }
    }
    ''', 'histogram')


def build_histograms(binned, grad, hess, sample_node_ids, active, *, memory_budget_bytes=256 * 1024**2):
    """Aggregate already-weighted G/H by routed node IDs and feature bins.

    Inputs must all be contiguous NumPy or same-device CuPy arrays: bins uint8
    (features, samples), G/H float32 (samples,), IDs int32 (samples,), active
    bool (slots,). IDs -1 exclude rows; inactive slots ignore assigned rows.
    Empty and zero-weight rows are legal; counts count rows, independently of H.
    Bin 255 is retained. Up to 511 fixed slots (depth 8); no bin reinterpretation.

    Budget covers returned histogram/count/mask arrays, excluding inputs and
    temporary device validation masks. CUDA uses the current CuPy stream and
    only scalar validation results reach host. No deterministic CUDA sum order
    is promised. The caller owns inputs; outputs are newly allocated.
    """
    xp = np
    if hasattr(binned, '__cuda_array_interface__'):
        import cupy as cp
        xp = cp
    arrays = (binned, grad, hess, sample_node_ids, active)
    dtypes = (np.uint8, np.float32, np.float32, np.int32, np.bool_)
    for a, dtype in zip(arrays, dtypes, strict=True):
        if not isinstance(a, xp.ndarray) or a.dtype != dtype or not a.flags.c_contiguous:
            raise TypeError('Inputs must be contiguous, correctly typed arrays on one device')
        if xp is not np and a.device.id != xp.cuda.runtime.getDevice():
            raise ValueError('All inputs must be on the current CUDA device')
    if binned.ndim != 2 or binned.shape[0] < 1 or active.ndim != 1 or not 1 <= len(active) <= 511:
        raise ValueError('Require features>0 and 1..511 fixed node slots')
    n_features, n_samples = binned.shape
    if n_samples > np.iinfo(np.int32).max or n_features > np.iinfo(np.int32).max:
        raise ValueError('Array dimensions exceed int32 indexing/count capacity')
    if any(a.shape != (n_samples,) for a in (grad, hess, sample_node_ids)):
        raise ValueError('Statistics and IDs require one entry per sample')
    required = len(active) * (n_features * 256 * 8 + 5)
    if isinstance(memory_budget_bytes, bool) or not isinstance(memory_budget_bytes, int) or memory_budget_bytes < 0:
        raise ValueError('memory_budget_bytes must be a nonnegative integer')
    if required > memory_budget_bytes:
        raise MemoryError(f'Histogram batch requires {required} bytes; budget is {memory_budget_bytes}')
    if not bool(xp.all(xp.isfinite(grad))) or not bool(xp.all(xp.isfinite(hess))) or bool(xp.any(hess < 0)):
        raise ValueError('Require finite G/H and nonnegative H')
    if bool(xp.any(sample_node_ids < -1)) or bool(xp.any(sample_node_ids >= len(active))):
        raise ValueError('Node IDs must be -1 or a valid fixed slot')
    shape = (len(active), n_features, 256)
    out_g, out_h = xp.zeros(shape, xp.float32), xp.zeros(shape, xp.float32)
    counts = xp.zeros(len(active), xp.int32)
    active_out = active.copy()
    if xp is np:
        _accumulate_cpu(binned, grad, hess, sample_node_ids, active_out, out_g, out_h, counts)
    elif n_samples:
        blocks = min(65535, (n_features * n_samples + 255) // 256)
        _cuda_kernel()((blocks,), (256,), (binned, grad, hess, sample_node_ids, active_out,
                                          np.int32(n_samples), np.int32(n_features), out_g, out_h, counts))
    if not bool(xp.all(xp.isfinite(out_g))) or not bool(xp.all(xp.isfinite(out_h))):
        raise ValueError('Histogram accumulation overflowed float32')
    return HistogramBatch(out_g, out_h, counts, active_out)
