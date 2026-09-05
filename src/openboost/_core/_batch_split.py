"""Numeric L2 split/routing primitives for a bounded level-wise builder."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit

from ._batch_primitives import HistogramBatch


@dataclass(frozen=True)
class SplitBatch:
    feature: object
    threshold: object
    left_child: object
    right_child: object
    gain: object
    valid: object
    n_features: int


def _namespace(a):
    if hasattr(a, "__cuda_array_interface__"):
        import cupy as cp

        return cp
    return np


def _array(a, xp, dtype, shape):
    if (
        not isinstance(a, xp.ndarray)
        or a.dtype != dtype
        or a.shape != shape
        or not a.flags.c_contiguous
    ):
        raise TypeError("Require matching contiguous arrays with declared shape/dtype/device")
    if xp is not np and a.device.id != xp.cuda.runtime.getDevice():
        raise ValueError("Inputs must be on the current CUDA device")


@njit(cache=True)
def _split_cpu(
    g, h, counts, active, lam, min_child, min_gain, feature, threshold, left, right, gain, valid
):
    for node in range(len(active)):
        if not active[node] or counts[node] < 2 or 2 * node + 2 >= len(active):
            continue
        best = 0.0
        for f in range(g.shape[1]):
            G, H = 0.0, 0.0
            for b in range(255):
                G += np.float64(g[node, f, b])
                H += np.float64(h[node, f, b])
            if H <= 0:
                continue
            parent = G * G / (H + lam)
            GL, HL = 0.0, 0.0
            for b in range(255):
                GL += np.float64(g[node, f, b])
                HL += np.float64(h[node, f, b])
                GR, HR = G - GL, H - HL
                if HL <= 0 or HR <= 0 or min_child > HL or min_child > HR:
                    continue
                score = GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent
                if np.isfinite(score) and score > best and score >= min_gain:
                    best = score
                    feature[node], threshold[node] = f, b
        if feature[node] >= 0:
            left[node], right[node] = 2 * node + 1, 2 * node + 2
            gain[node], valid[node] = best, True


@njit(cache=True)
def _partition_cpu(bins, ids, feature, threshold, left, right, valid, out):
    for i in range(len(ids)):
        node = ids[i]
        out[i] = node
        if node >= 0 and valid[node]:
            out[i] = left[node] if bins[feature[node], i] <= threshold[node] else right[node]


@lru_cache(maxsize=1)
def _kernels():
    import cupy as cp

    code = r"""
    extern "C" __global__ void split(
      const float* g, const float* h, const int* counts, const bool* active,
      int slots, int features, double lam, double min_child, double min_gain,
      int* feature, int* threshold, int* left, int* right, double* gain, bool* valid) {
      int node = blockIdx.x * blockDim.x + threadIdx.x;
      if (node >= slots || !active[node] || counts[node] < 2 || 2*node+2 >= slots) return;
      double best = 0.0;
      for (int f=0; f<features; ++f) {
        long long offset = ((long long)node*features+f)*256;
        double G=0.0, H=0.0;
        for (int b=0; b<255; ++b) { G+=(double)g[offset+b]; H+=(double)h[offset+b]; }
        if (H<=0.0) continue;
        double parent=G*G/(H+lam), GL=0.0, HL=0.0;
        for (int b=0; b<255; ++b) {
          GL+=(double)g[offset+b]; HL+=(double)h[offset+b];
          double GR=G-GL, HR=H-HL;
          if (HL<=0.0 || HR<=0.0 || HL<min_child || HR<min_child) continue;
          double score=GL*GL/(HL+lam)+GR*GR/(HR+lam)-parent;
          if (isfinite(score) && score>best && score>=min_gain) {
            best=score; feature[node]=f; threshold[node]=b;
          }
        }
      }
      if (feature[node]>=0) {
        left[node]=2*node+1; right[node]=2*node+2; gain[node]=best; valid[node]=true;
      }
    }
    extern "C" __global__ void route(
      const unsigned char* bins, const int* ids, int samples,
      const int* feature, const int* threshold, const int* left,
      const int* right, const bool* valid, int* out) {
      for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; i<samples;
           i+=(long long)gridDim.x*blockDim.x) {
        int node=ids[i]; out[i]=node;
        if (node>=0 && valid[node])
          out[i]=bins[(long long)feature[node]*samples+i]<=threshold[node] ? left[node] : right[node];
      }
    }
    """
    return tuple(cp.RawKernel(code, name, options=("--fmad=false",)) for name in ("split", "route"))


def find_splits(histograms, *, reg_lambda=1.0, min_child_weight=1.0, min_gain=0.0):
    """Find numeric L2 splits in fixed slots; no missing-value or category support.

    Both children must have positive H and meet min_child_weight (inclusive).
    Gain is unhalved, positive and >= min_gain. Exact ties prefer feature then
    threshold order. Scores/prefix sums use float64 from float32 histograms.
    Invalid/terminal slots have -1 indices, zero gain and valid=False. The
    caller must provide numeric bins; histogram bin 255 must have zero G/H.
    This conservative positive-curvature rule excludes zero-effective-weight
    children even when min_child_weight=0. No GPU histogram download occurs.
    """
    if not isinstance(histograms, HistogramBatch):
        raise TypeError("Expected HistogramBatch")
    g, h, counts, active = histograms.grad, histograms.hess, histograms.counts, histograms.active
    xp = _namespace(g)
    if g.ndim != 3 or g.shape[2] != 256 or not 1 <= g.shape[0] <= 511 or g.shape[1] < 1:
        raise ValueError("Invalid histogram shape")
    slots, features, _ = g.shape
    _array(g, xp, np.float32, g.shape)
    _array(h, xp, np.float32, g.shape)
    _array(counts, xp, np.int32, (slots,))
    _array(active, xp, np.bool_, (slots,))
    for value in (reg_lambda, min_child_weight, min_gain):
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError("Split parameters must be finite nonnegative scalars")
    if (
        not bool(xp.all(xp.isfinite(g)))
        or not bool(xp.all(xp.isfinite(h)))
        or bool(xp.any(h < 0))
        or bool(xp.any(counts < 0))
    ):
        raise ValueError("Invalid histogram statistics")
    if bool(xp.any(g[:, :, 255] != 0)) or bool(xp.any(h[:, :, 255] != 0)):
        raise ValueError("Numeric split primitive does not support missing bins")
    feature, threshold, left, right = (xp.full(slots, -1, xp.int32) for _ in range(4))
    gain, valid = xp.zeros(slots, xp.float64), xp.zeros(slots, xp.bool_)
    args = (g, h, counts, active)
    params = tuple(float(v) for v in (reg_lambda, min_child_weight, min_gain))
    outputs = (feature, threshold, left, right, gain, valid)
    if xp is np:
        _split_cpu(*args, *params, *outputs)
    else:
        _kernels()[0](
            ((slots + 127) // 128,),
            (128,),
            (
                *args,
                np.int32(slots),
                np.int32(features),
                *(np.float64(v) for v in params),
                *outputs,
            ),
        )
    return SplitBatch(*outputs, features)


def partition(binned, sample_node_ids, splits):
    """Return new node IDs from real rows. Leaves and ID -1 stay unchanged.

    All arrays remain on the input device/current CuPy stream. Missing bin 255
    is rejected, including zero-weight rows. Child IDs must match fixed slots
    left=2*i+1/right=2*i+2. Input IDs and split arrays are never modified.
    """
    if not isinstance(splits, SplitBatch):
        raise TypeError("Expected SplitBatch")
    xp = _namespace(binned)
    if binned.ndim != 2 or binned.shape[0] != splits.n_features:
        raise ValueError("Feature count must match split batch")
    samples = binned.shape[1]
    if samples > np.iinfo(np.int32).max:
        raise ValueError("Sample count exceeds int32 capacity")
    _array(binned, xp, np.uint8, binned.shape)
    _array(sample_node_ids, xp, np.int32, (samples,))
    slots = len(splits.valid)
    if not 1 <= slots <= 511:
        raise ValueError("Invalid slot count")
    for a in (splits.feature, splits.threshold, splits.left_child, splits.right_child):
        _array(a, xp, np.int32, (slots,))
    _array(splits.valid, xp, np.bool_, (slots,))
    _array(splits.gain, xp, np.float64, (slots,))
    if bool(xp.any(binned == 255)):
        raise ValueError("Numeric routing does not support missing bins")
    if bool(xp.any(sample_node_ids < -1)) or bool(xp.any(sample_node_ids >= slots)):
        raise ValueError("Invalid sample node IDs")
    mask = splits.valid
    node = xp.arange(slots, dtype=xp.int32)
    invalid = (
        (splits.feature < 0)
        | (splits.feature >= splits.n_features)
        | (splits.threshold < 0)
        | (splits.threshold >= 255)
        | (splits.left_child != 2 * node + 1)
        | (splits.right_child != 2 * node + 2)
        | (splits.right_child >= slots)
    )
    if bool(xp.any(mask & invalid)):
        raise ValueError("Invalid fixed-slot split routing")
    out = xp.empty_like(sample_node_ids)
    arrays = (
        binned,
        sample_node_ids,
        splits.feature,
        splits.threshold,
        splits.left_child,
        splits.right_child,
        mask,
        out,
    )
    if xp is np:
        _partition_cpu(*arrays)
    elif samples:
        _kernels()[1](
            (min(65535, (samples + 255) // 256),),
            (256,),
            (binned, sample_node_ids, np.int32(samples), *arrays[2:]),
        )
    return out
