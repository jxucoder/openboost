"""Explicit CUDA storage ownership; training recipes remain CPU-only."""

import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True, eq=False)
class DeviceBuffer:
    """Opaque allocation handle. Only its creating context can resolve storage."""

    shape: tuple[int, ...]
    dtype: str
    nbytes: int


class ExecutionContext:
    """Single-thread CUDA stream and private allocation pool, without CPU fallback.

    max_bytes caps the private CuPy pool, not CUDA context/driver memory. Public
    handles never expose mutable owned arrays. No training or callback API yet.
    """

    def __init__(self, device="cuda:0", *, max_bytes=16 * 1024**2):
        if not isinstance(device, str) or not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device):
            raise ValueError("explicit cuda:<nonnegative index> required")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("positive integer max_bytes required")
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError("CUDA storage requires CuPy; CPU fallback is unavailable") from error
        self._cp = cp
        self._device = cp.cuda.Device(int(device.split(":")[1]))
        self._thread = threading.get_ident()
        self._closed = False
        self._buffers = {}
        self._limit = max_bytes
        self._counts = dict(
            upload_bytes=0,
            export_bytes=0,
            device_copy_bytes=0,
            synchronizations=0,
            live_bytes=0,
            peak_live_bytes=0,
            peak_pool_bytes=0,
        )
        with self._device:
            self._stream = cp.cuda.Stream(non_blocking=True)
            self._pool = cp.cuda.MemoryPool()
            self._pool.set_limit(size=max_bytes)

    def _check(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("execution context belongs to another thread")
        if self._closed:
            raise RuntimeError("execution context is closed")

    @contextmanager
    def _scope(self):
        self._check()
        with self._device, self._stream, self._cp.cuda.using_allocator(self._pool.malloc):
            yield

    def _array(self, buffer):
        self._check()
        if not isinstance(buffer, DeviceBuffer) or buffer not in self._buffers:
            raise ValueError("foreign, forged or released buffer")
        return self._buffers[buffer]

    def _reserve(self, nbytes):
        if self._counts["live_bytes"] + nbytes > self._limit:
            raise MemoryError("context live allocation budget exceeded")

    def _register(self, array):
        handle = DeviceBuffer(tuple(array.shape), array.dtype.str, array.nbytes)
        self._buffers[handle] = array
        self._counts["live_bytes"] += array.nbytes
        self._counts["peak_live_bytes"] = max(
            self._counts["peak_live_bytes"], self._counts["live_bytes"]
        )
        self._counts["peak_pool_bytes"] = max(
            self._counts["peak_pool_bytes"], self._pool.total_bytes()
        )
        return handle

    def _empty(self, shape, dtype):
        """Internal owned output allocation, including genuine zero-length views."""
        dtype = np.dtype(dtype)
        self._reserve(int(np.prod(shape)) * dtype.itemsize)
        with self._scope():
            return self._register(self._cp.empty(shape, dtype=dtype))

    def synchronize(self):
        self._check()
        with self._device:
            self._stream.synchronize()
        self._counts["synchronizations"] += 1

    def upload(self, value):
        """Snapshot host values and finish the transfer before returning a handle."""
        self._check()
        if hasattr(value, "__cuda_array_interface__") or isinstance(value, DeviceBuffer):
            raise ValueError("upload accepts host data only; use context.copy for owned buffers")
        host = np.array(value, copy=True, order="C")
        if host.dtype.str[1:] not in ("f4", "f8", "i4", "i8", "u1", "b1") or not host.size:
            raise ValueError("nonempty float32/64, int32/64, uint8 or bool host data required")
        if not host.dtype.isnative:
            raise ValueError("native byte order required")
        self._reserve(host.nbytes)
        with self._scope():
            array = self._cp.array(host, copy=True, order="C")
            self.synchronize()
            handle = self._register(array)
        self._counts["upload_bytes"] += host.nbytes
        return handle

    def copy(self, buffer):
        """Independent device storage; no host transfer and no writable alias."""
        source = self._array(buffer)
        self._reserve(source.nbytes)
        with self._scope():
            result = self._register(source.copy(order="C"))
        self._counts["device_copy_bytes"] += source.nbytes
        return result

    def export(self, buffer):
        """Explicit blocking device-to-host copy; mutation cannot affect the buffer."""
        source = self._array(buffer)
        with self._scope():
            host = self._cp.asnumpy(source, stream=self._stream, blocking=True)
        self._counts["synchronizations"] += 1
        self._counts["export_bytes"] += source.nbytes
        return host

    def release(self, buffer):
        source = self._array(buffer)
        self.synchronize()
        del self._buffers[buffer]
        self._counts["live_bytes"] -= source.nbytes

    @property
    def metrics(self):
        """Detached counters; pool peak is sampled at successful allocation boundaries."""
        return MappingProxyType(dict(self._counts))

    def close(self):
        if self._closed:
            return
        self._check()
        try:
            self.synchronize()
        finally:
            self._buffers.clear()
            self._counts["live_bytes"] = 0
            self._closed = True
            with self._device:
                self._pool.free_all_blocks()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()
