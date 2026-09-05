"""Real CUDA batch histogram parity against direct sample sums."""

import hashlib

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_batch_histogram_device_oracle(checks, monkeypatch):
    import cupy as cp
    import histogram_oracle as reference
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    from openboost._core import _primitives as legacy
    from openboost.experimental import build_histograms

    rng = np.random.default_rng(103)
    bins = rng.integers(0, 256, (3, 4097), dtype=np.uint8)
    weights = rng.choice(np.array([0, .5, 1, 3], np.float32), 4097)
    grad = rng.normal(size=4097).astype(np.float32) * weights
    hess = rng.uniform(.1, 2, 4097).astype(np.float32) * weights
    random = (bins, grad, hess, rng.integers(-1, 7, 4097, dtype=np.int32),
              np.array([True, False, True, True, False, True, True]))
    empty = (np.zeros((2, 0), np.uint8), np.zeros(0, np.float32), np.zeros(0, np.float32),
             np.zeros(0, np.int32), np.ones(3, bool))
    records = []

    def forbidden(*args, **kwargs):
        raise AssertionError('Full host download or legacy histogram wrapper was called')

    for host in (reference.fixture(), random, empty):
        wanted = reference.oracle(*host)
        cpu = build_histograms(*host)
        device = tuple(cp.asarray(a) for a in host)
        # A non-default stream also exercises producer/kernel/output ordering.
        with cp.cuda.Stream(non_blocking=True) as stream:
            # Synchronize initial input copies before handing them to this stream.
            cp.cuda.Stream.null.synchronize()
            with monkeypatch.context() as m:
                m.setattr(cp, 'asnumpy', forbidden)
                m.setattr(DeviceNDArray, 'copy_to_host', forbidden)
                m.setattr(legacy, 'build_node_histograms', forbidden)
                batch = build_histograms(*device)
                assert all(isinstance(a, cp.ndarray) for a in (batch.grad, batch.hess, batch.counts, batch.active))
            stream.synchronize()
        errors = []
        for actual, cpu_array, expected in zip((batch.grad, batch.hess, batch.counts),
                                               (cpu.grad, cpu.hess, cpu.counts), wanted, strict=True):
            result = cp.asnumpy(actual)
            np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(result, cpu_array, rtol=2e-5, atol=2e-5)
            errors.append(float(np.max(np.abs(result - expected))))
        np.testing.assert_array_equal(cp.asnumpy(batch.active), host[-1])
        with pytest.raises(MemoryError):
            build_histograms(*device, memory_budget_bytes=batch.nbytes - 1)
        records.append({'input_sha256': hashlib.sha256(b''.join(a.tobytes() for a in host)).hexdigest(), 'seed': 103 if host is random else None, 'samples': host[0].shape[1], 'features': host[0].shape[0],
                        'slots': len(host[-1]), 'max_abs_errors': errors, 'bytes': batch.nbytes})
    invalid = list(device)
    invalid[1] = np.zeros(0, np.float32)
    with pytest.raises(TypeError):
        build_histograms(*invalid)
    device = tuple(cp.asarray(a) for a in reference.fixture())
    invalid = list(device)
    invalid[2] = cp.full(6, -1, cp.float32)
    with pytest.raises(ValueError):
        build_histograms(*invalid)
    invalid = list(device)
    invalid[3] = cp.full(6, 99, cp.int32)
    with pytest.raises(ValueError):
        build_histograms(*invalid)
    checks['batch_histograms'] = {'device_arrays': True, 'cases': records,
                                 'legacy_download_wrappers_blocked': True,
                                 'scope': 'named host wrappers only; scalar validation synchronization allowed; no profiler trace'}
