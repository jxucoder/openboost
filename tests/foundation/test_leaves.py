"""Real CUDA row reduction and custom leaf rule evidence."""

import hashlib

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_batch_leaf_rule_oracle(checks, monkeypatch):
    import cupy as cp
    import leaf_oracle as reference
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    from openboost._core import _primitives as legacy
    from openboost.experimental import TrainerConfig, leaf_values, reduce_leaves

    def forbidden(*args, **kwargs):
        raise AssertionError("Full sample-array download or legacy wrapper called")

    rng = np.random.default_rng(109)
    w = rng.choice(np.array([0, 0.5, 1, 3], np.float32), 4097)
    random = (
        rng.integers(-5, 6, 4097).astype(np.float32) * w,
        w,
        rng.integers(-1, 7, 4097, dtype=np.int32),
        np.array([True, False, True, True, False, True, True]),
    )
    empty = (
        np.zeros(0, np.float32),
        np.zeros(0, np.float32),
        np.zeros(0, np.int32),
        np.ones(3, bool),
    )
    records = []
    for host in (reference.example(), random, empty):
        device = tuple(cp.asarray(a) for a in host)
        cp.cuda.Stream.null.synchronize()
        with cp.cuda.Stream(non_blocking=True) as stream:
            with monkeypatch.context() as m:
                m.setattr(cp, "asnumpy", forbidden)
                m.setattr(DeviceNDArray, "copy_to_host", forbidden)
                m.setattr(legacy, "compute_leaf_values", forbidden)
                stats = reduce_leaves(*device)
                normal = leaf_values(*device)
                bounded = leaf_values(*device, leaf_rule=reference.BoundedLeaf())
                assert all(
                    isinstance(a, cp.ndarray)
                    for a in (stats.grad, stats.hess, stats.counts, normal, bounded)
                )
            stream.synchronize()
        G, H, counts = reference.direct(*host)
        for actual, expected in zip(
            (stats.grad, stats.hess, stats.counts), (G, H, counts), strict=True
        ):
            np.testing.assert_array_equal(cp.asnumpy(actual), expected)
        wanted = -G / (H + 1)
        np.testing.assert_allclose(cp.asnumpy(normal), wanted, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            cp.asnumpy(bounded), np.clip(wanted, -0.5, 0.5), atol=1e-6, rtol=1e-6
        )
        records.append(
            {
                "input_sha256": hashlib.sha256(b"".join(a.tobytes() for a in host)).hexdigest(),
                "samples": len(host[0]),
                "slots": len(host[-1]),
                "seed": 109 if host is random else None,
                "counts": counts.tolist(),
                "grad": G.tolist(),
                "hess": H.tolist(),
                "values": cp.asnumpy(normal).tolist(),
                "bounded": cp.asnumpy(bounded).tolist(),
            }
        )
    y, w = cp.asarray([2, 4], dtype=cp.float32), cp.asarray([1, 3], dtype=cp.float32)
    ids, active = cp.zeros(2, cp.int32), cp.asarray([True])
    rounds = []
    for rule in (None, reference.BoundedLeaf()):
        raw = cp.zeros(2, cp.float32)
        history = []
        with monkeypatch.context() as m:
            m.setattr(cp, "asnumpy", forbidden)
            m.setattr(DeviceNDArray, "copy_to_host", forbidden)
            for _ in range(2):
                g = (raw - y) * w
                history.append(g.copy())
                raw += leaf_values(g, w.copy(), ids, active, leaf_rule=rule)[ids]
        rounds.append(
            {"raw": cp.asnumpy(raw).tolist(), "second_gradient": cp.asnumpy(history[1]).tolist()}
        )
    np.testing.assert_allclose(rounds[0]["raw"], [3.36, 3.36], atol=1e-6)
    np.testing.assert_array_equal(rounds[1]["raw"], [1, 1])
    np.testing.assert_allclose(rounds[0]["second_gradient"], [0.8, -3.6], atol=1e-6)
    np.testing.assert_array_equal(rounds[1]["second_gradient"], [-1.5, -10.5])
    zero = cp.zeros(2, cp.float32)
    assert float(leaf_values(zero, zero, ids, active, config=TrainerConfig(reg_lambda=0))[0]) == 0
    with pytest.raises(ValueError, match="curvature"):
        leaf_values(cp.ones(2, cp.float32), zero, ids, active, config=TrainerConfig(reg_lambda=0))

    class Bad:
        supported_devices = frozenset({"cuda"})

        def __init__(self, kind):
            self.kind = kind

        def values(self, g, h, *, config, context):
            if self.kind == "mutation":
                g[0] += 1
            if self.kind == "host":
                return np.zeros(len(g), np.float32)
            if self.kind == "dtype":
                return cp.zeros(len(g), cp.float64)
            if self.kind == "nan":
                return cp.full(len(g), cp.nan, cp.float32)
            return cp.ones(len(g), cp.float32)

    device = tuple(cp.asarray(a) for a in reference.example())
    for kind in ("mutation", "host", "dtype", "nan", "inactive"):
        with pytest.raises((TypeError, ValueError)):
            leaf_values(*device, leaf_rule=Bad(kind))
    checks["batch_leaves"] = {
        "device_arrays": True,
        "row_sum_oracle": True,
        "bounded_changes_next_gradient": True,
        "cases": records,
        "two_rounds": rounds,
        "scope": "GPU primitive composition; CPU trainer tested separately; not assembled GPU Booster",
    }
