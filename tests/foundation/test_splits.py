"""Real-device split/routing with exhaustive row-mask reference."""

import hashlib

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_batch_split_routing_oracle(checks, monkeypatch):
    import cupy as cp
    import split_oracle as reference
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    from openboost._core import _primitives as legacy
    from openboost.experimental import build_histograms, find_splits, partition

    def forbidden(*args, **kwargs):
        raise AssertionError("Full-array host download or legacy wrapper used")

    records = []
    configs = [
        {},
        {"reg_lambda": 0.0, "min_child_weight": 0.0},
        {"min_gain": 1e6},
        {"min_child_weight": 20.0},
    ]
    for params in configs:
        host = reference.example()
        host[3][-1] = -1
        bins, g, h, ids, active = host
        expected = reference.exhaustive(*host, **params)
        device = [cp.asarray(a) for a in host]
        with monkeypatch.context() as m:
            m.setattr(cp, "asnumpy", forbidden)
            m.setattr(DeviceNDArray, "copy_to_host", forbidden)
            m.setattr(legacy, "build_node_histograms", forbidden)
            hist = build_histograms(*device)
            splits = find_splits(hist, **params)
            routed = partition(device[0], device[3], splits)
            assert isinstance(routed, cp.ndarray)
            assert all(
                isinstance(getattr(splits, k), cp.ndarray)
                for k in ("feature", "threshold", "left_child", "right_child", "gain", "valid")
            )
        np.testing.assert_array_equal(cp.asnumpy(splits.feature), [x[0] for x in expected])
        np.testing.assert_array_equal(cp.asnumpy(splits.threshold), [x[1] for x in expected])
        np.testing.assert_allclose(
            cp.asnumpy(splits.gain), [x[2] for x in expected], atol=1e-10, rtol=1e-10
        )
        wanted_ids = ids.copy()
        for i, node in enumerate(ids):
            if node >= 0 and expected[node][0] >= 0:
                f, threshold, _ = expected[node]
                wanted_ids[i] = 2 * node + (1 if bins[f, i] <= threshold else 2)
        np.testing.assert_array_equal(cp.asnumpy(routed), wanted_ids)
        np.testing.assert_array_equal(cp.asnumpy(device[3]), ids)
        child_active = cp.zeros_like(device[4])
        child_active[splits.left_child[splits.valid]] = True
        child_active[splits.right_child[splits.valid]] = True
        with monkeypatch.context() as m:
            m.setattr(cp, "asnumpy", forbidden)
            m.setattr(DeviceNDArray, "copy_to_host", forbidden)
            child = build_histograms(device[0], device[1], device[2], routed, child_active)
            second = find_splits(child, **params)
        active_host = cp.asnumpy(child_active)
        for node in np.flatnonzero(active_host):
            np.testing.assert_allclose(
                cp.asnumpy(child.grad)[node, 0].sum(), g[wanted_ids == node].sum(), atol=1e-6
            )
            np.testing.assert_allclose(
                cp.asnumpy(child.hess)[node, 0].sum(), h[wanted_ids == node].sum(), atol=1e-6
            )
        expected_second = reference.exhaustive(bins, g, h, wanted_ids, active_host, **params)
        np.testing.assert_array_equal(cp.asnumpy(second.feature), [x[0] for x in expected_second])
        np.testing.assert_array_equal(cp.asnumpy(second.threshold), [x[1] for x in expected_second])
        records.append(
            {
                "parameters": params,
                "input_sha256": hashlib.sha256(b"".join(a.tobytes() for a in host)).hexdigest(),
                "feature": cp.asnumpy(splits.feature).tolist(),
                "threshold": cp.asnumpy(splits.threshold).tolist(),
                "gain": cp.asnumpy(splits.gain).tolist(),
                "routed_ids": wanted_ids.tolist(),
            }
        )
    # Exact feature/threshold ties and min_gain equality are separate from random rounding.
    b = cp.asarray([[0, 0, 2, 2], [0, 0, 2, 2]], dtype=cp.uint8)
    hist = build_histograms(
        b,
        cp.asarray([2, 2, -2, -2], dtype=cp.float32),
        cp.ones(4, cp.float32),
        cp.zeros(4, cp.int32),
        cp.asarray([True, False, False]),
    )
    tie = find_splits(hist, reg_lambda=0.0, min_gain=16.0, min_child_weight=2.0)
    assert int(tie.feature[0]) == 0 and int(tie.threshold[0]) == 0 and float(tie.gain[0]) == 16.0
    assert not bool(
        find_splits(hist, reg_lambda=0.0, min_gain=np.nextafter(16.0, np.inf)).valid.any()
    )
    for kind in ("constant", "zero", "terminal", "inactive"):
        host = list(reference.example())
        if kind == "constant":
            host[0][:] = 2
        if kind == "zero":
            host[2][:] = 0
        if kind == "terminal":
            host[3][:] = 6
            host[4][:] = False
            host[4][6] = True
        if kind == "inactive":
            host[4][:] = False
        device = [cp.asarray(a) for a in host]
        result = find_splits(build_histograms(*device), min_child_weight=0.0)
        assert not bool(result.valid.any())
        np.testing.assert_array_equal(cp.asnumpy(partition(device[0], device[3], result)), host[3])
    with pytest.raises(ValueError):
        partition(b, cp.full(4, 99, cp.int32), tie)
    tie.left_child[0] = 2
    with pytest.raises(ValueError):
        partition(b, cp.zeros(4, cp.int32), tie)
    tie.left_child[0] = 1
    empty = build_histograms(
        b[:, :0].copy(),
        cp.zeros(0, cp.float32),
        cp.zeros(0, cp.float32),
        cp.zeros(0, cp.int32),
        cp.asarray([True, False, False]),
    )
    assert partition(b[:, :0].copy(), cp.zeros(0, cp.int32), find_splits(empty)).size == 0
    negative = build_histograms(
        b,
        cp.ones(4, cp.float32),
        cp.ones(4, cp.float32),
        cp.zeros(4, cp.int32),
        cp.asarray([True, False, False]),
    )
    assert not bool(find_splits(negative).valid.any())
    b[0, 0] = 255
    with pytest.raises(ValueError, match="missing"):
        partition(b, cp.zeros(4, cp.int32), tie)
    with pytest.raises(ValueError, match="missing"):
        find_splits(
            build_histograms(
                b,
                cp.ones(4, cp.float32),
                cp.ones(4, cp.float32),
                cp.zeros(4, cp.int32),
                cp.asarray([True, False, False]),
            )
        )
    checks["batch_splits"] = {
        "device_arrays": True,
        "routed_child_oracle": True,
        "cases": records,
        "exact_ties_and_gain_boundary": True,
        "scope": "numeric L2 positive-curvature children; no full-array named downloads; no whole-trainer claim",
    }
