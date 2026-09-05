"""Whole-tree GPU builder parity and two-channel distribution composition."""

import gc
import hashlib
from dataclasses import replace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_levelwise_builder_device_oracle(checks, monkeypatch, tmp_path):
    import builder_oracle as reference
    import cupy as cp
    import leaf_oracle
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from scipy.special import ndtr

    import openboost as ob
    from openboost.experimental import Booster, LevelWiseBuilder, TrainerConfig

    copy_to_host = cp.asnumpy
    transfers = []

    def build(builder, binned, g, h, cfg, ctx):
        slots = 2 ** (cfg.max_depth + 1) - 1

        def compact_only(a, *args, **kwargs):
            assert a.shape == (slots,) and a.dtype in (cp.int32, cp.float32)
            transfers.append(a.nbytes)
            return copy_to_host(a, *args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("Numba host download in device builder")

        with monkeypatch.context() as m:
            m.setattr(cp, "asnumpy", compact_only)
            m.setattr(DeviceNDArray, "copy_to_host", forbidden)
            result = builder.build(binned, g, h, config=cfg, context=ctx)
        assert isinstance(result.train_prediction, cp.ndarray)
        return result

    host, g, h = reference.example()
    cfg = TrainerConfig(max_depth=2)
    dg, dh = cp.asarray(g).view(), cp.asarray(h).view()
    device = replace(host, data=cp.asarray(host.data).view(), device="cuda")
    result = build(LevelWiseBuilder(), device, dg, dh, cfg, reference.context(cp))
    expected, prediction = reference.oracle_tree(host.data, g, h, cfg)
    for k, value in expected.items():
        np.testing.assert_allclose(getattr(result.tree, k), value, atol=1e-6, rtol=1e-6)
    del dg, dh, device
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    np.testing.assert_allclose(copy_to_host(result.train_prediction), prediction, atol=1e-6)
    with ob.backend_context("cpu"):
        np.testing.assert_array_equal(result.tree(host), copy_to_host(result.train_prediction))
    device = replace(host, data=cp.asarray(host.data), device="cuda")
    with cp.cuda.Stream(non_blocking=True), pytest.raises(ValueError, match="default CUDA stream"):
        LevelWiseBuilder().build(
            device, cp.asarray(g), cp.asarray(h), config=cfg, context=reference.context(cp)
        )
    for field, value in [("reg_alpha", 1.0), ("subsample", 0.5), ("colsample_bytree", 0.5)]:
        with pytest.raises(ValueError):
            LevelWiseBuilder().build(
                device,
                cp.asarray(g),
                cp.asarray(h),
                config=replace(cfg, **{field: value}),
                context=reference.context(cp),
            )
    with pytest.raises(MemoryError):
        LevelWiseBuilder(memory_budget_bytes=0).build(
            device, cp.asarray(g), cp.asarray(h), config=cfg, context=reference.context(cp)
        )
    device.data[0, 0] = 255
    with pytest.raises(ValueError, match="missing"):
        LevelWiseBuilder().build(
            device, cp.asarray(g), cp.asarray(h), config=cfg, context=reference.context(cp)
        )
    device.data[0, 0] = 0
    device.is_categorical = np.array([True, False])
    with pytest.raises(ValueError, match="categorical"):
        LevelWiseBuilder().build(
            device, cp.asarray(g), cp.asarray(h), config=cfg, context=reference.context(cp)
        )

    def metrics(raw, y):
        mu, logs = raw["mu"], raw["log_sigma"]
        sigma = np.exp(logs.astype(np.float64))
        z = (y - mu) / sigma
        nll = np.mean(logs + 0.5 * z * z + 0.5 * np.log(2 * np.pi))
        crps = np.mean(
            sigma
            * (
                z * (2 * ndtr(z) - 1)
                + 2 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
                - 1 / np.sqrt(np.pi)
            )
        )
        return {"nll": float(nll), "crps": float(crps)}

    records = []
    for samples in (16, 4097):
        rng = np.random.default_rng(127)
        X = rng.normal(size=(samples, 3)).astype(np.float32)
        y = (1.2 * X[:, 0] + 0.4 * rng.normal(size=samples)).astype(np.float32)
        weights = rng.choice(np.array([0, 0.5, 1, 2], np.float32), samples)
        with ob.backend_context("cpu"):
            cpu_bins = ob.array(X, n_bins=32)
        cfg = TrainerConfig(n_trees=2, max_depth=2, learning_rate=0.2, n_bins=32)
        default_output = None
        for clipped in (False, True):
            results = []
            for xp in (np, cp):
                rule = leaf_oracle.BoundedLeaf() if clipped else None
                builder = LevelWiseBuilder(leaf_rule=rule)
                bins = (
                    cpu_bins
                    if xp is np
                    else replace(cpu_bins, data=cp.asarray(cpu_bins.data), device="cuda")
                )
                target, w = xp.asarray(y), xp.asarray(weights)
                raw = {k: xp.zeros(samples, xp.float32) for k in ("mu", "log_sigma")}
                # Assemble persistence state explicitly; this is NOT Booster.fit on CUDA.
                model = Booster(objective=None)
                model.trees_ = {k: [] for k in raw}
                model.coefficients_ = {k: [] for k in raw}
                model._base_scores = dict.fromkeys(raw, 0.0)
                model.learning_rate, model.n_bins, model.X_binned_ = 0.2, 32, cpu_bins
                for round_idx in range(2):
                    residual = raw["mu"] - target
                    inv_var = xp.exp(-2 * raw["log_sigma"])
                    gradients = {
                        "mu": (residual * inv_var * w, inv_var * w),
                        "log_sigma": ((1 - residual * residual * inv_var) * w, 2 * w),
                    }
                    for channel in raw:
                        grad, hess = gradients[channel]
                        ctx = reference.context(xp, round_idx, channel)
                        tree = (
                            builder.build(bins, grad, hess, config=cfg, context=ctx)
                            if xp is np
                            else build(builder, bins, grad, hess, cfg, ctx)
                        )
                        coefficient = (0.2 if channel == "mu" else 0.1) / (round_idx + 1)
                        raw[channel] += coefficient * tree.train_prediction
                        model.trees_[channel].append(tree.tree)
                        model.coefficients_[channel].append(coefficient)
                output = {k: v.copy() if xp is np else copy_to_host(v) for k, v in raw.items()}
                path = tmp_path / f"model-{samples}-{clipped}-{xp.__name__}.ob"
                model.save(path)
                with pytest.warns(UserWarning, match="trusted"), ob.backend_context("cpu"):
                    restored = Booster.load(path)
                    inferred = restored.predict_raw(X)
                for channel in raw:
                    np.testing.assert_allclose(
                        inferred[channel], output[channel], atol=2e-6, rtol=2e-5
                    )
                results.append(output)
            for channel in results[0]:
                np.testing.assert_allclose(
                    results[1][channel], results[0][channel], atol=2e-5, rtol=2e-5
                )
            if not clipped:
                default_output = results[0]
            else:
                assert any(not np.allclose(results[0][k], default_output[k]) for k in results[0])
            cpu_metrics, gpu_metrics = metrics(results[0], y), metrics(results[1], y)
            for name in cpu_metrics:
                np.testing.assert_allclose(
                    gpu_metrics[name], cpu_metrics[name], atol=2e-5, rtol=2e-5
                )
            records.append(
                {
                    "samples": samples,
                    "seed": 127,
                    "clipped": clipped,
                    "data_sha256": hashlib.sha256(
                        X.tobytes() + y.tobytes() + weights.tobytes()
                    ).hexdigest(),
                    "cpu": cpu_metrics,
                    "cuda": gpu_metrics,
                    "max_raw_error": max(
                        float(np.max(np.abs(results[1][k] - results[0][k]))) for k in results[0]
                    ),
                }
            )
    assert len(transfers) == 5 * 17  # one oracle tree plus 4 cells * 2 rounds * 2 channels
    checks["levelwise_builder"] = {
        "device_cache_survives_owner_release": True,
        "compact_transfer_calls": len(transfers),
        "compact_transfer_bytes": sum(transfers),
        "two_channel_cases": records,
        "cpu_load_prediction": True,
        "scope": "direct GPU builder composition; GPU Booster.fit remains P5; named transfers only",
    }
