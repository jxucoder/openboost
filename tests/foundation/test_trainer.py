"""Actual strict experimental GPU fit, independent of legacy native dispatch."""

import hashlib
import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_strict_extension_trainer(checks, monkeypatch, tmp_path):
    import cupy as cp
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from scipy.special import ndtr

    import openboost as ob
    import openboost._trainer as trainer
    from openboost.experimental import (
        Booster,
        BuiltTree,
        DistributionObjectiveAdapter,
        ExecutionContext,
        LevelWiseBuilder,
        TrainerConfig,
    )

    class Decay:
        def coefficients(self, round_idx, channel_names, base_learning_rate):
            return {
                k: base_learning_rate / (round_idx + 1) / (i + 1)
                for i, k in enumerate(channel_names)
            }

    rng = np.random.default_rng(137)
    X = rng.normal(size=(257, 3)).astype(np.float32)
    y = (1.2 * X[:, 0] + 0.4 * rng.normal(size=257)).astype(np.float32)
    weights = rng.choice(np.array([0, 0.5, 1, 2], np.float32), 257)
    config = TrainerConfig(n_trees=2, max_depth=2, n_bins=32, learning_rate=0.2, random_state=7)
    objective = DistributionObjectiveAdapter("normal", natural=True)
    with ob.backend_context("cpu"):
        cpu = Booster(
            objective=objective,
            tree_builder=LevelWiseBuilder(),
            step_schedule=Decay(),
            config=config,
        ).fit(X, y, sample_weight=weights)
    expected = cpu.predict_raw(X)
    transfers, calls = [], []
    copy_to_host = cp.asnumpy
    original_build = LevelWiseBuilder.build

    def tracked(self, binned, grad, hess, *, config, context):
        assert context.device == "cuda" and context.xp is cp
        assert all(isinstance(a, cp.ndarray) for a in (binned.data, grad, hess))
        calls.append((context.round_idx, context.channel))
        return original_build(self, binned, grad, hess, config=config, context=context)

    def compact_only(a, *args, **kwargs):
        assert a.shape == (7,) and a.dtype in (cp.int32, cp.float32)
        transfers.append(a.nbytes)
        return copy_to_host(a, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy dispatch or host sample download during strict fit")

    class ExternalBuilder:
        supported_devices = frozenset({"cpu", "cuda"})

        def build(self, *args, **kwargs):
            return LevelWiseBuilder().build(*args, **kwargs)

    records = []
    for explicit in (False, True):
        model = Booster(
            objective=objective,
            tree_builder=ExternalBuilder() if explicit else None,
            step_schedule=Decay(),
            config=config,
            device="cuda",
        )
        with monkeypatch.context() as m:
            m.setattr(LevelWiseBuilder, "build", tracked)
            m.setattr(cp, "asnumpy", compact_only)
            m.setattr(DeviceNDArray, "copy_to_host", forbidden)
            m.setattr(trainer, "_to_host", forbidden)
            m.setattr(trainer, "fit_tree_gpu_native", forbidden)
            model.fit(X, y, sample_weight=weights)
        assert model.fit_report_["actual_device"] == "cuda"
        assert model.fit_report_["objective_device"] == model.fit_report_["update_device"] == "cuda"
        assert model.fit_report_["fallback_reason"] is None
        assert model.coefficients_ == {"loc": [0.2, 0.1], "scale": [0.1, 0.05]}
        actual = model.predict_raw(X)
        for k in expected:
            np.testing.assert_allclose(actual[k], expected[k], atol=2e-5, rtol=2e-5)
        path = tmp_path / f"model-{explicit}.ob"
        model.save(path)
        with pytest.warns(UserWarning, match="trusted"):
            loaded = Booster.load(path)
        for k in actual:
            np.testing.assert_array_equal(loaded.predict_raw(X)[k], actual[k])
        ctx = ExecutionContext("cpu", np, np.random.default_rng(7), 2)
        nll_cpu = objective.loss_value(expected, y, weights, context=ctx)
        nll_gpu = objective.loss_value(actual, y, weights, context=ctx)
        np.testing.assert_allclose(nll_gpu, nll_cpu, atol=2e-5, rtol=2e-5)

        def crps(raw):
            params = objective.constrain(raw)
            z = (y - params["loc"]) / params["scale"]
            scores = params["scale"] * (
                z * (2 * ndtr(z) - 1)
                + 2 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
                - 1 / np.sqrt(np.pi)
            )
            return float(np.average(scores, weights=weights))

        crps_cpu, crps_gpu = crps(expected), crps(actual)
        np.testing.assert_allclose(crps_gpu, crps_cpu, atol=2e-5, rtol=2e-5)
        records.append(
            {
                "explicit_builder": explicit,
                "fit_report": model.fit_report_,
                "max_raw_error": max(
                    float(np.max(np.abs(actual[k] - expected[k]))) for k in actual
                ),
                "cpu_crps": crps_cpu,
                "cuda_crps": crps_gpu,
                "cpu_nll": nll_cpu,
                "cuda_nll": nll_gpu,
            }
        )
    assert calls == [(r, k) for _ in range(2) for r in range(2) for k in ("loc", "scale")]
    assert len(transfers) == 40

    class Broken(DistributionObjectiveAdapter):
        def step(self, *args, **kwargs):
            raise RuntimeError("broken GPU kernel")

    class WrongDevice(DistributionObjectiveAdapter):
        def step(self, raw, y, *args, **kwargs):
            return {k: (np.zeros(len(y), np.float32), np.ones(len(y), np.float32)) for k in raw}

    class Mutating(DistributionObjectiveAdapter):
        def step(self, raw, y, *args, **kwargs):
            raw["loc"][:] = 123
            return super().step(raw, y, *args, **kwargs)

    before = model.predict_raw(X)
    report = model.fit_report_.copy()
    for bad, error, match in [
        (Broken("normal"), RuntimeError, "broken"),
        (WrongDevice("normal"), TypeError, "CuPy"),
        (Mutating("normal"), ValueError, "mutated"),
    ]:
        model.objective = bad
        with pytest.raises(error, match=match):
            model.fit(X, y, sample_weight=weights)
        assert model.fit_report_ == report
        for k in before:
            np.testing.assert_array_equal(model.predict_raw(X)[k], before[k])
    model.objective = objective

    class BadCache(LevelWiseBuilder):
        def build(self, *args, **kwargs):
            built = super().build(*args, **kwargs)
            return BuiltTree(built.tree, cp.full_like(built.train_prediction, 123))

    model.tree_builder, model._default_builder = BadCache(), False
    with pytest.raises(ValueError, match="prediction"):
        model.fit(X, y, sample_weight=weights)
    np.testing.assert_array_equal(model.predict_raw(X)["loc"], before["loc"])
    # The adapter advertises Normal/Poisson, natural and ordinary gradients.
    # Verify the actual shared trainer across that whole declared surface.
    adapters = []
    for distribution in ("normal", "poisson"):
        target = y if distribution == "normal" else rng.poisson(2, len(y)).astype(np.float32)
        for natural in (False, True):
            obj = DistributionObjectiveAdapter(distribution, natural=natural)
            models = [
                Booster(
                    objective=obj, tree_builder=LevelWiseBuilder(), config=config, device=device
                ).fit(X, target, sample_weight=weights)
                for device in ("cpu", "cuda")
            ]
            outputs = [m.predict_raw(X) for m in models]
            for channel in outputs[0]:
                np.testing.assert_allclose(
                    outputs[1][channel], outputs[0][channel], atol=2e-5, rtol=2e-5
                )
            adapters.append(
                {
                    "distribution": distribution,
                    "natural": natural,
                    "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                    "max_raw_error": max(
                        float(np.max(np.abs(outputs[1][k] - outputs[0][k]))) for k in outputs[0]
                    ),
                }
            )

    class InvalidStats(DistributionObjectiveAdapter):
        def __init__(self, fault):
            super().__init__("normal", natural=True)
            self.fault = fault

        def step(self, raw, y, *args, **kwargs):
            out = super().step(raw, y, *args, **kwargs)
            g, h = out["loc"]
            if self.fault == "dtype":
                out["loc"] = (g.astype(cp.float64), h)
            elif self.fault == "negative":
                out["loc"] = (g, -cp.ones_like(h))
            elif self.fault == "alias":
                out["loc"] = (raw["loc"], h)
            return out

    for fault, match in [("dtype", "float32"), ("negative", "nonnegative"), ("alias", "alias")]:
        with pytest.raises(ValueError, match=match):
            Booster(objective=InvalidStats(fault), device="cuda", config=config).fit(X, y)

    checks["strict_extension_trainer"] = {
        "adapter_cases": adapters,
        "additional_invalid_statistics": 3,
        "actual_fit": True,
        "legacy_dispatch_blocked": True,
        "rollback": True,
        "cpu_load_prediction": True,
        "cases": records,
        "compact_transfer_calls": len(transfers),
        "compact_transfer_bytes": sum(transfers),
        "data_sha256": hashlib.sha256(X.tobytes() + y.tobytes() + weights.tobytes()).hexdigest(),
        "samples": 257,
        "seed": 137,
        "nsys_available": shutil.which("nsys") is not None,
        "profiler_trace_collected": False,
        "scope": "named transfer wrappers only; scalar syncs and device defensive copies allowed",
    }
