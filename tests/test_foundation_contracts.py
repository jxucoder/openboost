"""Host regressions for the unified trainer's execution policy and RNG."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import openboost as ob
import openboost._trainer as trainer
from openboost._distributions import Normal, Poisson
from openboost._objectives import DistributionObjective


def test_same_name_custom_distribution_is_not_device_capable():
    custom = type("Normal", (Normal,), {})()
    assert not DistributionObjective(custom).device_capable
    assert DistributionObjective(Normal()).device_capable
    assert DistributionObjective(Poisson()).device_capable


def test_kernel_error_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("kernel sentinel")

    monkeypatch.setitem(
        sys.modules,
        "openboost._backends._cuda",
        SimpleNamespace(normal_step_gpu=fail, poisson_step_gpu=fail, scale_gh_gpu=fail),
    )
    objective = DistributionObjective(Normal(), natural=True)
    with pytest.raises(RuntimeError, match="kernel sentinel"):
        objective._step_device({"loc": np.zeros(4), "scale": np.zeros(4)}, np.ones(4), None)


@pytest.mark.parametrize("parameter", ["subsample", "colsample_bytree"])
def test_gpu_sampling_rejected_before_binning(parameter, monkeypatch):
    monkeypatch.setattr(trainer, "is_cuda", lambda: True)
    model = ob.NaturalBoost(**{parameter: 0.5})
    with pytest.raises(ValueError, match="sampling"):
        model.fit(np.ones((8, 2)), np.ones(8))
    assert model.X_binned_ is None
    assert not model.trees_
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.ones((8, 2)))


def test_failed_first_step_leaves_unfitted_model(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("objective sentinel")

    monkeypatch.setattr(DistributionObjective, "step", fail)
    model = ob.NaturalBoost(n_trees=1)
    with pytest.raises(RuntimeError, match="objective sentinel"):
        model.fit(np.ones((8, 2)), np.ones(8))
    assert not model.trees_
    assert model.X_binned_ is None


@pytest.mark.parametrize("parameter", ["subsample", "colsample_bytree"])
def test_sampling_uses_scoped_model_seed(parameter):
    rng = np.random.default_rng(78)
    X = rng.normal(size=(128, 6)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] ** 2 + rng.normal(size=128)).astype(np.float32)
    state = np.random.get_state()
    predictions = []
    for seed in (12, 12, 34):
        model = ob.NaturalBoost(n_trees=3, max_depth=2, random_state=seed, **{parameter: 0.5})
        model.fit(X, y)
        predictions.append(model.predict(X))
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    np.testing.assert_array_equal(predictions[0], predictions[1])
    assert not np.array_equal(predictions[0], predictions[2])


def test_known_host_objective_fallback_is_visible(monkeypatch):
    class StopBeforeDevice(Exception):
        pass

    def stop(*args, **kwargs):
        raise StopBeforeDevice

    monkeypatch.setattr(trainer, "is_cuda", lambda: True)
    monkeypatch.setattr(trainer, "_bin_features", stop)
    custom = type("Normal", (Normal,), {})()
    with (
        pytest.warns(RuntimeWarning, match="objective fallback to CPU"),
        pytest.raises(StopBeforeDevice),
    ):
        ob.NaturalBoost(distribution=custom).fit(np.ones((8, 2)), np.ones(8))


@pytest.mark.parametrize(
    "weights,expected", [(None, 1.0), ([1.0] * 8, 0.0), ([0, 1, 2, 4, 0, 2, 3, 5], 0.0)]
)
def test_weighted_native_hint_selection(weights, expected, monkeypatch):
    """Host policy check only; real histogram/Newton parity is the Modal gate."""
    import numba

    from openboost._array import BinnedArray

    class StopAtNative(Exception):
        pass

    class FixedObjective:
        channel_names = ["value"]
        device_capable = False
        unit_hessian = True

        def init_raw(self, *args):
            return {"value": 0.0}

        def step(self, raw, y, sample_weight, extra):
            h = np.ones(8, dtype=np.float32) if sample_weight is None else sample_weight
            return {"value": (y * h, h)}

    captured = {}

    def native(*args, **kwargs):
        captured.update(kwargs)
        raise StopAtNative

    monkeypatch.setattr(trainer, "is_cuda", lambda: True)
    monkeypatch.setattr(numba, "cuda", SimpleNamespace(to_device=lambda a: a))
    monkeypatch.setattr(trainer, "fit_tree_gpu_native", native)
    binned = BinnedArray(np.zeros((1, 8), dtype=np.uint8), [], 1, 8, "cpu")
    with pytest.warns(RuntimeWarning, match="fallback"), pytest.raises(StopAtNative):
        trainer.fit_boosting(
            SimpleNamespace(),
            FixedObjective(),
            binned,
            np.ones(8),
            config=trainer.TrainerConfig(n_trees=1),
            sample_weight=weights,
        )
    assert captured["const_hess"] == expected


def test_seed_survives_persistence(tmp_path):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(64, 3)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=64)).astype(np.float32)
    model = ob.NaturalBoost(n_trees=2, max_depth=2, subsample=0.5, random_state=42).fit(X, y)
    path = tmp_path / "seeded.ob"
    model.save(path)
    restored = ob.load(path)
    assert restored.random_state == 42
    np.testing.assert_array_equal(model.predict(X), restored.predict(X))
    restored.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), restored.predict(X))
