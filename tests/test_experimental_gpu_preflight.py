"""Strict GPU capability errors must precede device imports and objective work."""

import numpy as np
import pytest

from openboost.experimental import Booster, LevelWiseBuilder, TrainerConfig
from tests.test_experimental_objective import TwoSquared


class DeclaredGPU(TwoSquared):
    supported_devices = frozenset({"cpu", "cuda"})


@pytest.mark.parametrize(
    "kind", ["missing", "sampling", "l1", "eval", "callback", "early_stop", "budget", "categorical"]
)
def test_gpu_preflight_without_cuda(kind):
    X, y = np.zeros((4, 1), np.float32), np.ones(4, np.float32)
    config, kwargs = TrainerConfig(n_trees=1), {}
    if kind == "missing":
        X[0, 0] = np.nan
    if kind == "categorical":
        import openboost as ob

        X = ob.array(X, categorical_features=[0])
    if kind == "sampling":
        config.subsample = 0.5
    if kind == "l1":
        config.reg_alpha = 1
    if kind == "eval":
        kwargs["eval_sets"] = [{"X": X, "y": y}]
    if kind == "callback":
        kwargs["callbacks"] = [object()]
    if kind == "early_stop":
        kwargs["early_stopping_rounds"] = 1
    with pytest.raises(ValueError, match="Strict CUDA"):
        Booster(
            objective=DeclaredGPU(),
            tree_builder=LevelWiseBuilder(
                memory_budget_bytes=0 if kind == "budget" else 256 * 1024**2
            ),
            config=config,
            device="cuda",
        ).fit(X, y, **kwargs)


def test_unsupported_objective_full_cpu_fallback():
    with pytest.warns(RuntimeWarning, match="CPU fallback"):
        model = Booster(
            objective=TwoSquared(), device="cuda", fallback="warn", config=TrainerConfig(n_trees=1)
        ).fit(np.zeros((4, 1)), np.ones(4))
    assert model.fit_report_["actual_device"] == "cpu"
    assert model.fit_report_["fallback_reason"]
