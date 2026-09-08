"""CPU structural checks of objective comparison callbacks; no CUDA execution."""

from dataclasses import replace

import pytest

from openboost import LossChange
from openboost import device_normal as normal
from openboost.device_objectives import SQUARED
from openboost.device_runtime import DeviceRun


def test_missing_operation_is_explicit_and_cannot_read_reporting_loss():
    def forbidden(*args):
        raise AssertionError("must not subtract reporting losses")

    objective = replace(SQUARED, loss=forbidden)
    with pytest.raises(NotImplementedError, match="loss-change"):
        objective.loss_change(None, None, None, None)


def test_external_comparison_callback_receives_exact_borrowed_arguments():
    values = tuple(object() for _ in range(4))
    expected = LossChange(-0.5, -0.25, "external-objective", "declared-bound")

    def operation(*args):
        assert all(a is b for a, b in zip(args, values, strict=True))
        return expected

    objective = replace(SQUARED, compare=operation)
    assert objective.loss_change(*values) is expected
    assert normal.objective().compare is normal.compare


def test_bad_comparison_dependency_or_result_is_a_structural_error():
    with pytest.raises(ValueError, match="callable"):
        replace(SQUARED, compare="total-loss-subtraction")
    with pytest.raises(TypeError, match="LossChange"):
        replace(SQUARED, compare=lambda *args: -1.0).loss_change(None, None, None, None)


@pytest.mark.parametrize("policy", [None, True, "typo", [], {}])
def test_invalid_run_comparison_policy_fails_before_preparation(policy):
    def forbidden(*args):
        raise AssertionError("policy validation must precede preparation")

    with pytest.raises(ValueError, match="comparison"):
        DeviceRun(
            None,
            None,
            None,
            run_id="invalid",
            seed=1,
            objective=replace(SQUARED, validate=forbidden, prepare=forbidden),
            comparison=policy,
        )


def test_missing_run_operation_fails_before_preparation_without_fallback():
    def forbidden(*args):
        raise AssertionError("missing comparison must precede preparation")

    with pytest.raises(NotImplementedError, match="loss-change"):
        DeviceRun(
            None,
            None,
            None,
            run_id="missing",
            seed=1,
            objective=replace(SQUARED, validate=forbidden, prepare=forbidden),
            comparison="objective",
        )


def test_normal_recipe_explicitly_requires_objective_comparison(monkeypatch):
    from openboost import device_recipes

    def forbidden(*args):
        raise AssertionError("Normal must select objective policy before preparation")

    configured = replace(normal.objective(), compare=None, validate=forbidden)
    monkeypatch.setattr(normal, "objective", lambda **kwargs: configured)
    with pytest.raises(NotImplementedError, match="loss-change"):
        device_recipes.normal(None, None, None, run_id="required", seed=7, rounds=0)
