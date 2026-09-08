"""CPU structural checks of objective comparison callbacks; no CUDA execution."""

from dataclasses import replace

import pytest

from openboost import LossChange
from openboost import device_normal as normal
from openboost.device_objectives import SQUARED


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
