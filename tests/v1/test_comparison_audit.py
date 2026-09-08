"""The offline audit must reject numerically false or missing comparison evidence."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import LossChange

from .comparison_audit import numerical_audit
from .reference.normal_comparison import compare
from .test_loss_change import CASES


@pytest.mark.parametrize(
    "name",
    [
        "analytic/tiny-improvement",
        "analytic/scale-cancellation",
        "analytic/outside-exponent-range",
        "run7/forward/channel0/alpha4.0/training",
    ],
)
def test_independent_audit_handles_hidden_changes_and_explicit_resolution(name):
    case = next(c for c in CASES if c["id"] == name)
    a = {k: np.array(v["values"]) for k, v in case["inputs"].items()}
    args = tuple(a[k] for k in ("before", "after", "target", "offset", "weight"))
    expected = compare(*args)
    change = LossChange(
        expected.lower,
        expected.upper,
        "independent-test",
        expected.reason,
        expected.status == "unchanged",
    )
    result = numerical_audit(change, *args)
    assert result["reference"]["status"] == expected.status


@pytest.mark.parametrize("bad", ["wrong_sign", "missing_bounds", "wrong_interval", "unchanged"])
def test_audit_rejects_false_evidence_instead_of_copying_reported_status(bad):
    args = np.array([[1.0, 0]]), np.zeros((1, 2)), np.zeros(1), np.zeros((1, 2)), np.ones(1)
    change = LossChange(-0.5, -0.5, "independent-test", "bounded")
    numerical_audit(change, *args)
    broken = {
        "wrong_sign": replace(change, lower=0.5, upper=0.5),
        "missing_bounds": replace(change, lower=None, upper=None),
        "wrong_interval": replace(change, lower=-1, upper=-0.75),
        "unchanged": replace(change, lower=0, upper=0, unchanged=True),
    }[bad]
    with pytest.raises(AssertionError):
        numerical_audit(broken, *args)
