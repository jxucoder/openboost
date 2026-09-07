"""Matched payment binding must not substitute raw claim counts or wrong rows."""

import numpy as np
import pytest
from benchmarks.v1.paid_event_data import bind


def fixture():
    raw = dict(
        group=np.array([10, 20, 30, 40]),
        exposure=np.array([0.5, 1.0, 2.0, 1.0]),
        paid_count=np.array([2, 0, 1, 1]),
        paid_total=np.array([8.0, 0, 3, 7]),
        aggregate_eligible=np.ones(4, dtype=bool),
        severity_policy_row=np.array([0, 2, 0, 3]),
        severity_y=np.array([2.0, 3, 6, 7]),
        y=np.array([9, 0, 2, 1]),
    )
    arrays = dict(
        x_train=np.array([[1.0], [0.0]]),
        y_train=np.array([0.0, 16.0]),
        weight_train=np.array([1.0, 0.5]),
        x_validation=np.array([[3.0], [2.0]]),
        y_validation=np.array([7.0, 1.5]),
        weight_validation=np.array([1.0, 2.0]),
        validation_row_ids=np.array([40, 30]),
    )
    return raw, arrays, np.array([20, 10])


def test_matched_payments_follow_frozen_order_not_raw_claim_counts():
    raw, arrays, ids = fixture()
    result = bind(raw, arrays, ids)
    np.testing.assert_array_equal(result["paid_count_train"], [0, 2])
    np.testing.assert_array_equal(result["paid_total_train"], [0, 8])
    np.testing.assert_array_equal(result["paid_count_validation"], [1, 1])
    assert not any("test" in k or "weight" in k for k in result)


@pytest.mark.parametrize(
    "bad",
    [
        "count",
        "total",
        "payment",
        "mapping",
        "eligibility",
        "weight",
        "target",
        "foreign",
        "overlap",
        "test",
    ],
)
def test_mismatched_binding_fails(bad):
    raw, arrays, ids = fixture()
    if bad == "count":
        raw["paid_count"] = raw["y"]
    elif bad == "total":
        raw["paid_total"][0] += 1
    elif bad == "payment":
        raw["severity_y"][0] = 0
    elif bad == "mapping":
        raw["severity_policy_row"][0] = 10
    elif bad == "eligibility":
        raw["aggregate_eligible"][0] = False
    elif bad == "weight":
        arrays["weight_train"][0] = 0.5
    elif bad == "target":
        arrays["y_train"][0] = 1
    elif bad == "foreign":
        ids[0] = 999
    elif bad == "overlap":
        ids[0] = 40
    else:
        arrays["y_test"] = np.ones(2)
    with pytest.raises(ValueError):
        bind(raw, arrays, ids)
