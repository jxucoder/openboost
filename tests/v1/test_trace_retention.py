"""Trace retention must not change algorithm or stopping semantics."""

from dataclasses import fields

import numpy as np
import pytest

from openboost import NumericData, Problem, RunContext
from openboost.recipes import normal, squared


@pytest.mark.parametrize("recipe,width", [(squared, 1), (normal, 2)])
def test_summary_retains_no_training_arrays_and_matches_full(recipe, width):
    x = np.arange(96).reshape(32, 3) / 32
    data = NumericData(x, np.arange(32), ("a", "b", "c"))
    p = Problem(data, (1 + np.sin(x[:, 0]))[:, None], data.row_ids, raw_width=width)
    options = dict(context=RunContext("retention", 0), rounds=8, bins=8, step="fixed")
    full = recipe(p, p, **options)
    arrays = {
        id(a): a
        for step in full.steps
        for f in fields(step)
        if isinstance(a := getattr(step, f.name), np.ndarray)
    }
    assert sum(a.nbytes for a in arrays.values()) > 8 * 32 * width * 8
    summary = recipe(p, p, retention="summary", **options)
    assert summary.state.identity == full.state.identity
    assert summary.stop == full.stop
    assert len(summary.steps) == len(full.steps)
    np.testing.assert_array_equal(summary.state.train_raw, full.state.train_raw)
    for step, detailed in zip(summary.steps, full.steps, strict=True):
        assert not any(isinstance(getattr(step, f.name), np.ndarray) for f in fields(step))
        values = dict(step.values)
        assert values["accepted"] == detailed.accepted
        assert values["coefficients"] == detailed.coefficients
        assert values["loss_after"] == detailed.loss_after


def application(name):
    from dataclasses import replace

    from openboost import ClassSchema
    from tests.v1.test_public_stopping import problem

    p = problem()
    if name in ("normal", "formula"):
        p = replace(
            p,
            raw_width=2,
            offset=np.zeros((6, 2)),
            structure={"x": np.arange(1, 7)[:, None]} if name == "formula" else {},
        )
    elif name in ("binary", "multiclass"):
        k = 2 if name == "binary" else 3
        p = replace(
            p,
            target=(np.arange(6) % k)[:, None],
            classes=ClassSchema(tuple(range(k))),
            raw_width=1 if k == 2 else k,
            offset=np.zeros((6, 1 if k == 2 else k)),
        )
    elif name == "ranking":
        p = replace(
            p,
            target=(np.arange(6) % 3)[:, None],
            structure={"query": [[0], [0], [0], [1], [1], [1]]},
        )
    elif name == "poisson":
        p = replace(p, structure={"exposure": np.ones((6, 1))})
    elif name == "aft":
        p = replace(p, target=np.column_stack((p.target, p.target)), target_kind="event_right")
    elif name == "multi_squared":
        p = replace(
            p,
            target=np.column_stack((p.target, p.target * 2)),
            raw_width=2,
            offset=np.zeros((6, 2)),
        )
    return p


@pytest.mark.parametrize(
    "name",
    [
        "squared",
        "normal",
        "formula",
        "binary",
        "multiclass",
        "ranking",
        "quantile",
        "poisson",
        "gamma",
        "tweedie",
        "aft",
        "multi_squared",
    ],
)
@pytest.mark.parametrize("mode", ["fit", "stopped", "zero"])
def test_all_recipes_retention_preserves_state_stop_and_scalar_diagnostics(name, mode):
    from openboost import recipes
    from openboost.results import validate_result

    p = application(name)
    options = dict(
        context=RunContext(name, 0),
        rounds=0 if mode == "zero" else 3,
        bins=4,
        learning_rate=0 if mode == "stopped" else 0.1,
        patience=1 if mode == "stopped" else None,
    )
    recipe = getattr(recipes, name)
    full = recipe(p, p, **options)
    summary = recipe(p, p, retention="summary", **options)
    assert summary.state.identity == full.state.identity
    assert summary.stop == full.stop
    validate_result(summary, context=options["context"], train=p, validation=p)
    np.testing.assert_array_equal(summary.state.model.predict(p.data), full.state.train_raw)
    for a, b in zip(summary.steps, full.steps, strict=True):
        for field, value in a.values:
            expected = getattr(b, field)
            if isinstance(expected, np.ndarray):
                assert tuple(expected) == value  # per-output MSE, not sample arrays
            else:
                assert expected == value
    with pytest.raises(ValueError, match="retention"):
        recipe(p, p, retention="typo", **options)


def test_summary_rejects_hidden_array_or_state_payloads():
    from openboost.diagnostics import TraceSummary

    for value in (np.ones(4), (np.ones(1),), {"raw": np.ones(2)}):
        with pytest.raises(ValueError, match="scalars"):
            TraceSummary("external", (("value", value),))
    with pytest.raises(ValueError, match="unique"):
        TraceSummary("external", (("value", 1),), ("value",))


@pytest.mark.parametrize("family", ["normal", "formula"])
def test_ordered_summary_keeps_outer_rounds_without_retaining_states(family):
    from openboost.diagnostics import TraceSummary
    from openboost.results import validate_result
    from tests.v1.test_public_ordered import extension, problem

    p = problem(family)
    recipe = getattr(extension(), family)
    options = dict(context=RunContext("ordered", 0), rounds=3, bins=4)
    full = recipe(p, p, **options)
    summary = recipe(p, p, retention="summary", **options)
    assert summary.state.identity == full.state.identity and summary.stop == full.stop
    validate_result(summary, context=options["context"], train=p, validation=p)
    for outer, detailed in zip(summary.steps, full.steps, strict=True):
        assert len(outer) == 2
        for item, old in zip(outer, detailed, strict=True):
            assert isinstance(item, TraceSummary)
            assert dict(item.values)["after_version"] == old.after.version
            assert dict(item.values)["loss_after"] == old.loss_after


def test_backtracking_rejection_summary():
    data = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
    p = Problem(data, [[-3], [-3], [3], [3]], data.row_ids)
    options = dict(
        context=RunContext("reject", 0), rounds=3, learning_rate=1024, step="backtracking", bins=4
    )
    full = squared(p, p, **options)
    summary = squared(p, p, retention="summary", **options)
    assert full.state.version == summary.state.version == 0
    assert summary.stop == full.stop
    assert all(not dict(s.values)["accepted"] for s in summary.steps)
