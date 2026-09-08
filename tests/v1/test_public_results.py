"""External result interoperability and fail-closed result validation."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from examples.v1_extensions.custom_stopping import squared_until_loss

from openboost import NumericData, Problem, RunContext
from openboost.binning import Binning, PreparedData
from openboost.recipes import squared
from openboost.runs import RunSpec, run_many
from openboost.stopping import StopState
from tests.v1.test_public_ordered import extension


def problems():
    x = NumericData(np.arange(6)[:, None], np.arange(6), ("x",))
    p = Problem(x, [[0.5], [1], [2], [3], [5], [8]], x.row_ids)
    return p, replace(p, raw_width=2, offset=np.zeros((6, 2)))


@dataclass(frozen=True)
class ExternalStop:
    rounds: int = 5
    completed_rounds: int = 2
    reason: str = "external_rule"
    diagnostics: tuple = ("author-owned",)


def test_structural_stopping_preserves_external_record():
    p, _ = problems()
    context = RunContext("external-stop", 7)
    fit = squared(p, p, context=context, rounds=2)
    stop = ExternalStop()
    result = SimpleNamespace(state=fit.state, steps=fit.steps, stop=stop)
    outcome = run_many([RunSpec(context, p, p, lambda *a, **kw: result)])[0]
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result is result
    assert outcome.result.stop is stop
    assert outcome.result.stop.diagnostics is stop.diagnostics


@pytest.mark.parametrize(
    "change",
    [
        {"rounds": True},
        {"rounds": 5.0},
        {"rounds": -1},
        {"completed_rounds": True},
        {"completed_rounds": 2.0},
        {"completed_rounds": -1},
        {"completed_rounds": 6},
        {"reason": None},
        {"reason": ""},
        {"reason": 1},
        {"reason": "budget"},
    ],
)
def test_malformed_external_completion_is_isolated(change):
    p, _ = problems()
    context = RunContext("invalid-external", 7)
    fit = squared(p, p, context=context, rounds=2)
    bad = SimpleNamespace(state=fit.state, steps=fit.steps, stop=replace(ExternalStop(), **change))
    outcomes = run_many(
        [
            RunSpec(context, p, p, lambda *a, **kw: bad),
            RunSpec(RunContext("valid-neighbor", 7), p, p, squared, {"rounds": 0}),
        ]
    )
    assert outcomes[0].error_type == "ValueError"
    assert outcomes[0].result is None
    assert outcomes[1].error_type is None
    assert outcomes[1].result.stop.reason == "budget"


@pytest.mark.parametrize("field", ["rounds", "completed_rounds", "reason"])
def test_missing_external_completion_field(field):
    p, _ = problems()
    context = RunContext("missing-stop-field", 7)
    fit = squared(p, p, context=context, rounds=2)
    parts = dict(rounds=5, completed_rounds=2, reason="external_rule")
    del parts[field]
    result = SimpleNamespace(state=fit.state, steps=fit.steps, stop=SimpleNamespace(**parts))
    out = run_many([RunSpec(context, p, p, lambda *a, **kw: result)])[0]
    assert out.error_type == "ValueError" and out.result is None


@pytest.mark.parametrize(
    "rounds,threshold,completed,reason",
    [
        (5, 0.05, 2, "loss_threshold"),
        (1, 0.05, 1, "budget"),
        (0, 0.05, 0, "budget"),
        (2, 0.05, 2, "loss_threshold"),
        (5, 0.2, 1, "loss_threshold"),
    ],
)
def test_real_external_loop_uses_measured_loss(rounds, threshold, completed, reason):
    x = NumericData([[-1], [1]], [0, 1], ("x",))
    p = Problem(x, [[-1], [1]], x.row_ids)
    # Opposite validation targets leave the initial model as validation best.
    valid = replace(p, target=-p.target)
    context = RunContext("threshold-loop", 7)
    out = run_many(
        [RunSpec(context, p, valid, squared_until_loss, dict(rounds=rounds, threshold=threshold))]
    )[0]
    assert out.error_type is None, out.error_message
    fit = out.result
    assert fit.stop.reason == reason
    assert fit.stop.completed_rounds == len(fit.steps) == completed
    assert fit.stop.losses is fit.steps
    # Independent recurrence: each pure leaf removes half the residual.
    np.testing.assert_array_equal(fit.steps, [0.5 * 0.25**i for i in range(1, completed + 1)])
    np.testing.assert_array_equal(fit.state.train_raw, p.target * (1 - 0.5**completed))
    assert fit.state.best_model.terms == ()
    assert fit.state.version == completed


def test_external_ordered_result_is_preserved():
    _, p = problems()
    recipe = extension().normal
    context = RunContext("external", 7)
    expected = recipe(p, p, context=context, rounds=2)
    outcome = run_many([RunSpec(context, p, p, lambda *args, **kwargs: expected)])[0]
    assert outcome.error_type is None
    assert outcome.result is expected
    assert outcome.result.state.version == 4
    assert outcome.result.stop.completed_rounds == len(outcome.result.steps) == 2


@pytest.mark.parametrize(
    "corruption", ["missing", "state", "stop", "unfinished", "trace", "list", "foreign"]
)
def test_invalid_result_fails_only_its_run(corruption):
    p, _ = problems()
    context = RunContext("bad", 7)
    fit = squared(p, p, context=context, rounds=1)
    parts = dict(state=fit.state, steps=fit.steps, stop=fit.stop)
    if corruption == "missing":
        del parts["stop"]
    elif corruption == "state":
        parts["state"] = object()
    elif corruption == "stop":
        parts["stop"] = object()
    elif corruption == "unfinished":
        parts["stop"] = StopState.start(1, rounds=2).observe(0.5)
    elif corruption == "trace":
        parts["steps"] = ()
    elif corruption == "list":
        parts["steps"] = list(fit.steps)
    else:
        parts["state"] = replace(fit.state, context=RunContext("foreign", 7))
    bad = SimpleNamespace(**parts)
    outcomes = run_many(
        [
            RunSpec(context, p, p, lambda *a, **kw: bad),
            RunSpec(RunContext("good", 7), p, p, squared, {"rounds": 1}),
        ]
    )
    assert outcomes[0].error_type == "ValueError" and outcomes[0].result is None
    assert outcomes[1].error_type is None and outcomes[1].result.stop.reason == "budget"


@pytest.mark.parametrize("count", [1, 8, 32])
def test_external_and_builtin_shared_runs(count, monkeypatch):
    p, normal = problems()
    prepared = PreparedData(p.data, bins=6)
    external = extension().normal
    specs = []
    expected = {}
    for i in range(count):
        train, recipe = (p, squared) if i % 2 else (normal, external)
        valid = replace(train, target=train.target[::-1])
        context = RunContext(f"mixed-{i}", 7)
        options = {"rounds": 6, "bins": 6, "patience": 1 + i % 3, "min_delta": 100.0}
        specs.append(RunSpec(context, train, valid, recipe, options, prepared))
        expected[context.run_id] = recipe(train, valid, context=context, **options)
    if count > 1:
        assert len({r.stop.completed_rounds for r in expected.values()}) > 1

    def forbidden(*a, **kw):
        raise AssertionError("shared preparation must not refit")

    monkeypatch.setattr(Binning, "fit", forbidden)
    failed = replace(specs[0], context=RunContext("failure", 7), recipe=lambda *a, **kw: object())
    executions = [
        run_many([failed, *specs]),
        run_many(reversed(specs)),
        tuple(r for group in (specs[::2], specs[1::2]) for r in run_many(group)),
        run_many(specs),
    ]
    assert executions[0][0].error_type == "ValueError"
    for outcomes in executions:
        for out in outcomes:
            if out.run_id == "failure":
                continue
            assert out.error_type is None
            ref = expected[out.run_id]
            assert out.result.state.identity == ref.state.identity
            assert out.result.stop == ref.stop
            np.testing.assert_array_equal(out.result.state.validation_raw, ref.state.validation_raw)
            np.testing.assert_array_equal(
                out.result.state.context.rng(1, "leaf", "sample").random(5),
                ref.state.context.rng(1, "leaf", "sample").random(5),
            )
