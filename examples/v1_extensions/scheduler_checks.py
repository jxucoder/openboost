"""Installed D5 structural-result and independent scheduling development checks."""

import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ob_ordered_updates as ordered

import openboost
from openboost import NumericData, Problem, RunContext
from openboost.binning import Binning, PreparedData
from openboost.recipes import squared
from openboost.runs import RunSpec, run_many

assert "site-packages" in ordered.__file__
assert "site-packages" in openboost.__file__
custom = runpy.run_path(str(Path(__file__).with_name("custom_stopping.py")))
threshold_recipe = custom["squared_until_loss"]
root = Path(sys.argv[1])
x = NumericData(np.arange(6)[:, None], np.arange(6), ("x",))
p = Problem(x, [[0.5], [1], [2], [3], [5], [8]], x.row_ids)
distribution = replace(p, raw_width=2, offset=np.zeros((6, 2)))
prepared = PreparedData(x, bins=6)
reports = []


def compare(outcome, reference):
    assert outcome.error_type is None, outcome.error_message
    fit = outcome.result
    assert fit.state.identity == reference.state.identity
    assert fit.stop == reference.stop
    np.testing.assert_array_equal(fit.state.train_raw, reference.state.train_raw)
    np.testing.assert_array_equal(fit.state.validation_raw, reference.state.validation_raw)
    np.testing.assert_array_equal(
        fit.state.context.rng(1, "leaf", "sample").random(5),
        reference.state.context.rng(1, "leaf", "sample").random(5),
    )
    if fit.stop.reason == "loss_threshold":
        assert fit.stop.losses is fit.steps
        assert fit.stop.threshold == reference.stop.threshold


def summary(fit):
    return dict(
        state=fit.state.identity,
        stop_round=fit.stop.completed_rounds,
        reason=fit.stop.reason,
        version=fit.state.version,
        predictions=fit.state.validation_raw.tolist(),
        rng=fit.state.context.rng(1, "leaf", "sample").random(5).tolist(),
    )


for count in (1, 8, 32):
    specs, expected = [], {}
    for i in range(count):
        train, recipe = (p, squared) if i % 2 else (distribution, ordered.normal)
        valid = replace(train, target=train.target[::-1])
        context = RunContext(f"mixed-{i}", 7)
        options = dict(rounds=6, bins=6, patience=1 + i % 3, min_delta=100.0)
        specs.append(RunSpec(context, train, valid, recipe, options, prepared))
        expected[context.run_id] = recipe(train, valid, context=context, **options)

    def forbidden(*args, **kwargs):
        raise AssertionError("shared execution refitted training preparation")

    original_fit = Binning.fit
    Binning.fit = forbidden
    try:
        failed = replace(
            specs[0], context=RunContext("bad-result", 7), recipe=lambda *a, **kw: object()
        )
        groups = [
            run_many([failed, *specs]),
            run_many(reversed(specs)),
            run_many(specs),
            tuple(r for batch in (specs[::2], specs[1::2]) for r in run_many(batch)),
        ]
        assert groups[0][0].error_type == "ValueError" and groups[0][0].result is None
        for outcomes in groups:
            for outcome in outcomes:
                if outcome.run_id == "bad-result":
                    continue
                assert outcome.error_type is None
                ref = expected[outcome.run_id]
                assert outcome.result.state.identity == ref.state.identity
                assert outcome.result.stop == ref.stop
                np.testing.assert_array_equal(
                    outcome.result.state.validation_raw, ref.state.validation_raw
                )
                np.testing.assert_array_equal(
                    outcome.result.state.context.rng(1, "leaf", "sample").random(5),
                    ref.state.context.rng(1, "leaf", "sample").random(5),
                )
    finally:
        Binning.fit = original_fit
    report = dict(
        count=count,
        failed_result_retained=True,
        reorder=True,
        regroup=True,
        retry=True,
        runs={
            key: dict(
                state=fit.state.identity,
                stop_round=fit.stop.completed_rounds,
                reason=fit.stop.reason,
                version=fit.state.version,
                predictions=fit.state.validation_raw.tolist(),
            )
            for key, fit in expected.items()
        },
    )
    if count > 1:
        assert len({v["stop_round"] for v in report["runs"].values()}) > 1
    reports.append(report)

# Preserve the earlier patience suite; custom completion now interacts with both
# built-in and ordered results at the same M sizes.
threshold_problem = replace(p, target=np.repeat([-1.0, 1.0], 3)[:, None])
custom_reports = []
for count in (1, 8, 32):
    specs, expected = [], {}
    for i in range(count):
        context = RunContext(f"custom-mixed-{i}", 7)
        if i % 3 == 0:
            train, recipe = threshold_problem, threshold_recipe
            options = dict(rounds=5, threshold=0.05, bins=6)
        elif i % 3 == 1:
            train, recipe = p, squared
            options = dict(rounds=6, bins=6, patience=1, min_delta=100.0)
        else:
            train, recipe = distribution, ordered.normal
            options = dict(rounds=6, bins=6, patience=3, min_delta=100.0)
        valid = replace(train, target=train.target[::-1])
        specs.append(RunSpec(context, train, valid, recipe, options, prepared))
        expected[context.run_id] = recipe(train, valid, context=context, **options)
    reference = expected[specs[0].context.run_id]
    assert reference.stop.reason == "loss_threshold"
    assert reference.stop.rounds == 5 and reference.stop.completed_rounds == 2
    np.testing.assert_array_equal(reference.steps, [0.125, 0.03125])
    np.testing.assert_array_equal(reference.state.train_raw, threshold_problem.target * 0.75)
    draws = [s.context.rng(1, "leaf", "sample").random(5) for s in specs]
    assert len({tuple(d) for d in draws}) == count
    # M=1 still checks a distinct ID without claiming predictions must differ.
    other = RunContext("distinct-stream-control", 7).rng(1, "leaf", "sample").random(5)
    assert not np.array_equal(draws[0], other)
    malformed = SimpleNamespace(
        state=reference.state,
        steps=reference.steps,
        stop=SimpleNamespace(rounds=5, completed_rounds=2, reason=None),
    )
    failed = replace(specs[0], recipe=lambda *a, result=malformed, **kw: result)
    original_fit = Binning.fit
    Binning.fit = forbidden
    try:
        faults = run_many([failed, *specs[1:]])
        assert faults[0].error_type == "ValueError" and faults[0].result is None
        for out in faults[1:]:
            compare(out, expected[out.run_id])
        executions = [
            run_many(specs),
            run_many(reversed(specs)),
            tuple(r for batch in (specs[::2], specs[1::2]) for r in run_many(batch)),
        ]
        for outcomes in executions:
            for out in outcomes:
                compare(out, expected[out.run_id])
    finally:
        Binning.fit = original_fit
    custom_reports.append(
        dict(
            count=count,
            distinct_run_id_streams=True,
            reorder=True,
            regroup=True,
            same_id_retry=True,
            no_refit=True,
            expected_failure=dict(
                run_id=faults[0].run_id,
                error_type=faults[0].error_type,
                message=faults[0].error_message,
            ),
            runs={key: summary(fit) for key, fit in expected.items()},
        )
    )

preparation_reports = []
for mutation in ("features", "row_ids", "target_weights"):
    if mutation == "features":
        changed_data = NumericData(np.arange(6)[::-1, None], x.row_ids, ("x",))
        changed = Problem(changed_data, p.target, changed_data.row_ids)
    elif mutation == "row_ids":
        changed_data = NumericData(np.arange(6)[:, None], x.row_ids + 100, ("x",))
        changed = Problem(changed_data, p.target, changed_data.row_ids)
    else:
        changed_data = x
        changed = replace(p, target=p.target + 3, weight=[1, 2, 3, 4, 5, 6])
    fresh = PreparedData(changed_data, bins=6)
    context = RunContext(f"preparation-{mutation}", 7)
    options = dict(rounds=2, bins=6)
    good = RunSpec(context, changed, changed, squared, options, fresh)
    stale = replace(good, prepared=prepared)
    neighbor = RunSpec(RunContext("unaffected-neighbor", 7), p, p, squared, options, prepared)
    expected = squared(changed, changed, context=context, **options)
    neighbor_expected = squared(p, p, context=neighbor.context, **options)
    original_fit = Binning.fit
    Binning.fit = forbidden
    try:
        outcomes = run_many([stale, neighbor])
        compare(outcomes[1], neighbor_expected)
        if mutation == "target_weights":
            compare(outcomes[0], expected)
        else:
            assert outcomes[0].error_type == "ValueError" and outcomes[0].result is None
        retries = run_many([neighbor, good])
        compare(retries[0], neighbor_expected)
        compare(retries[1], expected)
    finally:
        Binning.fit = original_fit
    preparation_reports.append(
        dict(
            mutation=mutation,
            old_preparation_accepted=mutation == "target_weights",
            input_identity=changed_data.identity,
            original_identity=x.identity,
            expected_failure=None
            if mutation == "target_weights"
            else dict(error_type=outcomes[0].error_type, message=outcomes[0].error_message),
            fresh_matches_direct=True,
            same_id_retry=True,
            neighbor_unchanged=True,
            result=summary(expected),
        )
    )

# Fresh inference also removes the copied policy source, not only training wheels.
context = RunContext("threshold-inference", 7)
fit = threshold_recipe(threshold_problem, threshold_problem, context=context)
fit.state.model.save(root / "threshold-model.json")
(root / "threshold-inference.json").write_text(
    json.dumps(
        dict(
            values=threshold_problem.data.values.tolist(),
            row_ids=x.row_ids.tolist(),
            names=["x"],
            predictions=fit.state.train_raw.tolist(),
            stop=fit.stop.reason,
            losses=list(fit.steps),
        ),
        indent=2,
    )
    + "\n"
)
(root / "scheduler-checks.json").write_text(
    json.dumps(
        dict(
            schema="installed-scheduling-065",
            imports=dict(openboost=openboost.__file__, ordered=ordered.__file__),
            patience=reports,
            custom=custom_reports,
            preparation=preparation_reports,
        ),
        indent=2,
    )
    + "\n"
)
