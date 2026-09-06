"""Installed D5 structural-result and independent scheduling development checks."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import ob_ordered_updates as ordered

from openboost import NumericData, Problem, RunContext
from openboost.binning import Binning, PreparedData
from openboost.recipes import squared
from openboost.runs import RunSpec, run_many

assert "site-packages" in ordered.__file__
root = Path(sys.argv[1])
x = NumericData(np.arange(6)[:, None], np.arange(6), ("x",))
p = Problem(x, [[0.5], [1], [2], [3], [5], [8]], x.row_ids)
distribution = replace(p, raw_width=2, offset=np.zeros((6, 2)))
prepared = PreparedData(x, bins=6)
reports = []
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
(root / "scheduler-checks.json").write_text(json.dumps(reports, indent=2) + "\n")
