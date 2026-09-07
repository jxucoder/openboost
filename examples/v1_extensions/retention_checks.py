"""Installed summary/full and mixed-run parity; author-owned ordered diagnostics."""

import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import ob_ordered_updates as ordered

from openboost import NumericData, Problem, RunContext, recipes
from openboost.runs import RunSpec, run_many
from openboost.runtime import AcceptedState


def check_payload(value):
    assert not isinstance(value, (np.ndarray, AcceptedState))
    if is_dataclass(value):
        for field in fields(value):
            check_payload(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            check_payload(item)


x = NumericData(np.arange(12)[:, None] / 10, np.arange(12), ("x",))
y = (1 + np.sin(x.values[:, 0]))[:, None]
reports = []
for count in (1, 8, 32):
    specs, expected = [], []
    for i in range(count):
        kind = i % 4
        recipe = (recipes.squared, ordered.normal, ordered.formula, recipes.multi_squared)[kind]
        p = Problem(
            x,
            np.column_stack((y, y * 2)) if kind == 3 else y,
            x.row_ids,
            raw_width=1 if kind == 0 else 2,
            structure={"x": x.values + 1} if kind == 2 else {},
        )
        context = RunContext(f"retention-{i}", 0)
        options = dict(rounds=3, bins=4, patience=2)
        expected.append(recipe(p, p, context=context, **options))
        specs.append(RunSpec(context, p, p, recipe, options | dict(retention="summary")))
    outcomes = run_many(specs)
    for full, outcome in zip(expected, outcomes, strict=True):
        assert outcome.error_type is None
        summary = outcome.result
        assert summary.state.identity == full.state.identity and summary.stop == full.stop
        np.testing.assert_array_equal(summary.state.train_raw, full.state.train_raw)
        assert len(summary.steps) == summary.stop.completed_rounds
        check_payload(summary.steps)
    reordered = run_many(tuple(reversed(specs)))
    assert [o.result.state.identity for o in reversed(reordered)] == [
        r.state.identity for r in expected
    ]
    reports.append(
        dict(
            models=count,
            exact_state_stop=True,
            no_summary_arrays_or_states=True,
            reordered_exact=True,
        )
    )
Path(sys.argv[1], "retention-checks.json").write_text(
    json.dumps(dict(passed=True, checks=reports), indent=2) + "\n"
)
