"""Check installed ordered recipes against separate frozen reference traces."""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import ob_ordered_updates as ordered

from openboost import NumericData, Problem, RunContext
from openboost.runs import RunSpec, run_many

assert "site-packages" in ordered.__file__
root = Path(sys.argv[1])
source = json.loads((root / "ordered-expected.json").read_text())
x = NumericData(source["values"], np.arange(6), ("feature",))
predictions, errors, versions = {}, [], {}
for i, case in enumerate(source["records"]):
    p = Problem(
        x,
        np.array(source["target"])[:, None],
        x.row_ids,
        weight=source["weight"],
        raw_width=2,
        structure={"x": np.array(source["structure"])[:, None]}
        if case["family"] == "formula"
        else {},
    )
    recipe = getattr(ordered, case["family"])
    options = {"mode": case["mode"]} if case["family"] == "normal" else {}
    context = RunContext(f"ordered-{i}", 7)
    result = recipe(p, p, context=context, rounds=3, bins=6, order=case["order"], **options)
    for actual, expected in zip((s for r in result.steps for s in r), case["trace"], strict=True):
        np.testing.assert_allclose(actual.before.train_raw, expected["raw_before"], atol=1e-9)
        np.testing.assert_allclose(actual.after.train_raw, expected["raw_after"], atol=1e-9)
        assert list(actual.coefficients) == expected["coefficients"]
        assert (actual.before is not actual.after) == expected["accepted"]
        errors.append(float(np.max(np.abs(actual.after.train_raw - expected["raw_after"]))))
    name = f"ordered-{i}"
    result.state.model.save(root / f"{name}.json")
    predictions[name] = result.state.model.predict(x).tolist()
    versions[name] = result.state.version
    assert result.stop.completed_rounds == 3
    assert result.state.version == sum(s["accepted"] for s in case["trace"])
    outcome = run_many(
        [
            RunSpec(
                context,
                p,
                p,
                partial(recipe, order=case["order"], **options),
                {"rounds": 3, "bins": 6},
            )
        ]
    )[0]
    assert outcome.error_type is None
    assert outcome.result.state.identity == result.state.identity
    assert outcome.result.stop == result.stop
(root / "ordered-checks.json").write_text(
    json.dumps(
        dict(
            values=source["values"],
            predictions=predictions,
            versions=versions,
            max_absolute_error=max(errors),
            scheduler_status="structural result accepted; all six cases match independent execution",
        ),
        indent=2,
    )
    + "\n"
)
