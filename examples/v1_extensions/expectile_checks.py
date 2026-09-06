"""Copied outside the repository and executed against installed D1 wheel."""

import json
import sys
from pathlib import Path

import numpy as np

from openboost import NumericData, Problem, RunContext
from openboost.runs import RunSpec, run_many


def check(plugin, root):
    record = json.loads((root / "expectile-expected.json").read_text())
    data = NumericData(record["values"], np.arange(6), ("feature",))
    p = Problem(
        data,
        np.array(record["target"])[:, None],
        data.row_ids,
        weight=record["weight"],
        offset=np.array(record["offset"])[:, None],
    )
    objective = plugin.Expectile()
    np.testing.assert_allclose(objective.base(p), [record["base"]], atol=1e-12)
    result = plugin.fit(p, p, context=RunContext("expectile", 19), bins=6)
    for (before, after), expected in zip(result.steps, record["trace"], strict=True):
        loss, g, h = objective.geometry(p, before)
        np.testing.assert_allclose(loss, expected["loss"], atol=1e-12)
        np.testing.assert_allclose(g, expected["gradient"], atol=1e-12)
        np.testing.assert_allclose(h, expected["curvature"], atol=1e-12)
        np.testing.assert_allclose(after[:, 0], expected["raw"], atol=1e-12)
    outcome = run_many([RunSpec(RunContext("expectile", 19), p, p, plugin.fit, {"bins": 6})])[0]
    assert outcome.error_type is None
    np.testing.assert_array_equal(outcome.result.state.train_raw, result.state.train_raw)
    result.state.model.save(root / "expectile-model.json")
    report = dict(
        passed=True,
        values=record["values"],
        raw=result.state.train_raw.tolist(),
        rounds=result.stop.completed_rounds,
        reason=result.stop.reason,
        max_abs_error=float(
            np.max(np.abs(result.state.train_raw[:, 0] - record["trace"][-1]["raw"]))
        ),
    )
    (root / "expectile-checks.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    import ob_expectile

    assert "site-packages" in ob_expectile.__file__
    check(ob_expectile, Path(sys.argv[1]))
