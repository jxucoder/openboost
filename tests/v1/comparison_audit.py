"""Read-only test instrumentation of actual device comparisons, never a policy."""

import hashlib
import json
import os
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from benchmarks.v1.normal_acceptance_trace import pack

from openboost import device_normal as normal

from .reference.normal_acceptance import compare_precisions, loss_difference
from .reference.normal_comparison import compare


def numerical_audit(result, before, after, target, offset, weight):
    """Check actual bounds against original-row mathematics, not production math."""
    args = before, after, target, offset, weight
    expected = compare(*args)
    oracle = compare_precisions(*args)
    values = [Decimal(oracle[f"decimal{p}"]) for p in (60, 100)]
    if not oracle["estimates_agree"]:
        values.extend(loss_difference(*args, precision=p) for p in (160, 220))
    if result.lower is not None:
        assert all(
            Decimal.from_float(result.lower) <= v <= Decimal.from_float(result.upper)
            for v in values
        )
        assert result.unchanged == np.array_equal(before, after)
    else:
        assert expected.lower is None and result.reason == expected.reason
    if expected.status in ("improvement", "worsening", "unchanged"):
        assert result.status == expected.status
    return dict(
        reference=asdict(expected),
        high_precision=oracle,
        additional_precision=[str(v) for v in values[2:]],
    )


@pytest.fixture
def comparison_audit(monkeypatch, request, tmp_path):
    """Install only in revised cohorts; restore after each test and keep partial data.

    Prepared target/offset handles are read from the operation registry solely for
    diagnostic observation. No author/recipe depends on this private inspection.
    The actual operation completes first. Checks cannot drive an algorithm choice.
    """
    original = normal.compare
    events, inputs = [], {}

    def observed(ops, problem, before, after):
        context = ops.execution
        buffers, records, start = set(context._buffers), set(ops._records), dict(context.metrics)
        result = original(ops, problem, before, after)
        end = dict(context.metrics)
        assert set(context._buffers) == buffers and set(ops._records) == records
        assert end["upload_bytes"] == start["upload_bytes"]
        assert end["live_bytes"] == start["live_bytes"]
        key = problem.data.problem_identity
        target, offset = (context.export(h) for h in ops._records[problem][0])
        weight = context.export(problem.data.weight)
        inputs.setdefault(key, dict(target=pack(target), offset=pack(offset), weight=pack(weight)))
        old, new = context.export(before), context.export(after)
        event = dict(
            problem=key,
            before=pack(old),
            after=pack(new),
            comparison=dict(asdict(result), status=result.status),
            counters={k: end.get(k, 0) - start.get(k, 0) for k in end},
        )
        events.append(event)
        try:
            event["audit"] = numerical_audit(result, old, new, target[:, 0], offset, weight)
        except BaseException as error:
            event["audit_error"] = type(error).__name__
            raise
        return result

    monkeypatch.setattr(normal, "compare", observed)
    try:
        yield events
    finally:
        directory = Path(os.environ.get("OPENBOOST_COMPARISON_TRAJECTORIES", tmp_path))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16] + ".json")
        path.write_text(
            json.dumps(
                dict(
                    case=request.node.nodeid,
                    inputs=inputs,
                    comparisons=events,
                    timing_scope="instrumented correctness only",
                ),
                allow_nan=False,
            )
            + "\n"
        )
