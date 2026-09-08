"""Observe the unchanged failing test; diagnostic success is not conformance."""

import hashlib
import json
import linecache
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from benchmarks.v1.normal_acceptance_trace import SCHEMA, analyze, pack, state_metrics

from openboost import device_normal as normal
from openboost import device_objectives as objectives
from openboost import device_tree as trees
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun

from . import test_device_normal_runtime_cuda as original
from .test_device_runtime_cuda import raw

pytestmark = pytest.mark.gpu
COEFFICIENT_ASSERTION = 'assert tuple(coefficients) == tuple(a[0] for a in ref["attempts"])'


@contextmanager
def observation(run=None, *, ops=None):
    """Copies/exports may add counters; observation must not change owned state."""
    ops = run.ops if run is not None else ops
    context = ops.execution
    before = (
        set(context._buffers),
        set(ops._records),
        None if run is None else (set(run._states), set(run._proposals), run._serial),
    )
    live, upload = context.metrics["live_bytes"], context.metrics["upload_bytes"]
    yield
    after = (
        set(context._buffers),
        set(ops._records),
        None if run is None else (set(run._states), set(run._proposals), run._serial),
    )
    assert after == before
    assert context.metrics["live_bytes"] == live and context.metrics["upload_bytes"] == upload


def snapshot(run, state):
    with observation(run):
        return dict(
            **state_metrics(state),
            training_raw=pack(raw(run.execution, run, state)),
            validation_raw=pack(raw(run.execution, run, state, True)),
        )


@pytest.mark.parametrize("update", ["forward", "reverse"])
def test_capture_original_acceptance_failure(update, monkeypatch, tmp_path):
    trace = dict(
        schema=SCHEMA,
        update=update,
        seed=7,
        run_id="090",
        inputs={},
        initial=None,
        steps=[],
        conformance={"status": "incomplete"},
        ownership_checks_complete=False,
    )
    run = None
    source = Path(original.__file__)
    trace["original_test_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    initialize, grow = DeviceRun.initialize, original.grow
    geometry, direction = normal.geometry, objectives.diagonal_direction
    least_squares, leaf = objectives.least_squares, DeviceOperations.leaf
    propose, resolve = DeviceRun.propose_terms, DeviceRun.resolve

    def capture_initialize(self):
        nonlocal run
        run = self
        state = initialize(self)
        with observation(self):
            # Diagnostic read only: prepared targets/offsets have no public export.
            # Capture actual buffers, not a CPU cast assumed to match preparation.
            for name, problem in (("training", self.problem), ("validation", self._validation)):
                target, offset = self.ops._records[problem][0]
                trace["inputs"][name] = dict(
                    target=pack(self.execution.export(target)),
                    offset=pack(self.execution.export(offset)),
                    weight=pack(self.execution.export(problem.data.weight)),
                    problem_identity=problem.data.problem_identity,
                )
            trace["base"] = pack(self.execution.export(self._base))
            trace["initial"] = snapshot(self, state)
        return state

    def capture_grow(run, state, channels, **kwargs):
        index = len(trace["steps"])
        trace["steps"].append(
            dict(
                round=index // 2,
                channels=list(channels),
                before=snapshot(run, state),
                fields=[],
                roots=[],
                terms=[],
                trials=[],
            )
        )
        result = grow(run, state, channels, **kwargs)
        with observation(run):
            trace["steps"][-1]["terms"] = [
                dict(
                    topology=t.tree.topology,
                    mapping=pack(t.mapping),
                    leaf_values=pack(np.asarray(trees.export(run.ops, t.tree).value)),
                )
                for t in result
            ]
        return result

    def capture_geometry(ops, problem, values):
        result = geometry(ops, problem, values)
        with observation(run, ops=ops):
            record = dict(
                raw=pack(ops.execution.export(values)),
                gradient=pack(ops.execution.export(result[0])),
                fisher=pack(ops.execution.export(result[1])),
            )
            if run is None:
                # Normal.base validates geometry before DeviceRun.initialize.
                trace["initial_geometry"] = record
            else:
                trace["steps"][-1].update(record)
        return result

    def capture_direction(ops, *args, **kwargs):
        result = direction(ops, *args, **kwargs)
        with observation(run):
            trace["steps"][-1]["direction"] = pack(ops.execution.export(result))
        return result

    def capture_fields(ops, *args, **kwargs):
        result = least_squares(ops, *args, **kwargs)
        with observation(run):
            trace["steps"][-1]["fields"].append(
                dict(
                    names=list(result.names),
                    roles=list(result.roles),
                    values=pack(ops.execution.export(result.values)),
                )
            )
        return result

    def capture_leaf(ops, histogram, *, reg_lambda=1.0):
        result = leaf(ops, histogram, reg_lambda=reg_lambda)
        with observation(run):
            trace["steps"][-1]["roots"].append(
                dict(
                    total=pack(ops.execution.export(histogram.total)),
                    row_positions=ops.execution.export(histogram.rows.positions).tolist(),
                    reg_lambda=float(reg_lambda),
                    leaf=pack(ops.execution.export(result)),
                )
            )
        return result

    def capture_propose(self, state, terms, *, coefficient=1.0):
        result = propose(self, state, terms, coefficient=coefficient)
        with observation(self):
            trace["steps"][-1]["trials"].append(
                dict(
                    coefficient=float(coefficient),
                    proposal=dict(
                        identity=result.identity,
                        parent_identity=result.parent_identity,
                        coefficient=result.coefficient,
                        loss=pack(np.float64(result.loss)),
                        validation_score=pack(np.float64(result.validation_score)),
                        training_raw=pack(raw(self.execution, self, result)),
                        validation_raw=pack(raw(self.execution, self, result, True)),
                    ),
                )
            )
        return result

    def capture_resolve(self, state, proposal, *, accept):
        result = resolve(self, state, proposal, accept=accept)
        trace["steps"][-1]["trials"][-1].update(accepted=accept, resolved=snapshot(self, result))
        return result

    output = (
        Path(os.environ.get("OPENBOOST_NORMAL_ARTIFACTS", tmp_path))
        / "acceptance"
        / f"{update}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with monkeypatch.context() as patch:
            for owner, name, wrapper in (
                (DeviceRun, "initialize", capture_initialize),
                (original, "grow", capture_grow),
                (normal, "geometry", capture_geometry),
                (objectives, "diagonal_direction", capture_direction),
                (objectives, "least_squares", capture_fields),
                (DeviceOperations, "leaf", capture_leaf),
                (DeviceRun, "propose_terms", capture_propose),
                (DeviceRun, "resolve", capture_resolve),
            ):
                patch.setattr(owner, name, wrapper)
            try:
                original.test_frozen_three_round_transactions(
                    "conflict", 0, None, "ordinary", 0, update, False, 8.0
                )
            except AssertionError as error:
                tb = error.__traceback__
                while tb.tb_next is not None:
                    tb = tb.tb_next
                line = linecache.getline(tb.tb_frame.f_code.co_filename, tb.tb_lineno).strip()
                trace["conformance"] = dict(status="failure", assertion=line, line=tb.tb_lineno)
                if Path(tb.tb_frame.f_code.co_filename) != source or line != COEFFICIENT_ASSERTION:
                    raise
                trace["conformance"]["kind"] = "known_coefficient_mismatch"
            else:
                trace["conformance"] = dict(status="pass")
        trace["ownership_checks_complete"] = True
        # Independent arithmetic is applied only after the unchanged test ends.
        # Its values never drive proposals, acceptance or best-model selection.
        report = analyze(trace)
        assert trace["initial_geometry"]["raw"] == trace["initial"]["training_raw"]
        assert all(step["raw"] == step["before"]["training_raw"] for step in trace["steps"])
        assert report["steps"]
        assert all(
            t["comparison"][name]["high_precision"]["estimates_agree"]
            for step in report["steps"]
            for t in step["trials"]
            for name in ("training", "validation")
        )
    finally:
        # The original test's ExecutionContext exits on assertion and owns cleanup.
        trace["final_live_bytes"] = None if run is None else run.execution.metrics["live_bytes"]
        trace["final_metrics"] = None if run is None else dict(run.execution.metrics)
        output.write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
    assert trace["final_live_bytes"] == 0
