"""One bounded diagnostic of the exact consumed mixed-policy reference failure."""

import json
import os
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

from openboost import device_group_runs, device_runs
from openboost.device import DeviceOperations, _workspace
from openboost.execution import ExecutionContext

from .glm_artifacts import input_snapshot
from .test_device_active_cuda import prepare
from .test_grouped_runs_cuda import observed, snapshot, unchanged

pytestmark = pytest.mark.gpu


def mixed(specs):
    # Freeze tooling compares this expression's AST to the failed original test.
    return tuple(replace(s, options=dict(s.options, max_depth=i%3, learning_rate=(8, 64, .5)[i%3],
                                        step='backtracking', max_trials=6, rounds=2, patience=None,
                                        reg_lambda=(1, 2)[i%2], split_penalty=.25)) for i, s in enumerate(specs))


def test_capture_exact_mixed_policy_reference_errors_and_scheduler_outcomes(tmp_path):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 6)
            specs = mixed(specs)
            before = snapshot(ops)
            baseline = device_runs.run_many(ops, specs)
            unchanged(ops, before)
            failed = [o for o in baseline if o.result is None]
            assert failed, 'consumed reference failure must reproduce with unchanged inputs/sources'
            traces = []
            for outcome in failed:
                spec = next(s for s in specs if s.run_id == outcome.run_id)
                try:
                    device_runs._one(ops, spec)
                except Exception as error:
                    assert type(error).__name__ == outcome.error_type and str(error) == outcome.error_message
                    traces.append(dict(run_id=spec.run_id, error_type=type(error).__name__,
                                       error_message=str(error), traceback=traceback.format_exc()))
                else:
                    pytest.fail('exact failed reference call did not reproduce')
                unchanged(ops, before)
            schedules = []
            for size in (2, 32):
                actual = device_group_runs.run_many(ops, specs, group_size=size)
                unchanged(ops, before)
                schedules.append(dict(group_size=size, outcomes=observed(actual)))
            payload = dict(scope='Diagnostic only: unchanged failed reference fixture and actual scheduler outcomes; no missing-consumer pass inferred.',
                           count=6, baseline=observed(baseline), traces=traces, schedules=schedules,
                           inputs=[dict(run_id=s.run_id, seed=s.seed, options=dict(s.options),
                                        train=input_snapshot(s.train), validation=input_snapshot(s.validation)) for s in specs],
                           input_live_bytes=before[0], final_live_bytes=context.metrics['live_bytes'])
            root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'scheduler-diagnostic'
            root.mkdir(parents=True, exist_ok=True)
            (root / 'mixed-policy.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
