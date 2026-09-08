"""092-D real lowering and separately uninstrumented bounded fit-cost evidence."""

import hashlib
import json
import os
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from openboost import device_normal, device_recipes
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.compared_normal import rounds
from .reference.coupled import normal_scores
from .test_compared_normal_extension_cuda import installed_learner
from .test_device_normal_comparison_cuda import prepared
from .test_device_normal_cuda import close
from .test_device_normal_reference import prepared_fixture
from .test_device_runtime_cuda import raw
from .test_device_score_symmetry_cuda import assembly

pytestmark = pytest.mark.gpu


def save(tmp_path, name, result):
    directory = Path(os.environ.get("OPENBOOST_COMPARISON_DIAGNOSTICS", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def test_actual_comparison_lowering_retains_directed_double_operations(tmp_path):
    from openboost import _device_kernels

    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        _, problem = prepared(ops, [0, 2], [[0, 0.125], [0.25, -0.125]], [1, 3])
        old = context.upload(np.array([[1, 0], [1, 0]], np.float32))
        new = context.upload(np.array([[0.5, 0.125], [1.5, -0.125]], np.float32))
        change = device_normal.compare(ops, problem, old, new)
        kernels = {
            name: assembly(context, getattr(_device_kernels, name))
            for name in ("normal_compare_rows", "normal_compare_reduce")
        }
        ptx = "\n".join(row["ptx"] for rows in kernels.values() for row in rows)
        required = [
            f"{operation}.{direction}.f64"
            for operation in ("add", "mul", "div")
            for direction in ("rm", "rp")
        ]
        present = {name: bool(re.search(r"\b" + re.escape(name) + r"\b", ptx)) for name in required}
        # Retain full lowering even if an expected instruction is absent.
        save(
            tmp_path,
            "lowering.json",
            dict(
                kernels=kernels,
                required=present,
                status=change.status,
                ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                counters=dict(context.metrics),
            ),
        )
        assert all(present.values())
        assert change.lower is not None


def test_uninstrumented_end_to_end_fit_cost_and_reference_quality(tmp_path):
    assert device_normal.compare.__module__ == "openboost.device_normal"
    measurements = []
    for case in ("weighted", "d2"):
        _, expected = rounds(
            case,
            depth=2 if case == "d2" else 1,
            minimum=1 if case == "d2" else None,
            update="forward",
            mode="natural",
            count=3,
            rate=0.1,
            fixed=True,
        )
        identities = []
        for repeat in range(2):
            started = perf_counter()
            train, validation, binned, information = prepared_fixture(case)
            with ExecutionContext() as context:
                ops = DeviceOperations(context)
                learner = (
                    installed_learner()(ops, train, information, binning=binned.binning)
                    if case == "d2"
                    else None
                )
                options = dict(learner=learner) if learner is not None else dict(max_depth=1)
                result = device_recipes.normal(
                    ops,
                    train,
                    validation,
                    run_id="092-cost",
                    seed=7,
                    binning=binned.binning,
                    rounds=3,
                    update="forward",
                    mode="natural",
                    step="fixed",
                    learning_rate=0.1,
                    **options,
                )
                model = result.run.export(result.state)
                context.synchronize()
                fit_seconds = perf_counter() - started
                fit_metrics = dict(context.metrics)
                prediction_started = perf_counter()
                prediction = model.predict(validation.data)
                prediction_seconds = perf_counter() - prediction_started
                actual = raw(context, result.run, result.state, True)
                close(actual, expected[-1]["validation_raw"])
                close(prediction, actual)
                scores = normal_scores(
                    actual + validation.offset, validation.target[:, 0], weight=validation.weight
                )
                reference_scores = normal_scores(
                    expected[-1]["validation_raw"] + validation.offset,
                    validation.target[:, 0],
                    weight=validation.weight,
                )
                close(scores, reference_scores)
                identities.append(model.identity)
                result.run.close()
                if learner is not None:
                    learner.close()
                assert context.metrics["live_bytes"] == 0
                assert fit_metrics["comparison_calls"] > 0
                assert (
                    fit_metrics["comparison_export_bytes"] == 32 * fit_metrics["comparison_calls"]
                )
                measurements.append(
                    dict(
                        case=case,
                        repeat=repeat,
                        fit_including_fixture_context_preparation_training_export_seconds=fit_seconds,
                        cpu_prediction_seconds=prediction_seconds,
                        fit_metrics=fit_metrics,
                        final_metrics=dict(context.metrics),
                        model_identity=model.identity,
                        validation_nll_crps=list(scores),
                        reference_nll_crps=list(reference_scores),
                    )
                )
                save(
                    tmp_path,
                    "cost.json",
                    dict(
                        timing_scope="Uninstrumented correctness fixtures; compilation may be warm from earlier tests.",
                        claim="No speed ratio, real-data quality, or author/adoption claim.",
                        measurements=measurements,
                    ),
                )
        assert identities[0] == identities[1]
