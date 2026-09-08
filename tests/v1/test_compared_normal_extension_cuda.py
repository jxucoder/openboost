"""092 revised extension cohort; original settings/tolerances, objective comparisons."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

import openboost
from openboost import device_recipes as recipes
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .comparison_audit import comparison_audit as comparison_audit
from .reference.compared_normal import rounds
from .reference.coupled import normal_scores
from .reference.normal_precision import LOSS_ATOL, LOSS_RTOL
from .test_device_normal_cuda import close
from .test_device_normal_reference import prepared_fixture
from .test_device_runtime_cuda import raw

pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("comparison_audit")]
ROOT = Path(__file__).resolve().parents[2]


def installed_learner():
    import ob_cohort_splits.device as extension

    path = Path(extension.__file__).resolve()
    assert "site-packages" in path.parts and not path.is_relative_to(ROOT)
    source = ROOT / "examples/v1_extensions/cohort_splits/src/ob_cohort_splits/device.py"
    assert path.read_bytes() == source.read_bytes()
    assert importlib.metadata.version("ob-cohort-splits") == "0.1.0"
    return extension.DeviceCohortLearner


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("mode,damping", [("ordinary", 0), ("natural", 0), ("natural", 0.25)])
@pytest.mark.parametrize("fixed", [True, False])
def test_installed_d2_normal_and_fresh_cpu(update, mode, damping, fixed, tmp_path):
    cls = installed_learner()
    train, validation, binned, information = prepared_fixture("d2")
    rate = 0.1 if fixed else 8.0
    _, expected = rounds(
        "d2", update=update, mode=mode, damping=damping, minimum=1, fixed=fixed, rate=rate
    )
    _, unconstrained = rounds("d2", update=update, mode=mode, damping=damping, fixed=True)
    assert expected[0]["nodes"][0][0]["key"] == (0, 1, False)
    assert unconstrained[0]["nodes"][0][0]["key"] == (0, 0, False)
    directory = Path(os.environ.get("OPENBOOST_REVISED_NORMAL_ARTIFACTS", tmp_path)) / (
        f"d2-{mode}-{damping}-{update}-{'fixed' if fixed else 'backtracking'}"
    )
    directory.mkdir(parents=True)
    measurements = []
    # First and repeated fit in this case. Prior tests may have compiled kernels;
    # neither measurement is labelled a process-cold end-to-end benchmark.
    for repeat in range(2):
        started = perf_counter()
        with ExecutionContext() as context:
            ops = DeviceOperations(context)
            learner = cls(ops, train, information, binning=binned.binning)
            cohort_bytes = context.metrics["live_bytes"]
            assert cohort_bytes == information.size * 4
            observed = []

            def tracked(ops, data, fields, learner=learner, observed=observed):
                before = context.metrics["upload_bytes"]
                tree = learner(ops, data, fields)
                assert context.metrics["upload_bytes"] - before == tree.n_nodes * 20
                observed.append([None if f == -1 else (f, t, m) for f, t, m, _, _ in tree.topology])
                return tree

            result = recipes.normal(
                ops,
                train,
                validation,
                run_id="installed-d2",
                seed=7,
                rounds=3,
                mode=mode,
                damping=damping,
                update=update,
                step="fixed" if fixed else "backtracking",
                learning_rate=rate,
                learner=tracked,
                binning=binned.binning,
            )
            model = result.run.export(result.state)
            model.save(directory / "model.json")
            context.synchronize()
            fit_seconds = perf_counter() - started
            fit_metrics = dict(context.metrics)
            assert observed == [[n["key"] for n in nodes] for s in expected for nodes in s["nodes"]]
            assert len(result.steps) == len(expected)
            assert [s.after_version for s in result.steps] == [s["version"] for s in expected]
            for step, reference in zip(result.steps, expected, strict=True):
                assert step.coefficients == tuple(
                    float(np.float32(a[0])) for a in reference["attempts"]
                )
                assert [t.accepted for t in step.trials] == [
                    a[2] == "accepted" for a in reference["attempts"]
                ]
            assert result.state.best_n_terms == expected[-1]["best_terms"]
            assert result.state.loss == pytest.approx(
                expected[-1]["loss"], rel=LOSS_RTOL, abs=LOSS_ATOL
            )
            actual = raw(context, result.run, result.state, True)
            close(actual, expected[-1]["validation_raw"])
            before_predict = perf_counter()
            prediction = model.predict(validation.data)
            prediction_seconds = perf_counter() - before_predict
            close(prediction, actual)
            scores = normal_scores(
                actual + validation.offset, validation.target[:, 0], weight=validation.weight
            )
            expected_scores = normal_scores(
                expected[-1]["validation_raw"] + validation.offset,
                validation.target[:, 0],
                weight=validation.weight,
            )
            close(scores, expected_scores)
            trials = [
                dict(
                    coefficient=t.coefficient,
                    loss=t.loss,
                    validation_score=t.validation_score,
                    accepted=t.accepted,
                    failure=t.failure,
                )
                for s in result.steps
                for t in s.trials
            ]
            result.run.close()
            assert context.metrics["live_bytes"] == cohort_bytes
            learner.close()
            learner.close()
            assert context.metrics["live_bytes"] == 0
            measurements.append(
                dict(
                    repeat=repeat,
                    instrumented_fit_including_context_preparation_export_seconds=fit_seconds,
                    cpu_prediction_seconds=prediction_seconds,
                    fit_metrics=fit_metrics,
                    final_metrics=dict(context.metrics),
                    trials=trials,
                    validation_nll_crps=list(scores),
                )
            )
    fresh_replay(directory, validation, actual, measurements)
    print("NORMAL_D2_MEASUREMENT=" + json.dumps(dict(case=directory.name, repeats=measurements)))


def fresh_replay(directory, validation, actual, measurements):
    # JSON null represents a missing numeric input, so all retained JSON is strict.
    values = [[None if np.isnan(v) else float(v) for v in row] for row in validation.data.values]
    record = dict(
        values=values,
        row_ids=validation.data.row_ids.tolist(),
        feature_names=validation.data.feature_names,
        expected_raw=actual.tolist(),
        offset=validation.offset.tolist(),
    )
    (directory / "inputs.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (directory / "measurement.json").write_text(
        json.dumps(measurements, indent=2, allow_nan=False) + "\n"
    )
    python = os.environ["OPENBOOST_FRESH_CPU_PYTHON"]
    subprocess.run(
        [python, "-I", str(ROOT / "benchmarks/v1/normal_cpu_inference.py"), str(directory)],
        check=True,
        timeout=30,
    )
    replay = json.loads((directory / "cpu-replay.json").read_text())
    installed = Path(openboost.__file__).parent
    assert replay["sources"] == {
        "src/openboost/" + str(p.relative_to(installed)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in installed.rglob("*.py")
    }
    assert replay["numpy"] == "2.3.5"
    close(replay["raw"], actual)


def test_missing_normal_replays_without_cuda_or_training_plugin(tmp_path):
    train, validation, binned, _ = prepared_fixture("weighted")
    assert np.any(np.isnan(train.data.values)) and np.any(np.isnan(validation.data.values))
    directory = (
        Path(os.environ.get("OPENBOOST_REVISED_NORMAL_ARTIFACTS", tmp_path)) / "missing-normal"
    )
    directory.mkdir(parents=True)
    with ExecutionContext() as context:
        result = recipes.normal(
            DeviceOperations(context),
            train,
            validation,
            run_id="missing-replay",
            seed=7,
            rounds=3,
            update="forward",
            step="fixed",
            max_depth=1,
            binning=binned.binning,
        )
        actual = raw(context, result.run, result.state, True)
        result.run.export(result.state).save(directory / "model.json")
        result.run.close()
        assert context.metrics["live_bytes"] == 0
    fresh_replay(directory, validation, actual, [])


def test_installed_d2_rejects_wrong_identity_and_releases_after_failure():
    cls = installed_learner()
    train, validation, binned, information = prepared_fixture("d2")
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        learner = cls(ops, train, information, binning=binned.binning)
        saved_bytes = context.metrics["live_bytes"]
        # Validation is a different problem even when its shape/schema matches.
        with pytest.raises(ValueError, match="different operations or problem"):
            recipes.normal(
                ops,
                validation,
                validation,
                run_id="wrong",
                seed=7,
                learner=learner,
                binning=binned.binning,
            )
        assert context.metrics["live_bytes"] == saved_bytes
        learner.close()
        with pytest.raises(ValueError, match="closed"):
            recipes.normal(
                ops,
                train,
                validation,
                run_id="closed",
                seed=7,
                learner=learner,
                binning=binned.binning,
            )
        assert context.metrics["live_bytes"] == 0
