"""One frozen CPU diagnostic fit; imports and input loading are in outer wall time."""

import argparse
import hashlib
import json
import resource
import sys
import time
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openboost import NumericData, Problem, RunContext
from openboost.artifacts import Model
from openboost.recipes import normal, squared
from openboost.tree import Tree, depthwise

if __package__:
    from benchmarks.v1.profile_worker import ProfileDeadline, profile_call
else:
    from profile_worker import ProfileDeadline, profile_call


def threadpool_info():
    from threadpoolctl import threadpool_info as inspect

    return inspect()


def rejection_fixtures():
    data = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
    scalar = Problem(data, [[-3], [-3], [3], [3]], data.row_ids)
    accepted = squared(
        scalar,
        scalar,
        context=RunContext("backtracking", 0),
        rounds=1,
        learning_rate=16,
        step="backtracking",
        learner=depthwise,
    )
    assert accepted.steps[0].coefficients == (16, 8, 4, 2)
    assert accepted.steps[0].accepted and accepted.state.version == 1
    rejected = squared(
        scalar,
        scalar,
        context=RunContext("rejection", 0),
        rounds=2,
        learning_rate=1024,
        step="backtracking",
        learner=depthwise,
    )
    distribution = Problem(data, scalar.target, data.row_ids, raw_width=2)

    def zero(binned, stats):
        return depthwise(binned, stats, max_depth=0, leaf=lambda *_: 0)

    joint = normal(
        distribution, distribution, context=RunContext("joint-rejection", 0), rounds=2, learner=zero
    )
    for result in (rejected, joint):
        assert result.state.version == 0 and not result.state.model.terms
        assert result.stop.completed_rounds == 2
        for item in result.steps:
            assert not item.accepted and len(item.coefficients) == 6
            np.testing.assert_array_equal(item.raw_before, item.raw_after)
    return dict(
        backtracking_coefficients=list(accepted.steps[0].coefficients),
        scalar_rejection_trials=[len(s.coefficients) for s in rejected.steps],
        joint_rejection_trials=[len(s.coefficients) for s in joint.steps],
        rejection_state_unchanged=True,
    )


def run(job, directory):
    root = Path(directory)
    input_path = Path(job["input_path"])
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != job["input_sha256"]:
        raise ValueError("diagnostic input hash mismatch")
    with np.load(input_path, allow_pickle=False) as z:
        expected = {
            "x_train",
            "y_train",
            "train_ids",
            "x_validation",
            "y_validation",
            "validation_ids",
        }
        if set(z.files) != expected:
            raise ValueError("only frozen training/validation arrays are allowed")
        arrays = {key: z[key] for key in z.files}
    start = time.monotonic()
    width = {"squared": 1, "normal": 2}[job["recipe"]]
    recipe = {"squared": squared, "normal": normal}[job["recipe"]]
    n = job["train_rows"]
    problems = []
    for part, count in (("train", n), ("validation", 1024)):
        x, y, ids = arrays["x_" + part], arrays["y_" + part], arrays[part + "_ids"]
        if len(x) < count or len(y) != len(x) or len(ids) != len(x):
            raise ValueError("frozen prefix unavailable or misaligned")
        data = NumericData(x[:count], ids[:count], tuple(job["features"]))
        problems.append(Problem(data, y[:count, None], data.row_ids, raw_width=width))
    train, valid = problems
    if np.intersect1d(train.row_ids, valid.row_ids).size:
        raise ValueError("training and validation source rows overlap")
    setup_s = time.monotonic() - start
    draws = RunContext(job["id"], 0)
    options = dict(
        rounds=job["rounds"],
        bins=32,
        max_depth=2,
        learning_rate=0.1,
        reg_lambda=1.0,
        patience=None,
        step="fixed",
    )
    if width == 2:
        options.update(mode="natural", damping=0.0, minimum_scale=1e-6)
    calls = []
    original = Tree.predict

    def counted(tree, data):
        calls.append(len(data.values))
        return original(tree, data)

    def fit():
        return recipe(train, valid, context=draws, **options)

    begin = time.monotonic()
    if job["instrumented"]:
        with patch.object(Tree, "predict", counted):
            result = profile_call(fit, root, 60)
    else:
        result = fit()
    fit_s = time.monotonic() - begin
    begin = time.monotonic()
    training_raw = result.state.model.predict(train.data)
    validation_raw = result.state.model.predict(valid.data)
    predict_s = time.monotonic() - begin
    begin = time.monotonic()
    result.state.model.save(root / "model.json")
    export_s = time.monotonic() - begin
    # Verification is outside fit/predict/export timing and the profiled interval.
    np.testing.assert_array_equal(training_raw, result.state.train_raw)
    np.testing.assert_array_equal(validation_raw, result.state.validation_raw)
    np.testing.assert_array_equal(
        Model.load(root / "model.json").predict(valid.data), validation_raw
    )
    assert result.stop.reason == "budget" and result.stop.completed_rounds == job["rounds"]
    assert result.state.version == job["rounds"]
    if width == 1:
        metric = float(np.mean((validation_raw[:, 0] - valid.target[:, 0]) ** 2) / 2)
    else:
        metric = float(
            np.mean(
                0.5 * np.log(2 * np.pi)
                + validation_raw[:, 1]
                + 0.5
                * ((valid.target[:, 0] - validation_raw[:, 0]) / np.exp(validation_raw[:, 1])) ** 2
            )
        )
    if not np.isfinite(metric):
        raise ValueError("nonfinite independent validation metric")
    retained = {
        id(v): v
        for item in result.steps
        for field in fields(item)
        if isinstance(v := getattr(item, field.name), np.ndarray)
    }
    np.savez(root / "predictions.npz", raw=validation_raw, row_ids=valid.row_ids)
    pools = threadpool_info()
    if not pools or any(p["num_threads"] != 2 for p in pools):
        raise ValueError("numerical thread policy differs")
    record = dict(
        status="pass",
        recipe=job["recipe"],
        rounds=job["rounds"],
        train_rows=n,
        validation_rows=len(valid.row_ids),
        parameters=width,
        options=options,
        setup_s=setup_s,
        fit_s=fit_s,
        predict_s=predict_s,
        export_s=export_s,
        end_to_end_s=setup_s + fit_s + predict_s + export_s,
        instrumented=job["instrumented"],
        validation_metric=metric,
        final_raw_exact=True,
        tree_terms=len(result.state.model.terms),
        tree_predict_calls=len(calls) if job["instrumented"] else None,
        tree_row_visits=sum(calls) if job["instrumented"] else None,
        trace_array_bytes=sum(v.nbytes for v in retained.values()),
        guest_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        address_limit_bytes=list(resource.getrlimit(resource.RLIMIT_AS)),
        pools=pools,
        timing_scope="setup records + full fit including binning/validation + train/validation prediction + export; outer wall also includes imports, input load and verification",
    )
    (root / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, nargs="?")
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args()
    if args.fixtures:
        Path("fixtures.json").write_text(json.dumps(rejection_fixtures(), indent=2) + "\n")
        return
    try:
        run(json.loads(args.job.read_text()), Path.cwd())
    except ProfileDeadline:
        raise SystemExit(124) from None


if __name__ == "__main__":
    main()
