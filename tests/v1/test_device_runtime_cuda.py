"""088 accepted/proposal and two-round checks; real CUDA execution is required."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from openboost import device_recipes as recipes
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun, add_raw
from openboost.execution import ExecutionContext
from openboost.runtime import RunContext
from openboost.stopping import StopState

from .reference.device_rounds import fixture, loss, predict, rounds
from .test_device_round_reference import prepared_fixture
from .test_device_tree_cuda import close, cohort_minimum

pytestmark = pytest.mark.gpu


def start(context, case="weighted", run_id="088"):
    train, validation, binned, info = prepared_fixture(case)
    ops = DeviceOperations(context)
    run = DeviceRun(ops, train, validation, run_id=run_id, seed=7, binning=binned.binning)
    return run, train, validation, info


def raw(context, run, record, validation=False):
    snapshot = run.raw(record, validation=validation)
    value = context.export(snapshot)
    context.release(snapshot)
    return value


def grow(run, state, info, *, depth=2, minimum=None):
    ops, context = run.ops, run.execution
    fields = run.fields(state)
    for i, name in ((1, "cohort:blue"), (0, "cohort:red")):
        column = context.upload(np.asarray(info[:, i], np.float32))
        augmented = ops.add_independent(fields, name, column, nonnegative=True)
        context.release(column)
        ops.release(fields)
        fields = augmented
    tree = trees.depthwise(
        ops,
        run.data,
        fields,
        binning=run.binning,
        max_depth=depth,
        legality=cohort_minimum(minimum) if minimum is not None else None,
    )
    ops.release(fields)
    return tree


@pytest.mark.parametrize("case", ["weighted", "d2", "conflict"])
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("minimum", [None, 0, 1, 2])
def test_two_round_resident_transactions_against_oracle(case, depth, minimum, monkeypatch):
    base, expected = rounds(case, depth=depth, minimum=minimum)
    with ExecutionContext() as context:
        run, train, validation, info = start(context, case)
        state = run.initialize()
        close(raw(context, run, state), np.full((run.data.n_rows, 1), base))
        stop = StopState.start(state.validation_score, rounds=2)
        calls = []
        original_predict = trees.predict

        def counted(ops, tree, data):
            calls.append(data.problem_identity)
            return original_predict(ops, tree, data)

        monkeypatch.setattr(trees, "predict", counted)
        for i, ref in enumerate(expected):
            original_raw = raw(context, run, state)
            gradient = run.gradient(state)
            close(context.export(gradient), ref["gradient"])
            context.release(gradient)
            tree = grow(run, state, info, depth=depth, minimum=minimum)
            assert [None if f == -1 else (f, t, m) for f, t, m, _, _ in tree.topology] == [
                n["key"] for n in ref["nodes"]
            ]
            close(trees.export(run.ops, tree).value[:, 0], [n["value"] for n in ref["nodes"]])
            before = dict(context.metrics)
            proposal = run.propose(state, tree, coefficient=0.5)
            updated = run.resolve(state, proposal, accept=True)
            after = dict(context.metrics)
            assert after["upload_bytes"] == before["upload_bytes"]
            assert after["export_bytes"] - before["export_bytes"] == sum(
                after.get(k, 0) - before.get(k, 0)
                for k in ("validation_export_bytes", "metric_export_bytes")
            )
            assert len(calls) == 2 * (
                i + 1
            )  # Add only the proposed tree, never replay the ensemble.
            close(raw(context, run, state), original_raw)
            close(raw(context, run, proposal)[:, 0], ref["raw"])
            run.ops.release(tree)
            run.release(proposal)
            run.release(state)
            state = updated
            assert state.version == state.n_terms == i + 1
            close(raw(context, run, state)[:, 0], ref["raw"])
            close(raw(context, run, state, True)[:, 0], ref["validation_raw"])
            assert abs(state.loss - ref["loss"]) <= 1e-3 * max(1, abs(ref["loss"]))
            assert abs(state.validation_score - ref["validation_score"]) <= (
                1e-3 * max(1, abs(ref["validation_score"]))
            )
            assert state.best_n_terms == ref["best_round"]
            assert state.best_score == pytest.approx(ref["best_score"], rel=1e-3, abs=1e-3)
            stop = stop.observe(state.validation_score)
        model = run.export(state)
        close(model.predict(train.data), raw(context, run, state))
        close(model.predict(validation.data), raw(context, run, state, True))
        close(
            model.predict(validation.data, offset=validation.offset),
            raw(context, run, state, True) + validation.offset,
        )
        assert len(run.export(state, best=True).terms) == expected[-1]["best_round"]
        assert stop.reason == "budget" and stop.completed_rounds == 2
        run.close()
        assert context.metrics["live_bytes"] == 0


def test_rejection_retry_rng_and_independent_lifetimes():
    with ExecutionContext() as context:
        run, train, validation, info = start(context, "conflict")
        state = run.initialize()
        stop = StopState.start(state.validation_score, rounds=3, patience=2)
        tree = grow(run, state, info)
        original = raw(context, run, state)
        old_snapshot = run.raw(state)
        model_id = run.export(state).identity
        rng = run.rng(0, "learner", "sample").integers(0, 1000000, 12)
        first = run.propose(state, tree, coefficient=0.5)
        proposed = raw(context, run, first)
        assert run.resolve(state, first, accept=False) is state
        assert state.version == 0 and run.export(state).identity == model_id
        np.testing.assert_array_equal(run.rng(0, "learner", "sample").integers(0, 1000000, 12), rng)
        np.testing.assert_array_equal(
            RunContext("088", 7).rng(0, "learner", "sample").integers(0, 1000000, 12), rng
        )
        assert stop.completed_rounds == 0
        stop = stop.observe(state.validation_score)
        run.release(first)
        retry = run.propose(state, tree, coefficient=0.5)
        run.ops.release(tree)
        close(raw(context, run, retry), proposed)
        snapshot = run.raw(retry)
        context.release(snapshot)
        accepted = run.resolve(state, retry, accept=True)
        run.release(retry)
        run.release(state)
        close(raw(context, run, accepted), proposed)
        close(context.export(old_snapshot), original)
        assert accepted.best_n_terms == 0 and accepted.n_terms == 1
        assert not run.export(accepted, best=True).terms
        other, _, _, _ = start(context, "conflict", run_id="other")
        assert not np.array_equal(other.rng(0, "learner", "sample").integers(0, 1000000, 12), rng)
        other.close()
        run.close()
        close(context.export(old_snapshot), original)
        assert context.metrics["live_bytes"] == old_snapshot.nbytes
        context.release(old_snapshot)
        with pytest.raises(RuntimeError, match="closed"):
            run.export(accepted)


@pytest.mark.parametrize("decision", [None, 1, "yes", np.bool_(True)])
def test_explicit_acceptance_and_parent_record_identity(decision):
    with ExecutionContext() as context:
        run, _, _, info = start(context)
        state = run.initialize()
        tree = grow(run, state, info)
        proposal = run.propose(state, tree)
        original = raw(context, run, state)
        with pytest.raises(ValueError, match="boolean"):
            run.resolve(state, proposal, accept=decision)
        with pytest.raises(ValueError, match="forged"):
            run.resolve(replace(state), proposal, accept=True)
        with pytest.raises(ValueError, match="forged"):
            run.resolve(state, replace(proposal), accept=True)
        another = run.initialize()
        with pytest.raises(ValueError, match="parent"):
            run.resolve(another, proposal, accept=True)
        same_identity, _, _, _ = start(context)
        with pytest.raises(ValueError, match="foreign"):
            same_identity.raw(state)
        same_identity.close()
        run.release(proposal)
        with pytest.raises(ValueError, match="released"):
            run.resolve(state, proposal, accept=True)
        close(raw(context, run, state), original)
        run.ops.release(tree)
        run.close()


@pytest.mark.parametrize("phase", ["initialize", "propose", "resolve"])
def test_partial_allocation_failure_is_atomic(phase, monkeypatch):
    with ExecutionContext() as context:
        run, _, _, info = start(context)
        state = run.initialize()
        tree = grow(run, state, info)
        proposal = run.propose(state, tree)
        prior, candidate = raw(context, run, state), raw(context, run, proposal)
        handles, records = set(context._buffers), set(run.ops._records)
        states, proposals = set(run._states), set(run._proposals)
        before, serial = context.metrics["live_bytes"], run._serial
        original_reserve = context._reserve
        calls = 0

        fail_at = 4 if phase == "initialize" else 2

        def fail_later(nbytes):
            nonlocal calls
            calls += 1
            if calls == fail_at:
                raise MemoryError("intentional allocation failure after partial output")
            original_reserve(nbytes)

        with monkeypatch.context() as patch:
            patch.setattr(context, "_reserve", fail_later)
            with pytest.raises(MemoryError, match="after partial output"):
                if phase == "initialize":
                    run.initialize()
                elif phase == "propose":
                    run.propose(state, tree)
                else:
                    run.resolve(state, proposal, accept=True)
        assert context.metrics["live_bytes"] == before
        assert set(context._buffers) == handles and set(run.ops._records) == records
        assert set(run._states) == states and set(run._proposals) == proposals
        assert run._serial == serial
        close(raw(context, run, state), prior)
        close(raw(context, run, proposal), candidate)
        updated = run.resolve(state, proposal, accept=True)
        run.release(state)
        run.release(proposal)
        run.ops.release(tree)
        close(raw(context, run, updated), candidate)
        run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize(
    "coefficient",
    [True, None, "1", 1j, np.nan, np.inf, 10**1000],
    ids=["bool", "none", "string", "complex", "nan", "infinity", "overflow"],
)
def test_invalid_coefficients_preserve_state(coefficient):
    with ExecutionContext() as context:
        run, _, _, info = start(context)
        state = run.initialize()
        tree = grow(run, state, info)
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError, match="coefficient"):
            run.propose(state, tree, coefficient=coefficient)
        assert context.metrics["live_bytes"] == before and state.version == 0
        run.ops.release(tree)
        run.close()


def test_signed_update_overflow_and_run_close_preserves_caller_buffers():
    with ExecutionContext() as context:
        source = context.upload(np.array([[1], [2]], np.float32))
        delta = context.upload(np.array([[3], [-1]], np.float32))
        result = add_raw(DeviceOperations(context), source, delta, -0.5)
        close(context.export(result), [[-0.5], [2.5]])
        huge = context.upload(np.full((2, 1), np.finfo(np.float32).max, np.float32))
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError, match="finite"):
            add_raw(DeviceOperations(context), source, huge, 2)
        assert context.metrics["live_bytes"] == before
        close(context.export(source), [[1], [2]])
        run, _, _, _ = start(context)
        state = run.initialize()
        run.ops.release(run.data)
        with pytest.raises(ValueError, match="released"):
            run.raw(state)
        run.close()
        assert context.metrics["live_bytes"] == before
        context.close()
        with pytest.raises(RuntimeError, match="closed"):
            context.export(source)


@pytest.mark.parametrize("case", ["weighted", "d2", "conflict"])
@pytest.mark.parametrize("policy", ["fixed", "backtracking"])
@pytest.mark.parametrize("count", [0, 2])
def test_recipe_composition_and_scalar_retention(case, policy, count):
    train, validation, binned, _ = prepared_fixture(case)
    base, expected = rounds(case, count=count)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        result = recipes.squared(
            ops,
            train,
            validation,
            run_id="088",
            seed=7,
            binning=binned.binning,
            rounds=count,
            learning_rate=0.5,
            step=policy,
        )
        state, run = result.state, result.run
        assert result.stop.completed_rounds == count and result.stop.reason == "budget"
        assert len(result.steps) == count
        assert len(run._states) == 1 and not run._proposals
        assert all(step.accepted and step.coefficients == (0.5,) for step in result.steps)
        if count:
            close(raw(context, run, state)[:, 0], expected[-1]["raw"])
            assert state.best_n_terms == expected[-1]["best_round"]
        else:
            close(raw(context, run, state), np.full((run.data.n_rows, 1), base))
        tree_bytes = sum(term.tree.n_nodes * 24 for term in run._states[state].terms)
        # Private raw buffers only cover current train/validation; no per-round raw history.
        expected_bytes = sum(h.nbytes for record in run._prepared for h in ops._records[record][1])
        expected_bytes += 4 + 4 * (run.data.n_rows + run.validation_data.n_rows) + tree_bytes
        assert context.metrics["live_bytes"] == expected_bytes
        run.close()
        assert context.metrics["live_bytes"] == 0


def test_recipe_backtracking_patience_and_rejected_rounds():
    train, validation, binned, _ = prepared_fixture("d2")
    f = fixture("d2")
    base, expected = rounds("d2", depth=1, count=1)
    initial = np.full(len(f["x"]), base)
    prediction = predict(expected[0]["nodes"], f["x"])
    initial_loss = loss(f["target"], f["offset"], f["weight"], initial)
    coefficients = []
    for trial in range(6):
        rate = 8 * 0.5**trial
        coefficients.append(rate)
        candidate = initial + rate * prediction
        if loss(f["target"], f["offset"], f["weight"], candidate) < initial_loss:
            break
    with ExecutionContext() as context:
        result = recipes.squared(
            DeviceOperations(context),
            train,
            validation,
            run_id="trials",
            seed=7,
            binning=binned.binning,
            rounds=1,
            max_depth=1,
            learning_rate=8,
            step="backtracking",
        )
        assert result.steps[0].coefficients == tuple(coefficients)
        close(raw(context, result.run, result.state)[:, 0], candidate)
        assert result.stop.completed_rounds == 1
        result.run.close()
        rejected = recipes.squared(
            DeviceOperations(context),
            train,
            validation,
            run_id="rejected",
            seed=7,
            binning=binned.binning,
            rounds=8,
            learning_rate=0,
            step="backtracking",
            max_trials=3,
            patience=2,
        )
        assert rejected.state.version == 0 and rejected.stop.reason == "patience"
        assert rejected.stop.completed_rounds == 2
        assert all(not step.accepted and step.coefficients == (0, 0, 0) for step in rejected.steps)
        rejected.run.close()


def test_long_recipe_retention_and_fresh_cpu_model(tmp_path):
    train, validation, binned, info = prepared_fixture("weighted")
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        called = []

        def learner(ops, data, fields):
            called.append(data.problem_identity)
            for i, name in ((1, "cohort:blue"), (0, "cohort:red")):
                column = context.upload(np.asarray(info[:, i], np.float32))
                fields = ops.add_independent(fields, name, column, nonnegative=True)
            return trees.depthwise(
                ops, data, fields, binning=binned.binning, legality=cohort_minimum(1), max_depth=2
            )

        result = recipes.squared(
            ops,
            train,
            validation,
            run_id="extension",
            seed=7,
            binning=binned.binning,
            rounds=24,
            learner=learner,
            learning_rate=0.5,
        )
        assert len(called) == 24 and result.state.n_terms == 24
        assert len(result.run._states) == 1 and not result.run._proposals
        # Only prepared records and immutable model trees remain in the operations registry.
        assert len(ops._records) == 4 + 24
        expected = raw(context, result.run, result.state, True)
        path, inputs, output = (tmp_path / n for n in ("model.json", "x.npy", "prediction.npy"))
        result.run.export(result.state).save(path)
        np.save(inputs, validation.data.values)
        result.run.close()
        assert context.metrics["live_bytes"] == 0
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys; sys.modules['cupy']=None; sys.modules['numba']=None",
                    "import numpy as np",
                    "from openboost.artifacts import Model",
                    "from openboost.data import NumericData",
                    "model = Model.load(sys.argv[1]); x = np.load(sys.argv[2])",
                    "data = NumericData(x, np.arange(len(x)), model.feature_names)",
                    "np.save(sys.argv[3], model.predict(data))",
                ]
            ),
            str(path),
            str(inputs),
            str(output),
        ],
        check=True,
    )
    close(np.load(output), expected)


@pytest.mark.parametrize("invalid", ["target", "offset", "weight", "base"])
def test_unrepresentable_preparation_rolls_back(invalid):
    train, validation, binned, _ = prepared_fixture("weighted")
    if invalid in ("target", "offset"):
        train = replace(train, **{invalid: np.full(train.target.shape, 1e100)})
    elif invalid == "weight":
        train = replace(train, weight=np.full(len(train.weight), 1e-100))
    else:
        train = replace(
            train,
            target=np.full(train.target.shape, 3e38),
            offset=np.full(train.offset.shape, -3e38),
        )
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with pytest.raises((ValueError, FloatingPointError)):
            DeviceRun(ops, train, validation, run_id="invalid", seed=0, binning=binned.binning)
        assert context.metrics["live_bytes"] == 0 and not ops._records


def test_automatic_training_binning_and_owned_stream():
    import cupy as cp

    train, validation, _, _ = prepared_fixture("weighted")
    expected = Binning.fit(train.data, bins=4)
    with ExecutionContext() as context:
        external = cp.cuda.Stream(non_blocking=True)
        with external:
            run = DeviceRun(
                DeviceOperations(context), train, validation, run_id="auto", seed=7, bins=4
            )
            assert run.binning.identity == expected.identity
            np.testing.assert_array_equal(
                context.export(run.validation_data.codes), expected.transform(validation.data).codes
            )
            state = run.initialize()
            fields = run.fields(state)
            tree = trees.depthwise(run.ops, run.data, fields, binning=run.binning)
            proposal = run.propose(state, tree)
            updated = run.resolve(state, proposal, accept=True)
            close(run.export(updated).predict(validation.data), raw(context, run, updated, True))
            assert cp.cuda.get_current_stream().ptr == external.ptr
            run.ops.release(fields)
            run.ops.release(tree)
            run.close()
            assert cp.cuda.get_current_stream().ptr == external.ptr


def test_retained_prior_state_and_failed_recipe_cleanup():
    with ExecutionContext() as context:
        run, train, validation, info = start(context)
        state0 = run.initialize()
        tree = grow(run, state0, info)
        proposal = run.propose(state0, tree, coefficient=0.5)
        state1 = run.resolve(state0, proposal, accept=True)
        run.release(proposal)
        run.release(state0)
        model1 = run.export(state1)
        proposal = run.propose(state1, tree, coefficient=0.25)
        state2 = run.resolve(state1, proposal, accept=True)
        run.release(proposal)
        run.ops.release(tree)
        assert run.export(state1).identity == model1.identity
        close(run.export(state1).predict(train.data), raw(context, run, state1))
        run.release(state1)
        close(run.export(state2).predict(validation.data), raw(context, run, state2, True))
        run.close()
        assert context.metrics["live_bytes"] == 0

        def broken(ops, data, fields):
            context.upload(np.ones(10, np.float32))
            raise RuntimeError("learner failed after allocating scratch")

        ops = DeviceOperations(context)
        with pytest.raises(RuntimeError, match="learner failed"):
            recipes.squared(
                ops, train, validation, run_id="failure", seed=0, learner=broken, bins=4
            )
        assert context.metrics["live_bytes"] == 0 and not ops._records
