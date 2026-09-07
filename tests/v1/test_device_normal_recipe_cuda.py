"""090-D actual joint/ordered resident recipes and bounded trial accounting."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost import device_normal as objective
from openboost import device_recipes as recipes
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun, DeviceTerm
from openboost.execution import ExecutionContext

from .reference.device_normal import rounds
from .reference.normal_precision import LOSS_ATOL, LOSS_RTOL
from .test_device_normal_cuda import close
from .test_device_normal_reference import prepared_fixture
from .test_device_runtime_cuda import raw

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("mode", ["ordinary", "natural"])
@pytest.mark.parametrize("step", ["fixed", "backtracking"])
@pytest.mark.parametrize("count", [0, 3])
def test_recipe_matches_frozen_trajectory_and_retention(update, mode, step, count):
    train, validation, binned, _ = prepared_fixture("weighted")
    rate = 0.1 if step == "fixed" else 8
    initial, expected = rounds(
        "weighted", depth=1, update=update, mode=mode, fixed=step == "fixed", rate=rate, count=count
    )
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        result = recipes.normal(
            ops,
            train,
            validation,
            run_id="recipe",
            seed=7,
            rounds=count,
            update=update,
            mode=mode,
            step=step,
            learning_rate=rate,
            binning=binned.binning,
            max_depth=1,
        )
        assert result.stop.completed_rounds == count and result.stop.reason == "budget"
        assert len(result.steps) == len(expected)
        before_version = 0
        for actual, ref in zip(result.steps, expected, strict=True):
            assert actual.channels == ref["channels"] and actual.round_index == ref["round"]
            assert (
                actual.before_version == before_version and actual.after_version == ref["version"]
            )
            before_version = actual.after_version
            assert actual.accepted == ref["accepted"]
            assert actual.coefficients == tuple(float(np.float32(t[0])) for t in ref["attempts"])
            assert actual.failures == (None,) * len(ref["attempts"])
            assert actual.loss == pytest.approx(ref["loss"], rel=LOSS_RTOL, abs=LOSS_ATOL)
            for trial, expected_trial in zip(actual.trials, ref["attempts"], strict=True):
                assert trial.accepted == (expected_trial[2] == "accepted")
                assert trial.loss == pytest.approx(expected_trial[1], rel=LOSS_RTOL, abs=LOSS_ATOL)
        state, run = result.state, result.run
        close(
            raw(context, run, state),
            expected[-1]["raw"] if count else np.broadcast_to(initial, train.offset.shape),
        )
        assert state.best_n_terms == (expected[-1]["best_terms"] if count else 0)
        assert len(run._states) == 1 and not run._proposals
        expected_bytes = sum(h.nbytes for record in run._prepared for h in ops._records[record][1])
        expected_bytes += 8 + 8 * (run.data.n_rows + run.validation_data.n_rows)
        expected_bytes += sum(t.tree.n_nodes * 24 for t in run._states[state].terms)
        assert context.metrics["live_bytes"] == expected_bytes
        run.close()
        assert context.metrics["live_bytes"] == 0


def test_ordered_rejection_then_acceptance_observes_one_round():
    data = NumericData([[0], [0]], [101, 301], ("x",))
    train = Problem(data, [[-1], [1]], data.row_ids, raw_width=2)
    with ExecutionContext() as context:
        result = recipes.normal(
            DeviceOperations(context),
            train,
            train,
            run_id="partial",
            seed=7,
            minimum_scale=np.e,
            rounds=1,
            max_depth=0,
            update="forward",
        )
        assert [s.accepted for s in result.steps] == [False, True]
        assert [len(s.trials) for s in result.steps] == [6, 1]
        assert result.state.version == 1 and result.state.n_terms == 1
        assert result.stop.completed_rounds == 1
        result.run.close()
        assert context.metrics["live_bytes"] == 0


def test_real_numeric_retry_keeps_every_trial_and_parent():
    data = NumericData([[0], [0]], [101, 301], ("x",))
    train = Problem(data, [[100], [100]], data.row_ids, raw_width=2)
    binned = Binning(("x",), (np.array([]),))
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        # Explicit public objective dependency: test a valid off-optimum base.
        configured = replace(
            objective.objective(), base=lambda ops, p: context.upload(np.zeros(2, np.float32))
        )
        run = DeviceRun(
            ops, train, train, run_id="numeric", seed=7, binning=binned, objective=configured
        )
        state = run.initialize()
        saved = raw(context, run, state)
        original = run.rng(0, "tree", "rows").integers(1000, size=8)
        fields_buffer = context.upload(np.tile([0, 1], (2, 1)).astype(np.float32))
        fields = ops.fields(
            run.data, fields_buffer, names=("gradient", "curvature"), roles=("training", "training")
        )
        terms = []
        for k, value in enumerate((100, 5000)):
            leaf = context.upload(np.array([value], np.float32))
            tree = trees.depthwise(
                ops, run.data, fields, binning=binned, max_depth=0, leaf=lambda *_, leaf=leaf: leaf
            )
            context.release(leaf)
            terms.append(DeviceTerm(tree, np.eye(2)[k : k + 1]))
        updated, trials = recipes.try_terms(run, state, terms, learning_rate=0.2)
        assert len(trials) == 6 and [t.accepted for t in trials] == [False] * 5 + [True]
        assert all(t.failure and t.loss is None for t in trials[:5]) and trials[-1].failure is None
        assert updated.version == 1 and updated.n_terms == 2
        assert not run._proposals
        close(raw(context, run, state), saved)
        np.testing.assert_array_equal(run.rng(0, "tree", "rows").integers(1000, size=8), original)
        with pytest.raises(ValueError, match="forged"):
            recipes.try_terms(run, replace(state), terms)
        for term in terms:
            ops.release(term.tree)
        ops.release(fields)
        context.release(fields_buffer)
        run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
def test_full_rejection_patience_and_outer_sweeps(update):
    train, validation, binned, _ = prepared_fixture("weighted")
    with ExecutionContext() as context:
        result = recipes.normal(
            DeviceOperations(context),
            train,
            validation,
            run_id="reject",
            seed=7,
            rounds=8,
            patience=2,
            learning_rate=0,
            update=update,
            binning=binned.binning,
            max_depth=1,
        )
        assert result.stop.reason == "patience" and result.stop.completed_rounds == 2
        assert len(result.steps) == (2 if update == "joint" else 4)
        assert all(not s.accepted and len(s.trials) == 6 for s in result.steps)
        assert result.state.version == result.state.n_terms == result.state.best_n_terms == 0
        result.run.close()
        assert context.metrics["live_bytes"] == 0


def test_bad_second_learner_fails_before_search_and_cleans_up(monkeypatch):
    train, validation, binned, _ = prepared_fixture("weighted")
    with ExecutionContext() as context:
        ops, calls, trials = DeviceOperations(context), [], []

        def learner(ops, data, fields):
            calls.append(1)
            if len(calls) == 2:
                context.upload(np.ones(3, np.float32))
                raise ValueError("invalid second learner")
            return trees.depthwise(ops, data, fields, binning=binned.binning, max_depth=1)

        monkeypatch.setattr(recipes, "try_terms", lambda *a, **k: trials.append(1))
        with pytest.raises(ValueError, match="invalid second learner"):
            recipes.normal(
                ops,
                train,
                validation,
                run_id="bad",
                seed=7,
                learner=learner,
                binning=binned.binning,
            )
        assert len(calls) == 2 and not trials
        assert context.metrics["live_bytes"] == 0 and not ops._records


def test_long_ordered_fit_and_fresh_cpu_inference(tmp_path):
    train, validation, binned, _ = prepared_fixture("weighted")
    with ExecutionContext() as context:
        result = recipes.normal(
            DeviceOperations(context),
            train,
            validation,
            run_id="long",
            seed=7,
            rounds=24,
            update="forward",
            step="fixed",
            max_depth=1,
            binning=binned.binning,
        )
        assert len(result.steps) == 48 and result.stop.completed_rounds == 24
        assert len(result.run._states) == 1 and not result.run._proposals
        expected = raw(context, result.run, result.state, True)
        path, inputs, output = (tmp_path / n for n in ("normal.json", "x.npy", "prediction.npy"))
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
                    "model=Model.load(sys.argv[1]); x=np.load(sys.argv[2])",
                    "data=NumericData(x, np.arange(len(x)), model.feature_names)",
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
