"""092-C Normal recipe comparison/stop anchors; real CUDA execution required."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import device_normal as normal
from openboost import device_recipes as recipes
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun
from openboost.execution import ExecutionContext
from openboost.stopping import StopState

from .test_device_comparison_consumers_cuda import owned_bytes, snapshot
from .test_loss_change import problem_for

pytestmark = pytest.mark.gpu


def configured_recipe(monkeypatch, base, means, update, *, failure=None):
    """Controlled valid initialization/leaves isolate the real recipe's consumers."""
    p = problem_for([0], [[0, 0]], [1])
    binning = Binning(("x",), (np.array([]),))
    factory = normal.objective
    comparisons, calls = [], []

    def configured(**kwargs):
        objective = factory(**kwargs)

        def compare(ops, problem, before, after):
            comparisons.append(1)
            if failure == "comparison" and len(comparisons) == 3:
                ops.execution.upload(np.ones(3, np.float32))
                raise RuntimeError("injected patience comparison failure")
            if failure == "result" and len(comparisons) == 3:
                return -1
            return objective.compare(ops, problem, before, after)

        return replace(
            objective,
            compare=compare,
            base=lambda ops, problem: ops.execution.upload(np.array(base, np.float32)),
        )

    monkeypatch.setattr(normal, "objective", configured)
    values, previous = [], np.float32(base[0])
    for mean in means:
        delta = np.float32(mean) - previous
        values.extend((0, delta) if update == "reverse" else (delta, 0))
        previous = np.float32(previous + delta)
    leaves = iter(values)

    def learner(ops, data, fields):
        calls.append(1)
        if failure == "learner" and len(calls) == 3:
            ops.execution.upload(np.ones(3, np.float32))
            raise ValueError("injected later learner failure")
        leaf = ops.execution.upload(np.array([next(leaves)], np.float32))
        return trees.depthwise(
            ops, data, fields, binning=binning, max_depth=0, leaf=lambda *_: leaf
        )

    return p, binning, learner


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
def test_recipe_current_best_and_patience_have_distinct_validation_anchors(monkeypatch, update):
    p, binning, learner = configured_recipe(monkeypatch, (2, 0), [3, 1.95, 1.97, 1.9, 1.89], update)
    observations = []
    observe = StopState.observe_change

    def tracked(stop, score, change):
        updated = observe(stop, score, change)
        observations.append((updated.completed_rounds, updated.stale_rounds, change))
        return updated

    monkeypatch.setattr(StopState, "observe_change", tracked)
    with ExecutionContext() as context:
        result = recipes.normal(
            DeviceOperations(context),
            p,
            p,
            run_id="three-anchors",
            seed=7,
            learner=learner,
            binning=binning,
            rounds=5,
            patience=5,
            min_delta=0.2,
            learning_rate=1,
            step="fixed",
            update=update,
        )
        assert result.run.comparison == "objective"
        assert [(r, s) for r, s, _ in observations] == list(enumerate([1, 2, 3, 4, 0], 1))
        assert result.stop.reason == "budget" and result.state.best_n_terms == 10
        observed_steps = [step for step in result.steps if step.validation_change is not None]
        assert len(observed_steps) == 5
        assert [s.validation_change for s in observed_steps] == [
            change for _, _, change in observations
        ]
        best_scores = [s.best_score for s in observed_steps]
        expected = np.array([2, 1.95, 1.95, 1.9, 1.89], np.float64) ** 2 / 2 + 0.5 * np.log(
            2 * np.pi
        )
        np.testing.assert_allclose(best_scores, expected, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            snapshot(result.run, result.state, validation=True, best=True),
            [[1.89, 0]],
            rtol=0,
            atol=2e-7,
        )
        assert all(t.comparison is not None for s in result.steps for t in s.trials)
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
def test_equal_score_improvements_replace_patience_anchor(monkeypatch, update):
    tiny = 2.0**-30
    p, binning, learner = configured_recipe(monkeypatch, (2 * tiny, 0), [tiny, 0], update)
    with ExecutionContext() as context:
        result = recipes.normal(
            DeviceOperations(context),
            p,
            p,
            run_id="tiny-recipe",
            seed=7,
            learner=learner,
            binning=binning,
            rounds=2,
            patience=1,
            learning_rate=1,
            update=update,
        )
        assert result.stop.reason == "budget" and result.stop.stale_rounds == 0
        assert result.state.best_score == result.stop.reference_score == result.stop.last_score
        observations = [
            s.validation_change for s in result.steps if s.validation_change is not None
        ]
        assert len(observations) == 2 and all(c.improves() for c in observations)
        # The second change must use the replaced tiny anchor, even though every
        # absolute reporting score is identical. Using the initial anchor gives
        # a change four times larger and cannot satisfy this enclosure check.
        assert observations[1].lower <= -tiny * tiny / 2 <= observations[1].upper
        np.testing.assert_array_equal(snapshot(result.run, result.state), [[0, 0]])
        assert len(result.run._states) == 1 and not result.run._proposals
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
def test_full_rejection_observes_once_per_outer_sweep(monkeypatch, update):
    p, binning, learner = configured_recipe(monkeypatch, (1, 0), [1] * 8, update)
    with ExecutionContext() as context:
        result = recipes.normal(
            DeviceOperations(context),
            p,
            p,
            run_id="reject-recipe",
            seed=7,
            learner=learner,
            binning=binning,
            rounds=8,
            patience=2,
            update=update,
        )
        assert result.stop.reason == "patience" and result.stop.completed_rounds == 2
        assert result.state.version == 0
        assert all(not s.accepted and len(s.trials) == 6 for s in result.steps)
        assert all(t.comparison.unchanged for s in result.steps for t in s.trials)
        assert len([s for s in result.steps if s.validation_change is not None]) == 2
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize(
    "failure", ["initial_anchor", "observation_copy", "comparison", "result", "learner"]
)
def test_patience_copy_comparison_and_later_failure_release_every_owner(monkeypatch, failure):
    p, binning, learner = configured_recipe(monkeypatch, (2, 0), [1, 0], "joint", failure=failure)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        if failure in ("initial_anchor", "observation_copy"):
            raw, calls = DeviceRun.raw, 0
            number = 1 if failure == "initial_anchor" else 2

            def fail(run, record, *, validation=False, best=False):
                nonlocal calls
                if validation and not best:
                    calls += 1
                    if calls == number:
                        with monkeypatch.context() as patch:

                            def failed_copy(*args):
                                raise MemoryError("injected patience copy failure")

                            patch.setattr(context, "copy", failed_copy)
                            return raw(run, record, validation=validation, best=best)
                return raw(run, record, validation=validation, best=best)

            monkeypatch.setattr(DeviceRun, "raw", fail)
        error, message = {
            "initial_anchor": (MemoryError, "injected patience copy"),
            "observation_copy": (MemoryError, "injected patience copy"),
            "comparison": (RuntimeError, "injected patience comparison"),
            "result": (TypeError, "LossChange"),
            "learner": (ValueError, "injected later learner"),
        }[failure]
        with pytest.raises(error, match=message):
            recipes.normal(
                ops,
                p,
                p,
                run_id="failed-recipe",
                seed=7,
                learner=learner,
                binning=binning,
                rounds=2,
                learning_rate=1,
                step="fixed",
            )
        assert context.metrics["live_bytes"] == 0 and not ops._records


def test_zero_round_recipe_releases_temporary_patience_storage(monkeypatch):
    p, binning, learner = configured_recipe(monkeypatch, (1, 0), [], "joint")
    with ExecutionContext() as context:
        result = recipes.normal(
            DeviceOperations(context),
            p,
            p,
            run_id="zero-recipe",
            seed=7,
            learner=learner,
            binning=binning,
            rounds=0,
        )
        assert result.steps == () and result.stop.reason == "budget"
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0
