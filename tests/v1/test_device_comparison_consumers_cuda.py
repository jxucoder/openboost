"""092-C resident consumer/ownership cohort. Requires real CUDA; never emulated."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import device_normal as normal
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations, _workspace
from openboost.device_recipes import try_terms
from openboost.device_runtime import DeviceRun, DeviceTerm
from openboost.execution import ExecutionContext
from tests.v1.test_loss_change import CASES, problem_for

pytestmark = pytest.mark.gpu


def start(context, base=(2, 0), *, comparison="objective", train=None, validation=None):
    if train is None:
        train = problem_for([0], [[0, 0]], [1])
    if validation is None:
        validation = train
    configured = replace(
        normal.objective(), base=lambda ops, p: context.upload(np.array(base, np.float32))
    )
    return DeviceRun(
        DeviceOperations(context),
        train,
        validation,
        run_id="092-consumer",
        seed=7,
        binning=Binning(("x",), (np.array([]),)),
        objective=configured,
        comparison=comparison,
    )


def constant_terms(run, values):
    """Real resident trees from explicit leaves, independent of gradient fitting."""
    with _workspace(run.ops) as retained:
        buffer = run.execution.upload(np.tile([0, 1], (run.data.n_rows, 1)).astype(np.float32))
        fields = run.ops.fields(
            run.data, buffer, names=("gradient", "curvature"), roles=("training", "training")
        )
        terms = []
        for k, value in enumerate(values):
            leaf = run.execution.upload(np.array([value], np.float32))
            tree = trees.depthwise(
                run.ops,
                run.data,
                fields,
                binning=run.binning,
                max_depth=0,
                leaf=lambda *_, leaf=leaf: leaf,
            )
            terms.append(DeviceTerm(tree, np.eye(2)[k : k + 1]))
            retained.add(tree)
    return tuple(terms)


def snapshot(run, state, *, validation=False, best=False):
    handle = run.raw(state, validation=validation, best=best)
    try:
        return run.execution.export(handle)
    finally:
        run.execution.release(handle)


def owned_bytes(run, state):
    total = sum(h.nbytes for record in run._prepared for h in run.ops._records[record][1])
    storage = run._states[state]
    total += run._base.nbytes + sum(h.nbytes for h in storage.raw)
    total += 0 if storage.best_raw is None else storage.best_raw.nbytes
    total += sum(term.tree.n_nodes * 24 for term in storage.terms)
    return total


@pytest.mark.parametrize("policy", ["reported", "objective"])
def test_named_policy_distinguishes_equal_reporting_losses_and_owned_best(policy):
    with ExecutionContext() as context:
        run = start(context, (2.0**-30, 0), comparison=policy)
        state = run.initialize()
        assert run.comparison == policy and run.validation_problem.data is run.validation_data
        assert context.metrics["live_bytes"] == owned_bytes(run, state)
        if policy == "objective":
            assert run._states[state].best_raw.nbytes == 8
        else:
            assert run._states[state].best_raw is None
            with pytest.raises(ValueError, match="objective"):
                run.raw(state, validation=True, best=True)
        terms = constant_terms(run, (-(2.0**-30), 0))
        proposal = run.propose_terms(state, terms)
        assert proposal.loss == state.loss and proposal.validation_score == state.best_score
        assert run.compare(state, proposal).improves()
        before = dict(context.metrics)
        assert run.resolve(state, proposal, accept=False) is state
        assert context.metrics["live_bytes"] == before["live_bytes"]
        assert context.metrics["device_copy_bytes"] == before["device_copy_bytes"]
        run.release(proposal)
        updated, trials = try_terms(run, state, terms, learning_rate=1, max_trials=1)
        assert (updated is not state) == (policy == "objective")
        assert trials[0].accepted == (policy == "objective")
        assert (trials[0].comparison is not None) == (policy == "objective")
        if policy == "objective":
            assert updated.best_n_terms == updated.n_terms == 2
            assert updated.best_score == state.best_score
            np.testing.assert_array_equal(
                snapshot(run, updated, validation=True, best=True), [[0, 0]]
            )
            run.release(state)
        for term in terms:
            run.ops.release(term.tree)
        assert context.metrics["live_bytes"] == owned_bytes(run, updated)
        run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_captured_run7_worsening_is_rejected_at_actual_stored_raw(order):
    case = next(c for c in CASES if c["id"] == f"run7/{order}/channel0/alpha4.0/training")
    a = {key: np.array(value["values"], np.float32) for key, value in case["inputs"].items()}
    p = problem_for(a["target"], a["offset"], a["weight"])
    with ExecutionContext() as context:
        run = start(context, a["before"][0], train=p)
        state = run.initialize()
        terms = constant_terms(run, a["after"][0] - a["before"][0])
        proposal = run.propose_terms(state, terms)
        np.testing.assert_array_equal(snapshot(run, state), a["before"])
        np.testing.assert_array_equal(snapshot(run, proposal), a["after"])
        assert proposal.loss < state.loss  # Preserve the measured false improvement.
        assert run.compare(state, proposal).status == "worsening"
        run.release(proposal)
        updated, trials = try_terms(run, state, terms, learning_rate=1, max_trials=1)
        assert updated is state and not trials[0].accepted
        assert trials[0].comparison.status == "worsening" and trials[0].failure is None
        for term in terms:
            run.ops.release(term.tree)
        run.close()
        assert context.metrics["live_bytes"] == 0


def test_best_anchor_is_independently_copied_across_retained_states_and_releases():
    with ExecutionContext() as context:
        run = start(context)
        state = run.initialize()
        states = [state]
        raw_values, best_values = [2], [2]
        for mean, best in zip((3, 1.95, 1.97, 1.9, 1.89), (2, 1.95, 1.95, 1.9, 1.89), strict=True):
            terms = constant_terms(run, (np.float32(mean) - snapshot(run, state)[0, 0], 0))
            proposal = run.propose_terms(state, terms)
            prior = state
            state = run.resolve(prior, proposal, accept=True)
            run.release(proposal)
            for term in terms:
                run.ops.release(term.tree)
            assert state.best_n_terms == (prior.best_n_terms if mean != best else state.n_terms)
            states.append(state)
            raw_values.append(mean)
            best_values.append(best)
        handles = [run._states[s].best_raw for s in states]
        assert len(set(handles)) == len(states)
        assert all(h not in run._states[s].raw for s, h in zip(states, handles, strict=True))
        for old, current, best in zip(states, raw_values, best_values, strict=True):
            np.testing.assert_allclose(snapshot(run, old), [[current, 0]], rtol=0, atol=2e-7)
            np.testing.assert_allclose(
                snapshot(run, old, validation=True, best=True), [[best, 0]], rtol=0, atol=2e-7
            )
            np.testing.assert_allclose(
                run.export(old, best=True).predict(problem_for([0], [[0, 0]], [1]).data),
                [[best, 0]],
                rtol=0,
                atol=2e-7,
            )
        for old in states[:-1]:
            run.release(old)
        np.testing.assert_allclose(
            snapshot(run, state, validation=True, best=True), [[1.89, 0]], rtol=0, atol=2e-7
        )
        assert context.metrics["live_bytes"] == owned_bytes(run, state)
        run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("failure", ["comparison", "copy1", "copy2", "copy3"])
def test_failed_best_resolution_restores_ownership_serial_and_rng(failure, monkeypatch):
    with ExecutionContext() as context:
        run = start(context)
        state = run.initialize()
        terms = constant_terms(run, (-1, 0))
        proposal = run.propose_terms(state, terms)
        buffers, records = set(context._buffers), set(run.ops._records)
        states, proposals, serial = set(run._states), set(run._proposals), run._serial
        references = [t.references for t in run._proposals[proposal].terms]
        rng = run.rng(0, "tree", "rows").random(4)
        with monkeypatch.context() as patch:
            if failure == "comparison":

                def fail(*args):
                    context.upload(np.ones(2, np.float32))
                    raise RuntimeError("injected comparison failure")

                patch.setattr(run, "objective", replace(run.objective, compare=fail))
            else:
                copy, calls = context.copy, 0

                def fail(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == int(failure[-1]):
                        raise MemoryError("injected best-copy failure")
                    return copy(*args, **kwargs)

                patch.setattr(context, "copy", fail)
            with pytest.raises((RuntimeError, MemoryError), match="injected"):
                run.resolve(state, proposal, accept=True)
        assert set(context._buffers) == buffers and set(run.ops._records) == records
        assert (
            set(run._states) == states
            and set(run._proposals) == proposals
            and run._serial == serial
        )
        assert [t.references for t in run._proposals[proposal].terms] == references
        np.testing.assert_array_equal(snapshot(run, state, validation=True, best=True), [[2, 0]])
        np.testing.assert_array_equal(run.rng(0, "tree", "rows").random(4), rng)
        updated = run.resolve(state, proposal, accept=True)
        run.release(state)
        run.release(proposal)
        for term in terms:
            run.ops.release(term.tree)
        np.testing.assert_array_equal(snapshot(run, updated, validation=True, best=True), [[1, 0]])
        run.close()
        assert context.metrics["live_bytes"] == 0


def test_failed_initial_best_copy_and_parent_bound_comparison(monkeypatch):
    with ExecutionContext() as context:
        run = start(context)
        buffers, records = set(context._buffers), set(run.ops._records)
        with monkeypatch.context() as patch:

            def fail(*args):
                raise MemoryError("injected initialization copy failure")

            patch.setattr(context, "copy", fail)
            with pytest.raises(MemoryError, match="injected"):
                run.initialize()
        assert set(context._buffers) == buffers and set(run.ops._records) == records
        assert run._serial == 0 and not run._states
        state = run.initialize()
        terms = constant_terms(run, (-1, 0))
        proposal = run.propose_terms(state, terms)
        updated = run.resolve(state, proposal, accept=True)
        with pytest.raises(ValueError, match="parent"):
            run.compare(updated, proposal)
        with pytest.raises(ValueError, match="forged"):
            run.compare(replace(state), proposal)
        with pytest.raises(ValueError, match="validation"):
            run.raw(state, best=True)
        buffers, records, serial = set(context._buffers), set(run.ops._records), run._serial
        with monkeypatch.context() as patch:

            def fail(*args):
                context.upload(np.ones(2, np.float32))
                raise RuntimeError("injected callback failure")

            patch.setattr(run, "objective", replace(run.objective, compare=fail))
            with pytest.raises(RuntimeError, match="injected"):
                run.compare(state, proposal)
        assert set(context._buffers) == buffers and set(run.ops._records) == records
        assert run._serial == serial
        run.release(proposal)
        with pytest.raises(ValueError, match="released"):
            run.compare(state, proposal)
        for term in terms:
            run.ops.release(term.tree)
        run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("step", ["fixed", "backtracking"])
def test_unresolved_objective_evidence_has_no_reported_loss_fallback(step):
    with ExecutionContext() as context:
        run = start(context, (0, 33))
        state = run.initialize()
        terms = constant_terms(run, (1, 0))
        updated, trials = try_terms(run, state, terms, learning_rate=1, step=step)
        assert (updated is not state) == (step == "fixed")
        assert len(trials) == (1 if step == "fixed" else 6)
        assert all(t.comparison.status == "unresolved" for t in trials)
        assert all(t.failure is None for t in trials)
        assert updated.best_n_terms == 0
        for term in terms:
            run.ops.release(term.tree)
        run.close()
        assert context.metrics["live_bytes"] == 0
