"""090-C mapped Normal transactions; real hardware required, never emulated."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import device_normal as normal
from openboost import device_objectives as operations
from openboost import device_tree as trees
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun, DeviceTerm, map_update
from openboost.execution import ExecutionContext

from .reference.device_normal import rounds
from .reference.normal_precision import LOSS_ATOL, LOSS_RTOL
from .test_device_normal_cuda import close
from .test_device_normal_reference import prepared_fixture
from .test_device_runtime_cuda import raw
from .test_device_tree_cuda import cohort_minimum

pytestmark = pytest.mark.gpu


def start(context, case="weighted"):
    train, validation, binned, information = prepared_fixture(case)
    run = DeviceRun(
        DeviceOperations(context),
        train,
        validation,
        run_id="090",
        seed=7,
        binning=binned.binning,
        objective=normal.objective(),
    )
    return run, train, validation, information


def grow(run, state, channels=(0, 1), *, depth=1, mode="natural", damping=0, information=None):
    ops, context = run.ops, run.execution
    values = run.raw(state)
    gradient, fisher = normal.geometry(ops, run.problem, values)
    direction = operations.diagonal_direction(ops, gradient, fisher, mode=mode, damping=damping)
    result = []
    for k in channels:
        fields = operations.least_squares(ops, run.data, direction, k)
        if information is not None:
            for i, name in ((0, "cohort:red"), (1, "cohort:blue")):
                handle = context.upload(information[:, i].astype(np.float32))
                augmented = ops.add_independent(fields, name, handle, nonnegative=True)
                ops.release(fields)
                context.release(handle)
                fields = augmented
        tree = trees.depthwise(
            ops,
            run.data,
            fields,
            binning=run.binning,
            max_depth=depth,
            legality=cohort_minimum(1) if information is not None else None,
        )
        result.append(DeviceTerm(tree, np.eye(2)[k : k + 1]))
        ops.release(fields)
    for handle in (values, gradient, fisher, direction):
        context.release(handle)
    return tuple(result)


@pytest.mark.parametrize(
    "case,depth,minimum",
    [
        ("weighted", 1, None),
        ("d2", 1, None),
        ("d2", 2, 1),
        ("conflict", 0, None),
        ("conflict", 2, None),
    ],
)
@pytest.mark.parametrize("mode,damping", [("ordinary", 0), ("natural", 0), ("natural", 0.25)])
@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("fixed,rate", [(True, 0.1), (False, 8.0)])
def test_frozen_three_round_transactions(case, depth, minimum, mode, damping, update, fixed, rate):
    initial, expected = rounds(
        case,
        depth=depth,
        minimum=minimum,
        mode=mode,
        damping=damping,
        update=update,
        fixed=fixed,
        rate=rate,
    )
    with ExecutionContext() as context:
        run, train, validation, information = start(context, case)
        state = run.initialize()
        close(raw(context, run, state), np.broadcast_to(initial, train.offset.shape))
        for ref in expected:
            before = state
            terms = grow(
                run,
                state,
                ref["channels"],
                depth=depth,
                mode=mode,
                damping=damping,
                information=information if minimum is not None else None,
            )
            for term, nodes in zip(terms, ref["nodes"], strict=True):
                assert [None if f == -1 else (f, t, m) for f, t, m, _, _ in term.tree.topology] == [
                    n["key"] for n in nodes
                ]
                close(trees.export(run.ops, term.tree).value[:, 0], [n["value"] for n in nodes])
            coefficients = []
            for j in range(1 if fixed else 6):
                coefficient = rate * 0.5**j
                coefficients.append(coefficient)
                metrics = dict(context.metrics)
                proposal = run.propose_terms(state, terms, coefficient=coefficient)
                assert context.metrics["upload_bytes"] == metrics["upload_bytes"]
                accept = fixed or proposal.loss < state.loss
                updated = run.resolve(state, proposal, accept=accept)
                run.release(proposal)
                if accept:
                    run.release(state)
                    state = updated
                    break
                assert updated is state
            for term in terms:
                run.ops.release(term.tree)
            assert tuple(coefficients) == tuple(a[0] for a in ref["attempts"])
            assert (state is not before) == ref["accepted"]
            assert (state.version, state.n_terms, state.best_n_terms) == (
                ref["version"],
                ref["nterms"],
                ref["best_terms"],
            )
            close(raw(context, run, state), ref["raw"])
            close(raw(context, run, state, True), ref["validation_raw"])
            assert state.loss == pytest.approx(ref["loss"], rel=LOSS_RTOL, abs=LOSS_ATOL)
            assert state.validation_score == pytest.approx(
                ref["validation_score"], rel=LOSS_RTOL, abs=LOSS_ATOL
            )
            assert state.best_score == pytest.approx(
                ref["best_score"], rel=LOSS_RTOL, abs=LOSS_ATOL
            )
        close(run.export(state).predict(train.data), raw(context, run, state))
        close(run.export(state).predict(validation.data), raw(context, run, state, True))
        assert len(run._states) == 1 and not run._proposals
        run.close()
        assert context.metrics["live_bytes"] == 0


def test_joint_reject_accept_and_owned_tree_mapping_snapshots():
    with ExecutionContext() as context:
        run, train, _, _ = start(context)
        state = run.initialize()
        original = raw(context, run, state)
        rng = run.rng(0, "tree", "rows").integers(10000, size=12)
        terms = grow(run, state)
        mapping = np.array([[0.75, -0.125]], np.float32)
        custom = (DeviceTerm(terms[0].tree, mapping), terms[1])
        mapping[:] = 99
        proposal = run.propose_terms(state, custom, coefficient=0.1)
        candidate = raw(context, run, proposal)
        assert run.resolve(state, proposal, accept=False) is state
        close(raw(context, run, state), original)
        np.testing.assert_array_equal(run.rng(0, "tree", "rows").integers(10000, size=12), rng)
        for term in terms:
            run.ops.release(term.tree)
        accepted = run.resolve(state, proposal, accept=True)
        model = run.export(accepted)
        np.testing.assert_array_equal(model.terms[0].mapping, [[0.75, -0.125]])
        run.release(proposal)
        run.release(state)
        close(raw(context, run, accepted), candidate)
        close(model.predict(train.data), candidate)
        assert accepted.version == 1 and accepted.n_terms == 2
        run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("phase", ["second_tree", "second_prediction", "validation", "resolve"])
def test_partial_joint_failure_preserves_every_owner(phase, monkeypatch):
    with ExecutionContext() as context:
        run, _, _, _ = start(context)
        state = run.initialize()
        terms = grow(run, state)
        proposal = run.propose_terms(state, terms, coefficient=0.1)
        before = raw(context, run, state)
        buffers, records = set(context._buffers), set(run.ops._records)
        states, proposals, serial = set(run._states), set(run._proposals), run._serial
        live = context.metrics["live_bytes"]
        with monkeypatch.context() as patch:
            if phase == "validation":
                original = run._loss

                def fail(problem, values):
                    if problem.data is run.validation_data:
                        raise ValueError("injected validation failure")
                    return original(problem, values)

                patch.setattr(run, "_loss", fail)
            else:
                owner, name = (
                    (context, "copy")
                    if phase == "resolve"
                    else (trees, "copy" if phase == "second_tree" else "predict")
                )
                original = getattr(owner, name)
                calls = 0

                def fail(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise ValueError("injected partial failure")
                    return original(*args, **kwargs)

                patch.setattr(owner, name, fail)
            with pytest.raises(ValueError, match="injected"):
                if phase == "resolve":
                    run.resolve(state, proposal, accept=True)
                else:
                    run.propose_terms(state, terms, coefficient=0.1)
        assert set(context._buffers) == buffers and set(run.ops._records) == records
        assert set(run._states) == states and set(run._proposals) == proposals
        assert run._serial == serial and context.metrics["live_bytes"] == live
        close(raw(context, run, state), before)
        accepted = run.resolve(state, proposal, accept=True)
        run.release(state)
        run.release(proposal)
        for term in terms:
            run.ops.release(term.tree)
        run.export(accepted)
        run.close()
        assert context.metrics["live_bytes"] == 0


def test_term_schema_and_map_overflow_are_not_partial_updates():
    with ExecutionContext() as context:
        run, _, _, _ = start(context)
        state = run.initialize()
        terms = grow(run, state)
        for invalid in (
            (),
            (terms[0].tree,),
            (DeviceTerm(terms[0].tree, [[1]]),),
            (DeviceTerm(replace(terms[0].tree), [[1, 0]]),),
        ):
            with pytest.raises(ValueError):
                run.validate_terms(invalid)
        values = run.raw(state)
        delta = context.upload(np.full((run.data.n_rows, 1), 3e38, np.float32))
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError):
            map_update(run.ops, values, delta, [[1, 1]], 2)
        assert context.metrics["live_bytes"] == before
        close(context.export(values), raw(context, run, state))
        for term in terms:
            run.ops.release(term.tree)
        context.release(values)
        context.release(delta)
        run.close()
        assert context.metrics["live_bytes"] == 0
