"""106 resident GLM contracts: collect locally, run only on real CUDA hardware."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import ClassSchema, NumericData, Problem
from openboost import device_glm as glm
from openboost import device_objectives as operations
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.device_glm import (
    ATOL,
    DOMAIN_CASES,
    GEOMETRY_ATOL,
    GEOMETRY_RTOL,
    RTOL,
    base,
    geometry,
)
from .test_device_glm_reference import fixture

pytestmark = pytest.mark.gpu


def prepared(ops, p, family):
    binned = Binning.fit(p.data, bins=8).transform(p.data)
    return getattr(glm, family)().prepare(ops, ops.prepare(binned, p), p)


def snapshot(ops):
    return ops.execution.metrics["live_bytes"], set(ops._records), set(ops.execution._buffers)


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_base_geometry_fields_loss_and_owned_stream(family):
    p = fixture(family)
    exposure = p.structure.get("exposure")
    exposure = None if exposure is None else exposure[:, 0]
    expected_base = base(family, p.target[:, 0], p.offset[:, 0], p.weight, exposure)
    objective = getattr(glm, family)()
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p, family)
        before = dict(context.metrics)
        initial = objective.base(ops, problem)
        raw = operations.broadcast(ops, initial, len(p.target))
        matrix = glm.geometry(ops, problem, raw, family=family)
        gradient = objective.gradient(ops, problem, raw)
        fields = objective.fields(ops, problem, raw)
        score = objective.loss(ops, problem, raw)
        after = dict(context.metrics)
        assert after["upload_bytes"] == before["upload_bytes"]
        assert after["export_bytes"] - before["export_bytes"] == (
            after["validation_export_bytes"] - before["validation_export_bytes"] + 8
        )
        assert after["metric_export_bytes"] - before.get("metric_export_bytes", 0) == 8
        np.testing.assert_allclose(context.export(initial), [expected_base], rtol=RTOL, atol=ATOL)
        expected = geometry(
            family, context.export(raw)[:, 0], p.target[:, 0], p.offset[:, 0], p.weight, exposure
        )
        expected_matrix = np.column_stack(expected[1:])
        np.testing.assert_allclose(
            context.export(matrix), expected_matrix, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL
        )
        np.testing.assert_array_equal(context.export(gradient), context.export(matrix)[:, 0])
        np.testing.assert_allclose(
            context.export(fields.values),
            expected_matrix * p.weight[:, None],
            rtol=GEOMETRY_RTOL,
            atol=GEOMETRY_ATOL,
        )
        assert score == pytest.approx(expected[0], rel=GEOMETRY_RTOL, abs=GEOMETRY_ATOL)
        assert fields.names == ("gradient", "curvature")
        assert fields.roles == ("training", "training")
        with pytest.raises(ValueError, match="already applied"):
            ops.apply_weight(fields)
        context.release(matrix)
        context.release(raw)
        ops.release(problem)
        np.testing.assert_allclose(
            context.export(gradient), expected[1], rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL
        )
        np.testing.assert_allclose(
            context.export(fields.values),
            expected_matrix * p.weight[:, None],
            rtol=GEOMETRY_RTOL,
            atol=GEOMETRY_ATOL,
        )


@pytest.mark.parametrize("zero_weight", [False, True])
@pytest.mark.parametrize(
    "family,name,raw,target,offset,exposure,valid", DOMAIN_CASES, ids=[r[1] for r in DOMAIN_CASES]
)
def test_stored_domains_and_atomic_failure(
    family, name, raw, target, offset, exposure, valid, zero_weight
):
    data = NumericData([[0], [1]], [0, 1], ("x",))
    p = Problem(
        data,
        [[target], [1]],
        data.row_ids,
        offset=[[offset], [0]],
        weight=[0 if zero_weight else 1, 1],
        classes=ClassSchema(("no", "yes")) if family == "binary" else None,
        structure={"exposure": [[exposure], [1]]} if family == "poisson" else None,
    )
    objective = getattr(glm, family)()
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        if name in ("lost_count", "lost_exposure"):
            binned = Binning.fit(p.data, bins=8).transform(p.data)
            device_data = ops.prepare(binned, p)
            before = snapshot(ops)
            with pytest.raises(ValueError, match="float32"):
                objective.prepare(ops, device_data, p)
            assert snapshot(ops) == before
            return
        problem = prepared(ops, p, family)
        values = context.upload(np.array([[raw], [0]], np.float32))
        before = snapshot(ops)
        calls = (
            lambda: glm.geometry(ops, problem, values, family=family),
            lambda: objective.gradient(ops, problem, values),
            lambda: objective.fields(ops, problem, values),
            lambda: objective.loss(ops, problem, values),
        )
        if not valid:
            for call in calls:
                with pytest.raises(ValueError, match="float32 support"):
                    call()
                assert snapshot(ops) == before
            np.testing.assert_array_equal(context.export(values), [[raw], [0]])
        else:
            expected = geometry(
                family, [raw, 0], p.target[:, 0], p.offset[:, 0], p.weight, [exposure, 1]
            )
            actual = context.export(calls[0]())
            np.testing.assert_allclose(
                actual, np.column_stack(expected[1:]), rtol=GEOMETRY_RTOL, atol=0
            )  # Tiny positive tails cannot disappear.
            assert objective.loss(ops, problem, values) == pytest.approx(
                expected[0], rel=GEOMETRY_RTOL, abs=GEOMETRY_ATOL
            )


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_family_identity_shape_and_failure_recovery(family):
    p, other = fixture(family), "poisson" if family == "binary" else "binary"
    objective = getattr(glm, family)()
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p, family)
        before = snapshot(ops)
        for wrong_ops, wrong_problem in (
            (ops, replace(problem)),
            (DeviceOperations(context), problem),
        ):
            with pytest.raises(ValueError, match="forged"):
                objective.base(wrong_ops, wrong_problem)
        for wrong in (getattr(glm, other)(), operations.SQUARED):
            with pytest.raises(ValueError, match="family"):
                wrong.base(ops, problem)
        assert snapshot(ops) == before
        values = context.upload(np.zeros((8, 2), np.float32))
        with pytest.raises(ValueError, match="shape"):
            objective.loss(ops, problem, values)
        raw = operations.broadcast(ops, objective.base(ops, problem), 8)
        assert np.isfinite(objective.loss(ops, problem, raw))
        ops.release(problem)
        with pytest.raises(ValueError, match="released"):
            objective.loss(ops, problem, raw)


@pytest.mark.parametrize("family", ["binary", "poisson"])
@pytest.mark.parametrize("operation", ["base", "gradient", "fields", "loss"])
def test_partial_dispatch_failure_rolls_back_new_outputs(family, operation, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        objective = getattr(glm, family)()
        problem = prepared(ops, fixture(family), family)
        raw = operations.broadcast(ops, objective.base(ops, problem), 8)
        before = snapshot(ops)
        launch = ops._launch

        def fail(name, *args):
            if name.startswith(family) or name == "glm_gradient":
                raise RuntimeError("injected GLM dispatch failure")
            return launch(name, *args)

        with monkeypatch.context() as patch:
            patch.setattr(ops, "_launch", fail)
            with pytest.raises(RuntimeError, match="injected"):
                getattr(objective, operation)(
                    ops, problem, *(() if operation == "base" else (raw,))
                )
        assert snapshot(ops) == before
        assert np.isfinite(objective.loss(ops, problem, raw))


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_initialization_extremes_and_zero_weight_rows(family):
    p = fixture(family)
    cases = [p]
    if family == "poisson":
        cases += [
            replace(p, offset=p.offset + 800),
            replace(p, target=np.where(p.weight[:, None] > 0, 0, 9)),
        ]
    else:
        cases += [replace(p, weight=(p.target[:, 0] == 0).astype(float))]
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        objective = getattr(glm, family)()
        for case in cases:
            problem = prepared(ops, case, family)
            expected = base(
                family,
                case.target[:, 0],
                case.offset[:, 0],
                case.weight,
                None if family == "binary" else case.structure["exposure"][:, 0],
            )
            np.testing.assert_allclose(
                context.export(objective.base(ops, problem)), [expected], rtol=RTOL, atol=ATOL
            )
        if family == "binary":
            single = prepared(ops, replace(p, target=np.zeros((8, 1))), family)
            before = snapshot(ops)
            with pytest.raises(ValueError):
                objective.base(ops, single)
            assert snapshot(ops) == before
