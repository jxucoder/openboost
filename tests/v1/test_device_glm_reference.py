"""106 GLM oracle/domain and complete-scope controls; no GPU emulation."""

import copy

import numpy as np
import pytest

from openboost import ClassSchema, NumericData, Problem
from openboost.objectives import Binary, Poisson

from .reference.device_glm import (
    CHECKS,
    DOMAIN_CASES,
    REQUIRED_CELLS,
    base,
    complete_cells,
    geometry,
)


def fixture(family, *, validation=False):
    x = np.array([0, 1, 2, 3, 4, np.nan, 6, 7])[:, None]
    ids = np.arange(8) + (100 if validation else 0)
    data = NumericData(x, ids, ("x",))
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0] if family == "binary" else [0, 1, 0, 4, 7, 2, 8, 3])
    offset = (np.arange(8) % 3 - 1) / 8
    weight = np.array([1, 2, 1, 3, 0, 1, 2, 1])
    return Problem(
        data,
        y[:, None],
        ids,
        weight=weight,
        offset=offset[:, None],
        classes=ClassSchema(("no", "yes")) if family == "binary" else None,
        structure={"exposure": (np.arange(8) % 4 + 1)[:, None] / 2}
        if family == "poisson"
        else None,
    )


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_oracle_matches_cpu_and_finite_difference(family):
    p = fixture(family)
    e = p.structure.get("exposure")
    e = None if e is None else e[:, 0]
    expected_base = base(family, p.target[:, 0], p.offset[:, 0], p.weight, e)
    objective = Binary if family == "binary" else Poisson()
    assert expected_base == pytest.approx(objective.base(p)[0], abs=1e-6)
    raw = np.full((8, 1), expected_base, np.float32)
    value, g, h = geometry(family, raw[:, 0], p.target[:, 0], p.offset[:, 0], p.weight, e)
    expected = objective.geometry(p, raw)
    assert value == pytest.approx(expected[0], rel=1e-6, abs=1e-8)
    np.testing.assert_allclose(g, expected[1], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(h, expected[2], rtol=1e-6, atol=1e-8)
    for row in (0, 3, 6):
        step = np.zeros_like(raw, dtype=float)
        step[row] = 1e-4
        plus, minus = objective.loss(p, raw + step), objective.loss(p, raw - step)
        mass = p.weight[row] / p.weight.sum()
        assert (plus - minus) / 2e-4 == pytest.approx(g[row] * mass, rel=1e-5, abs=1e-8)
        assert (plus + minus - 2 * value) / 1e-8 == pytest.approx(h[row] * mass, rel=2e-5, abs=1e-7)


@pytest.mark.parametrize(
    "family,name,raw,target,offset,exposure,valid", DOMAIN_CASES, ids=[r[1] for r in DOMAIN_CASES]
)
def test_preregistered_stored_domain(family, name, raw, target, offset, exposure, valid):
    args = family, [raw], [target], [offset], [1], [exposure]
    if not valid:
        with pytest.raises(ValueError):
            geometry(*args)
    else:
        _, g, h = geometry(*args)
        assert h[0] > 0
        if name in ("positive_tail", "negative_tail"):
            assert abs(g[0]) == pytest.approx(np.exp(-80), rel=1e-6, abs=0)


def test_poisson_initializer_handles_large_offsets_and_zero_weight_counts():
    p = fixture("poisson")
    shifted = Problem(
        p.data, p.target, p.row_ids, weight=p.weight, offset=p.offset + 800, structure=p.structure
    )
    e = p.structure["exposure"][:, 0]
    for case in (p, shifted):
        actual = base("poisson", case.target[:, 0], case.offset[:, 0], case.weight, e)
        assert actual == pytest.approx(Poisson().base(case)[0], rel=1e-6, abs=1e-5)
    assert base("poisson", [0, 9], [0, 0], [1, 0], [1, 2]) == pytest.approx(np.log(1e-6), rel=1e-6)


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_weights_do_not_enter_unweighted_geometry(family):
    args = (family, [0, 1], [0, 1], [0.25, -0.25])
    _, g, h = geometry(*args, [1, 1], [1, 2])
    _, weighted_g, weighted_h = geometry(*args, [2, 3], [1, 2])
    np.testing.assert_array_equal(weighted_g, g)
    np.testing.assert_array_equal(weighted_h, h)
    assert not np.array_equal(g * [2, 3], g * [4, 9])


def test_poisson_exposure_enters_mean_once():
    _, g, h = geometry("poisson", [0], [1], [0], [1], [2])
    np.testing.assert_array_equal(g, [1])
    np.testing.assert_array_equal(h, [2])
    _, twice_g, _ = geometry("poisson", [0], [1], [0], [1], [4])
    assert twice_g[0] != g[0]


def complete_record():
    return {
        k: dict(backend="cuda", passed=True, skipped=0, cpu_fallback=False, checks=list(CHECKS))
        for k in REQUIRED_CELLS
    }


def test_complete_scope_judge_control_is_not_device_evidence():
    assert complete_cells(complete_record())


@pytest.mark.parametrize("cell", REQUIRED_CELLS)
def test_no_required_cell_can_be_dropped(cell):
    records = complete_record()
    del records[cell]
    assert not complete_cells(records)


@pytest.mark.parametrize(
    "fault",
    [
        dict(skipped=1),
        dict(cpu_fallback=True),
        dict(backend="cpu"),
        dict(passed=False),
        dict(checks=["geometry"]),
    ],
)
def test_partial_or_fallback_cannot_pass_scope(fault):
    records = copy.deepcopy(complete_record())
    records["R4/poisson"].update(fault)
    assert not complete_cells(records)
