"""110 independent multiclass domain, geometry and CPU semantic controls."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import ClassSchema, NumericData, Problem
from openboost.objectives import Multiclass

from .reference.device_multiclass import DOMAIN_CASES, base, geometry


def fixture(width=3, *, validation=False):
    n = width * 3
    x = np.arange(n, dtype=float)[:, None]
    x[-2] = np.nan
    ids = np.arange(n) + (100 if validation else 0)
    order = np.roll(np.arange(n)[::-1], 2) if validation else np.arange(n)
    data = NumericData(x[order], ids, ("x",))
    return Problem(
        data,
        (np.arange(n) // 3)[order, None],
        ids,
        raw_width=width,
        classes=ClassSchema(tuple(f"class-{i}" for i in range(width))),
        weight=(np.arange(n) % 4 + 1) * (2 if validation else 1),
        offset=((np.arange(n * width).reshape(n, width) * 7 % 11 - 5) / 16)[order]
        * (0.5 if validation else 1),
    )


@pytest.mark.parametrize("width", [2, 3, 5])
def test_cpu_geometry_matches_decimal_gradient_and_hessian_bound(width):
    p = fixture(width)
    raw = np.linspace(-0.8, 1.2, len(p.target) * width).reshape(-1, width)
    expected = geometry(raw, p.target[:, 0], p.offset, p.weight, stored=False)
    loss, g, h = Multiclass.geometry(p, raw)
    assert loss == pytest.approx(expected[0], rel=1e-14)
    np.testing.assert_allclose(g, expected[1], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(h, expected[2], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(g.sum(axis=1), 0, atol=5e-16)
    for row in (0, 3):
        hessian = expected[4][row]
        np.testing.assert_allclose(hessian @ np.ones(width), 0, atol=1e-16)
        assert np.linalg.eigvalsh(np.diag(h[row]) - hessian).min() >= -1e-15
        delta = np.zeros_like(raw)
        delta[row, 1] = 1e-4
        plus, minus = Multiclass.loss(p, raw + delta), Multiclass.loss(p, raw - delta)
        mass = p.weight[row] / p.weight.sum()
        assert (plus - minus) / 2e-4 == pytest.approx(g[row, 1] * mass, rel=1e-7)
        assert (plus + minus - 2 * loss) / 1e-8 == pytest.approx(hessian[1, 1] * mass, abs=1e-7)


@pytest.mark.parametrize("code", [0, 1, 2])
def test_cpu_dominant_class_tail_does_not_vanish(code):
    p = replace(fixture(), target=np.full((9, 1), code), offset=np.zeros((9, 3)))
    raw = np.tile([80, 0, 0], (9, 1))
    expected = geometry(raw, p.target[:, 0], p.offset, p.weight, stored=False)
    actual = Multiclass.geometry(p, raw)
    assert actual[0] == pytest.approx(expected[0], rel=1e-14, abs=0)
    for a, b in zip(actual[1:], expected[1:3], strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=0)


@pytest.mark.parametrize("name,values,code,valid", DOMAIN_CASES, ids=[c[0] for c in DOMAIN_CASES])
@pytest.mark.parametrize("zero_weight", [False, True])
def test_stored_domains_include_zero_weight_rows(name, values, code, valid, zero_weight):
    args = [values, (0, 0, 0)], [code, 1], np.zeros((2, 3)), [int(not zero_weight), 1]
    if valid:
        _, g, h, _, _ = geometry(*args)
        assert (h > 0).all()
        if name in ("positive_tail", "shifted_tail"):
            assert g[0, 0] == pytest.approx(-2 * np.exp(-80), rel=1e-6, abs=0)
    else:
        with pytest.raises(ValueError, match="stored support"):
            geometry(*args)


def test_base_offsets_weights_class_permutation_and_common_shift():
    p = fixture()
    np.testing.assert_array_equal(base(p.target[:, 0], 3), Multiclass.base(p))
    for q in (p, replace(p, weight=(p.target[:, 0] == 0).astype(float))):
        np.testing.assert_array_equal(Multiclass.base(q), [0, 0, 0])
    with pytest.raises(ValueError, match="every declared class"):
        base([0, 2], 3)
    with pytest.raises(ValueError, match="every declared class"):
        Multiclass.base(replace(p, target=np.zeros_like(p.target)))
    raw = np.zeros((9, 3))
    original = Multiclass.geometry(p, raw)
    shifted = Multiclass.geometry(p, raw + 800)
    for a, b in zip(original, shifted, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-15)
    weighted = Multiclass.geometry(replace(p, weight=p.weight[::-1]), raw)
    for a, b in zip(original[1:], weighted[1:], strict=True):
        np.testing.assert_array_equal(a, b)
    permutation = np.array([2, 0, 1])
    q = replace(
        p, target=np.argsort(permutation)[p.target.astype(int)], offset=p.offset[:, permutation]
    )
    permuted = Multiclass.geometry(q, raw)
    assert permuted[0] == pytest.approx(original[0], rel=1e-14)
    for a, b in zip(permuted[1:], original[1:], strict=True):
        np.testing.assert_allclose(a, b[:, permutation], rtol=1e-14)
