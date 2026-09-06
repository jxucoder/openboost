"""Routed original-weight quantiles and D3 leaves against independent oracles."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import MixedData, Problem, RunContext
from openboost.binning import Binning
from openboost.leaves import ResidualContext, quantile_leaf
from openboost.objectives import Quantile
from openboost.recipes import quantile
from openboost.tree import best_first, depthwise, symmetric
from tests.v1.reference.author import penalized_quantile
from tests.v1.reference.mixed import Transformer, grow
from tests.v1.reference.quantile import weighted_quantile


def fixture():
    x = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [4, "c"], [None, "b"]],
        [10, 12, 14, 16, 18, 20],
        ("x", "c"),
        ("numeric", "categorical"),
    )
    return Problem(
        x,
        [[-5], [2], [8], [3], [-2], [15]],
        x.row_ids,
        weight=[1, 3, 2, 0, 4, 2],
        offset=np.arange(6.0)[:, None] / 4,
    )


@pytest.mark.parametrize("penalty", [0.0, 0.3, 5.0, 100.0])
def test_leaf_optimum_matches_independent_enumeration(penalty):
    p = fixture()
    rng = np.random.default_rng(29)
    for _ in range(20):
        residual = rng.normal(size=6)
        view = ResidualContext(p, residual).view(np.arange(6))
        anchor = -0.7 if penalty else 0
        got = quantile_leaf(view, q=0.7, penalty=penalty, anchor=anchor)
        expected = (
            penalized_quantile(residual, 0.7, p.weight, penalty=penalty, anchor=anchor)
            if penalty
            else weighted_quantile(residual, 0.7, p.weight)
        )
        np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("policy", [depthwise, best_first, symmetric])
@pytest.mark.parametrize("penalty", [0.0, 2.0])
def test_three_round_routed_leaves_match_independent_tree(policy, penalty):
    p = fixture()
    q, anchor = 0.7, (-1 if penalty else 0)
    objective = Quantile(q)
    fit = quantile(
        p,
        p,
        context=RunContext("quantile", 1),
        q=q,
        penalty=penalty,
        anchor=anchor,
        rounds=3,
        bins=4,
        grower=policy,
    )
    transform = Transformer.fit(
        p.data.values, names=p.data.feature_names, kinds=p.data.feature_kinds, bins=4
    )
    raw = np.full((6, 1), weighted_quantile((p.target - p.offset)[:, 0], q, p.weight))
    for step in fit.steps:
        residual = (p.target - p.with_offset(raw))[:, 0]
        tree = grow(
            p.data.values,
            ((residual < 0).astype(float) - q)[:, None],
            np.ones((6, 1)),
            transform,
            weight=p.weight,
            policy=policy.__name__,
        )
        nodes = []
        for node in tree.nodes:
            if node.condition is None:
                rows = list(node.rows)
                value = (
                    penalized_quantile(
                        residual[rows], q, p.weight[rows], penalty=penalty, anchor=anchor
                    )
                    if penalty
                    else weighted_quantile(residual[rows], q, p.weight[rows])
                )
                node = replace(node, value=np.array([value]))
            nodes.append(node)
        tree = replace(tree, nodes=tuple(nodes))
        np.testing.assert_allclose(step.raw_before, raw)
        raw += 0.1 * tree.predict(p.data.values)
        np.testing.assert_allclose(step.raw_after, raw)
        np.testing.assert_allclose(step.loss_after, objective.loss(p, raw))
    assert fit.state.version == 3


def test_routed_identity_original_weights_and_owned_rows():
    p = fixture()
    objective = Quantile()
    raw = np.zeros((6, 1))
    context = ResidualContext(p, objective.residuals(p, raw))
    seen = []

    def solver(view, total, names):
        seen.append(tuple(view.row_ids))
        positions = [list(p.row_ids).index(i) for i in view.row_ids]
        np.testing.assert_array_equal(view.weight, p.weight[positions])
        np.testing.assert_array_equal(view.residual, context.residual[positions])
        with pytest.raises(ValueError):
            view.weight.setflags(write=True)
        return quantile_leaf(view)

    b = Binning.fit(p.data, bins=4).transform(p.data)
    depthwise(b, objective.fields(p, raw), row_leaf=solver, leaf_context=context)
    assert len(seen) > 1 and any(len(rows) < 6 for rows in seen)
    foreign = ResidualContext(replace(p, offset=np.ones((6, 1))), context.residual)
    with pytest.raises(ValueError, match="match problem"):
        depthwise(b, objective.fields(p, raw), row_leaf=solver, leaf_context=foreign)
    with pytest.raises(ValueError):
        context.view([0, 0])
    with pytest.raises(ValueError):
        depthwise(b, objective.fields(p, raw), row_leaf=solver)


def test_penalty_uses_original_mass_and_backtracking_can_reject():
    p = fixture()
    context = ResidualContext(p, np.arange(6.0) + 1)
    view = context.view(np.arange(6))
    small = quantile_leaf(view, penalty=1, anchor=-10)
    large = quantile_leaf(view, penalty=100, anchor=-10)
    assert abs(large + 10) < abs(small + 10)
    scaled = replace(view, weight=view.weight * 10)
    assert quantile_leaf(scaled, penalty=1, anchor=-10) != small
    fit = quantile(
        p, p, context=RunContext("reject", 1), max_depth=0, rounds=2, step="backtracking"
    )
    assert fit.state.version == 0
    assert all(not s.accepted and s.raw_before is s.raw_after for s in fit.steps)


def test_persist_penalized_mixed_model(tmp_path):
    from openboost.artifacts import Model

    p = fixture()
    fit = quantile(p, p, context=RunContext("persist", 1), penalty=2, anchor=-1)
    path = tmp_path / "model.json"
    fit.state.model.save(path)
    loaded = Model.load(path)
    x = MixedData(
        [[0, "unknown"], [None, None], [10, "a"]],
        [30, 31, 32],
        p.data.feature_names,
        p.data.feature_kinds,
    )
    np.testing.assert_array_equal(loaded.predict(x), fit.state.model.predict(x))
