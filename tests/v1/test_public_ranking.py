"""Ranking geometry and round composition against independent query oracles."""

import numpy as np
import pytest

from openboost import NumericData, Problem, RunContext
from openboost.ranking import Ranking
from openboost.recipes import ranking
from tests.v1.reference.ranking import pairwise, query_ndcg


def problem():
    x = NumericData([[0], [2], [1], [3], [4], [5]], [6, 2, 3, 4, 5, 1], ("x",))
    return Problem(
        x,
        [[2], [0], [1], [0], [2], [1]],
        x.row_ids,
        structure={
            "query": [[0], [0], [0], [1], [1], [1]],
            "query_weight": [[2], [2], [2], [0.5], [0.5], [0.5]],
        },
    )


@pytest.mark.parametrize("lambdas", [False, True])
def test_geometry_matches_query_oracle(lambdas):
    p = problem()
    raw = np.array([[0], [1], [1], [-1], [2], [0.0]])
    objective = Ranking(lambdas=lambdas, k=2)
    result = objective.geometry(p, raw)
    ref = pairwise(
        raw[:, 0],
        p.target[:, 0],
        p.structure["query"][:, 0].astype(int),
        row_ids=p.row_ids,
        query_weight={0: 2, 1: 0.5},
        lambdas=lambdas,
        k=2,
    )
    np.testing.assert_allclose(result.gradient, ref.gradient)
    np.testing.assert_allclose(result.curvature, ref.curvature)
    np.testing.assert_allclose(result.loss, ref.loss)


@pytest.mark.parametrize("lambdas", [False, True])
def test_three_rounds_recompute_geometry_and_match_independent_tree(lambdas):
    from tests.v1.reference.mixed import Transformer, grow

    p = problem()
    fit = ranking(
        p, p, context=RunContext("rank", 7), lambdas=lambdas, rounds=3, bins=4, k=2, learning_rate=2
    )
    transform = Transformer.fit(
        p.data.values, names=p.data.feature_names, kinds=p.data.feature_kinds, bins=4
    )
    raw = np.zeros((6, 1))
    for step in fit.steps:
        expected = pairwise(
            raw[:, 0],
            p.target[:, 0],
            [0, 0, 0, 1, 1, 1],
            row_ids=p.row_ids,
            query_weight={0: 2, 1: 0.5},
            lambdas=lambdas,
            k=2,
        )
        np.testing.assert_allclose(step.gradient, expected.gradient)
        np.testing.assert_allclose(step.curvature, expected.curvature)
        tree = grow(
            p.data.values, expected.gradient[:, None], expected.curvature[:, None], transform
        )
        np.testing.assert_allclose(step.raw_before, raw)
        raw += 2 * tree.predict(p.data.values)
        np.testing.assert_allclose(step.raw_after, raw)
    assert fit.state.version == 3
    assert not np.allclose(fit.steps[0].gradient, fit.steps[1].gradient)
    objective = Ranking(lambdas=lambdas, k=2)
    expected_ndcg = (
        sum(
            w * query_ndcg(raw[rows, 0], p.target[rows, 0], row_ids=p.row_ids[rows], k=2)
            for rows, w in [(slice(0, 3), 2), (slice(3, 6), 0.5)]
        )
        / 2.5
    )
    np.testing.assert_allclose(objective.score(p, raw), 1 - expected_ndcg)


def test_query_isolation_offsets_and_tie_permutation():
    from dataclasses import replace

    p = problem()
    raw = np.zeros((6, 1))
    objective = Ranking(lambdas=True, k=2)
    base = objective.geometry(p, raw)
    changed = raw.copy()
    changed[3:] = [[5], [-3], [1]]
    np.testing.assert_array_equal(objective.geometry(p, changed).gradient[:3], base.gradient[:3])
    shifted = replace(p, offset=changed)
    np.testing.assert_array_equal(
        objective.geometry(shifted, raw).gradient, objective.geometry(p, changed).gradient
    )
    order = np.array([5, 2, 0, 4, 1, 3])
    x = NumericData(p.data.values[order], p.row_ids[order], p.data.feature_names)
    shuffled = Problem(
        x,
        p.target[order],
        x.row_ids,
        structure={key: value[order] for key, value in p.structure.items()},
    )
    np.testing.assert_allclose(objective.geometry(shuffled, raw).gradient, base.gradient[order])
    np.testing.assert_allclose(objective.score(shuffled, raw), objective.score(p, raw))


def test_logistic_derivatives_and_degenerate_queries():
    from dataclasses import replace

    p = problem()
    objective = Ranking()
    raw = np.arange(6.0)[:, None] / 3
    result = objective.geometry(p, raw)
    epsilon = 1e-4
    for i in range(6):
        delta = np.zeros_like(raw)
        delta[i] = epsilon
        plus = objective.geometry(p, raw + delta).loss
        minus = objective.geometry(p, raw - delta).loss
        np.testing.assert_allclose((plus - minus) / (2 * epsilon), result.gradient[i], atol=1e-8)
        np.testing.assert_allclose(
            (plus + minus - 2 * result.loss) / epsilon**2, result.curvature[i], atol=1e-7
        )
    flat = replace(p, target=np.zeros((6, 1)))
    assert objective.geometry(flat, raw).loss == 0
    assert objective.score(flat, raw) == 0


@pytest.mark.parametrize("change", ["weight", "query", "query_weight", "structure", "target"])
def test_invalid_roles_rejected(change):
    from dataclasses import replace

    p = problem()
    if change == "weight":
        p = replace(p, weight=np.full(6, 2))
    elif change == "target":
        p = replace(p, target=np.full((6, 1), 0.5))
    else:
        roles = dict(p.structure)
        if change == "structure":
            roles["ignored"] = np.ones((6, 1))
        else:
            roles[change] = np.arange(6.0)[:, None] / 3
        p = replace(p, structure=roles)
    with pytest.raises(ValueError):
        ranking(p, p, context=RunContext("invalid", 1))


def test_score_model_roundtrip_needs_no_query_roles(tmp_path):
    import json
    import subprocess
    import sys

    from openboost.artifacts import Model

    p = problem()
    model = ranking(p, p, context=RunContext("persist-ranking", 1), rounds=2).state.model
    path = tmp_path / "ranking.json"
    model.save(path)
    assert Model.load(path).identity == model.identity
    code = """import json, sys
from openboost import NumericData
from openboost.artifacts import Model
x = NumericData([[0], [float('nan')], [5]], [10, 11, 12], ('x',))
print(json.dumps(Model.load(sys.argv[1]).predict(x).tolist()))
"""
    x = NumericData([[0], [np.nan], [5]], [10, 11, 12], ("x",))
    output = json.loads(subprocess.check_output([sys.executable, "-c", code, str(path)], text=True))
    np.testing.assert_array_equal(output, model.predict(x))
