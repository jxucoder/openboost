"""Explicit histogram selection binding, growth and complete recipe equivalence."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from openboost import NumericData, Problem, RunContext, newton_order, ops
from openboost.binning import Binning
from openboost.recipes import squared
from openboost.stats import newton
from openboost.tree import Tree, depthwise
from tests.v1.test_newton_choice import fixture
from tests.v1.test_public_categorical import fixture as mixed_fixture


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("depth,leaves", [(0, 1), (2, 3), (6, None)])
@pytest.mark.parametrize("regularizer", [0.0, 10.0])
def test_explicit_selection_matches_complete_original_tree(seed, depth, leaves, regularizer):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 10, (24, 3)).astype(float)
    x[rng.random(x.shape) < 0.2] = np.nan
    b, fields = fixture(x, rng.normal(size=24), rng.uniform(0.1, 2, 24), rng.integers(0, 4, 24))
    common = dict(
        max_depth=depth, max_leaves=leaves, leaf=partial(ops.newton_leaf, reg_lambda=regularizer)
    )
    old = depthwise(
        b,
        fields,
        **common,
        scoring=partial(ops.score, reg_lambda=regularizer, split_penalty=0.01),
        legality=partial(ops.feasible, min_child_h=0.1),
    )
    new = depthwise(
        b,
        fields,
        **common,
        selection=partial(
            ops.newton_choice, reg_lambda=regularizer, min_child_h=0.1, split_penalty=0.01
        ),
    )
    assert new.record() == old.record()
    assert new.identity == old.identity
    assert new.predict(b.data).tobytes() == old.predict(b.data).tobytes()
    assert Tree.from_record(new.record()).predict(b.data).tobytes() == old.predict(b.data).tobytes()


@pytest.mark.parametrize(
    "field",
    [
        "data_identity",
        "rows_identity",
        "names",
        "roles",
        "left",
        "right",
        "parent",
        "left_count",
        "right_count",
        "kind",
        "threshold",
        "feature",
    ],
)
def test_selection_cannot_forge_actual_node_histogram(field):
    b, f = fixture([[0], [1], [2], [3]], [-3, -1, 1, 3])

    def forged(hist):
        c, gain = ops.newton_choice(hist)
        value = getattr(c, field)
        if isinstance(value, np.ndarray):
            value = value.copy()
            value[0] = np.nextafter(value[0], np.inf)
        elif isinstance(value, tuple):
            value = (value[0] + "-foreign", *value[1:])
        elif isinstance(value, str):
            value += "-foreign"
        else:
            value += 99
        return replace(c, **{field: value}), gain

    with pytest.raises(ValueError, match="selection"):
        depthwise(b, f, selection=forged)


@pytest.mark.parametrize("gain", [0, -1, np.nan, np.inf])
def test_selection_rejects_invalid_gain(gain):
    b, f = fixture([[0], [1]], [-1, 1])
    with pytest.raises(ValueError, match="positive gain"):
        depthwise(b, f, selection=lambda h: (ops.newton_choice(h)[0], gain))


@pytest.mark.parametrize(
    "conflict",
    [dict(scoring=lambda _: 1), dict(legality=lambda _: True), dict(ordering=newton_order.rank)],
)
def test_selection_is_explicit_and_excludes_other_ranking_callbacks(conflict):
    b, f = fixture([[0], [1]], [-1, 1])
    with pytest.raises(ValueError, match="selection"):
        depthwise(b, f, selection=ops.newton_choice, **conflict)


def test_root_only_does_not_call_selector_and_none_keeps_leaf():
    b, f = fixture([[0], [1]], [-1, 1])
    calls = []

    def selection(hist):
        calls.append(hist)
        return None

    assert len(depthwise(b, f, selection=selection, max_depth=0).value) == 1
    assert not calls
    assert len(depthwise(b, f, selection=selection).value) == 1
    assert len(calls) == 1


def test_categorical_and_independent_constraints_preserve_original_tree():
    p, gradient = mixed_fixture()
    b = Binning.fit(p.data, bins=4).transform(p.data)
    fields = newton(p, gradient, np.ones(len(gradient))).add_independent(
        "cohort", np.ones(len(gradient))
    )
    old = depthwise(
        b, fields, max_depth=3, legality=partial(ops.feasible, min_information={"cohort": 2})
    )
    new = depthwise(
        b, fields, max_depth=3, selection=partial(ops.newton_choice, min_information={"cohort": 2})
    )
    assert new.identity == old.identity
    assert new.predict(p.data).tobytes() == old.predict(p.data).tobytes()


def test_selector_buffers_are_borrowed_and_tree_owns_its_fields():
    b, f = fixture([[0], [1], [2], [3]], [-3, -1, 1, 3])
    buffers = []

    def selection(hist):
        result = ops.newton_choice(hist)
        if result is None:
            return None
        c, gain = result
        left, right, parent = c.left.copy(), c.right.copy(), c.parent.copy()
        buffers.extend((left, right, parent))
        return replace(c, left=left, right=right, parent=parent), gain

    tree = depthwise(b, f, selection=selection)
    expected = tree.record()
    for buffer in buffers:
        buffer[:] = 999
    assert tree.record() == expected


@pytest.mark.parametrize("rate,patience", [(0.1, 3), (0.0, 2)])
def test_complete_recipe_final_best_stop_and_history_match(rate, patience):
    rng = np.random.default_rng(901)
    x = NumericData(rng.normal(size=(32, 3)), np.arange(32), ("a", "b", "c"))
    y = (x.values[:, 0] * 2 - x.values[:, 1] ** 2)[:, None]
    p = Problem(x, y, x.row_ids)
    validation_x = NumericData(rng.normal(size=(12, 3)), np.arange(100, 112), x.feature_names)
    validation = Problem(
        validation_x,
        (validation_x.values[:, 0] * 2 - validation_x.values[:, 1] ** 2)[:, None],
        validation_x.row_ids,
    )
    old = squared(
        p,
        validation,
        context=RunContext("selection", 2),
        rounds=8,
        learning_rate=rate,
        patience=patience,
        learner=partial(
            depthwise,
            max_depth=3,
            scoring=partial(ops.score, reg_lambda=2.0),
            legality=ops.feasible,
            leaf=partial(ops.newton_leaf, reg_lambda=2.0),
        ),
    )
    new = squared(
        p,
        validation,
        context=RunContext("selection", 2),
        rounds=8,
        learning_rate=rate,
        patience=patience,
        max_depth=3,
        reg_lambda=2.0,
    )
    assert old.stop == new.stop
    assert old.state.model.record() == new.state.model.record()
    assert old.state.best_model.record() == new.state.best_model.record()
    assert old.state.train_raw.tobytes() == new.state.train_raw.tobytes()
    assert old.state.validation_raw.tobytes() == new.state.validation_raw.tobytes()
    assert len(old.steps) == len(new.steps)
    for a, b in zip(old.steps, new.steps, strict=True):
        assert a.loss_before == b.loss_before and a.loss_after == b.loss_after
        assert a.accepted == b.accepted and a.coefficients == b.coefficients
