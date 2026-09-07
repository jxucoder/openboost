"""CPU agreement for the 088 oracle, frozen before resident training kernels."""

from functools import partial

import numpy as np
import pytest

from openboost.artifacts import TreeTerm
from openboost.binning import Binning
from openboost.data import NumericData, Problem
from openboost.objectives import Squared
from openboost.ops import feasible
from openboost.runtime import RunContext, initialize, propose_terms, resolve
from openboost.tree import depthwise

from .reference.device_rounds import fixture, rounds


def prepared_fixture(case):
    f = fixture(case)
    names = tuple(f"f{i}" for i in range(f["x"].shape[1]))
    train_data = NumericData(f["x"], 101 + 7 * np.arange(len(f["x"])), names)
    validation_data = NumericData(
        f["validation_x"], 301 + 11 * np.arange(len(f["validation_x"])), names
    )
    train = Problem(
        train_data,
        f["target"][:, None],
        train_data.row_ids,
        weight=f["weight"],
        offset=f["offset"][:, None],
    )
    validation = Problem(
        validation_data,
        f["validation_target"][:, None],
        validation_data.row_ids,
        weight=f["validation_weight"],
        offset=f["validation_offset"][:, None],
    )
    cuts = tuple(
        np.arange(int(np.nanmax(col))) + 0.5 if np.any(~np.isnan(col)) else np.array([])
        for col in f["x"].T
    )
    binned = Binning(names, cuts).transform(train_data)
    return train, validation, binned, f["information"]


@pytest.mark.parametrize("case", ["weighted", "d2", "conflict"])
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("minimum", [None, 0, 1, 2])
def test_two_round_public_cpu_agreement(case, depth, minimum):
    train, validation, binned, information = prepared_fixture(case)
    base, expected = rounds(case, depth=depth, minimum=minimum)

    def learner(data, fields):
        fields = fields.add_independent("a", information[:, 0]).add_independent(
            "b", information[:, 1]
        )
        return depthwise(
            data,
            fields,
            max_depth=depth,
            legality=partial(feasible, min_information={"a": minimum, "b": minimum})
            if minimum is not None
            else feasible,
        )

    state = initialize(
        RunContext("088", 7), train, validation, Squared.base(train), score=Squared.loss
    )
    np.testing.assert_allclose(state.model.base, [base], rtol=0, atol=1e-12)
    for ref in expected:
        np.testing.assert_allclose(
            Squared.gradient(train, state.train_raw), ref["gradient"], rtol=0, atol=1e-12
        )
        tree = learner(binned, Squared.fields(train, state.train_raw))
        proposal = propose_terms(state, (TreeTerm(tree, [[1]], 0.5),))
        state = resolve(state, proposal, accept=True, score=Squared.loss)
        np.testing.assert_allclose(state.train_raw[:, 0], ref["raw"], rtol=0, atol=1e-12)
        actual = [
            None if f == -1 else (int(f), int(t), bool(m))
            for f, t, m in zip(tree.feature, tree.threshold, tree.missing_left, strict=True)
        ]
        assert actual == [node["key"] for node in ref["nodes"]]
        np.testing.assert_allclose(
            tree.value[:, 0], [node["value"] for node in ref["nodes"]], atol=1e-12
        )
    np.testing.assert_allclose(
        state.validation_raw[:, 0], expected[-1]["validation_raw"], atol=1e-12
    )
    assert Squared.loss(train, state.train_raw) == pytest.approx(expected[-1]["loss"])
    assert len(state.best_model.terms) == expected[-1]["best_round"]


def test_frozen_d2_change_and_validation_conflict():
    assert rounds("d2", depth=1)[1][0]["nodes"][0]["key"] == (0, 0, False)
    assert rounds("d2", depth=1, minimum=1)[1][0]["nodes"][0]["key"] == (0, 1, False)
    assert rounds("conflict")[1][-1]["best_round"] == 0
