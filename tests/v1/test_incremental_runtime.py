"""New-term work and exact independent replay of public transactions."""

from unittest.mock import patch

import numpy as np
import pytest

from openboost import NumericData, Problem, RunContext
from openboost.binning import Binning
from openboost.recipes import normal, squared
from openboost.tree import Tree


@pytest.mark.parametrize("recipe,width", [(squared, 1), (normal, 2)])
@pytest.mark.parametrize("rounds", [4, 8])
def test_new_term_work_is_linear_and_replay_exact(recipe, width, rounds):
    x = np.arange(96, dtype=float).reshape(32, 3) / 32
    data = NumericData(x, np.arange(32), ("a", "b", "c"))
    other = NumericData(x[:16] + 0.01, np.arange(32, 48), data.feature_names)
    train = Problem(data, (1 + np.sin(x[:, 0]))[:, None], data.row_ids, raw_width=width)
    valid = Problem(other, (1 + np.cos(x[:16, 0]))[:, None], other.row_ids, raw_width=width)
    calls, encodings = [], []
    predict, transform = Tree.predict, Binning.transform

    def counted(tree, inputs, **kwargs):
        calls.append(len(inputs.values))
        return predict(tree, inputs, **kwargs)

    def encoded(binning, inputs):
        encodings.append((binning.identity, inputs.identity))
        return transform(binning, inputs)

    with patch.object(Tree, "predict", counted), patch.object(Binning, "transform", encoded):
        result = recipe(
            train,
            valid,
            context=RunContext("linear", 63),
            rounds=rounds,
            patience=None,
            step="fixed",
            bins=8,
            max_depth=1,
        )
    np.testing.assert_array_equal(result.state.train_raw, result.state.model.predict(data))
    np.testing.assert_array_equal(result.state.validation_raw, result.state.model.predict(other))
    assert len(calls) == 2 * width * rounds
    assert len(encodings) <= 3  # learner preparation plus runtime train/validation


def test_atomic_candidate_rounding_and_cache_ownership():
    from dataclasses import replace

    from openboost.artifacts import ConstantTerm
    from openboost.runtime import Proposal, initialize, preview, preview_raw, propose_terms, resolve

    data = NumericData([[0], [1]], [10, 11], ("x",))
    problem = Problem(data, [[0], [0]], data.row_ids, offset=[[1], [2]])

    def score(p, raw):
        return float(np.mean(p.with_offset(raw) ** 2))

    state = initialize(RunContext("rounding", 0), problem, problem, [1e16], score=score)
    # Combining these updates into a delta would lose the unit term.
    terms = (ConstantTerm([-1e16]), ConstantTerm([1]))
    proposal = Proposal(state.identity, terms)  # Direct construction must also evaluate safely.
    train_raw, valid_raw = preview_raw(state, proposal)
    np.testing.assert_array_equal(train_raw, [[1], [1]])
    np.testing.assert_array_equal(train_raw, preview(state, proposal).predict(data))
    for a in (train_raw, valid_raw):
        with pytest.raises(ValueError):
            a.flags.writeable = True
    assert resolve(state, proposal, accept=False, score=score) is state
    with pytest.raises(ValueError, match="finite"):
        resolve(state, proposal, accept=True, score=lambda *_: np.nan)
    updated = resolve(state, proposal, accept=True, score=score)
    np.testing.assert_array_equal(updated.train_raw, train_raw)
    assert state.version == 0 and updated.version == 1
    # Replacement goes through public construction and recomputes raw values.
    rebuilt = replace(updated, model=state.model)
    np.testing.assert_array_equal(rebuilt.train_raw, state.train_raw)
    changed = replace(proposal, terms=(ConstantTerm([0]),))
    np.testing.assert_array_equal(preview_raw(state, changed)[0], state.train_raw)
    with pytest.raises(TypeError):
        Proposal(state.identity, terms, _evaluation=proposal._evaluation)
    with pytest.raises(ValueError):
        replace(updated, train_raw=np.zeros((2, 1)))
    foreign = replace(state, context=RunContext("other", 0))
    with pytest.raises(ValueError, match="foreign"):
        preview_raw(foreign, proposal)
    fresh = propose_terms(updated, (ConstantTerm([2]),))
    assert preview_raw(updated, fresh)[0][0, 0] == 3


def test_encoding_binding_with_missing_categories_and_vector_leaves():
    from openboost.artifacts import TreeTerm
    from openboost.data import MixedData
    from openboost.runtime import initialize, preview_raw, propose_terms, resolve

    data = MixedData(
        [[0, "a"], [np.nan, "b"], [2, None]],
        [1, 2, 3],
        ("x", "category"),
        ("numeric", "categorical"),
    )
    bins = Binning.fit(data, bins=3)
    tree = Tree(
        bins,
        np.array([0, -1, -1]),
        np.array([0, -1, -1]),
        np.array([True, False, False]),
        np.array([1, -1, -1]),
        np.array([2, -1, -1]),
        np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]),
    )
    encoded = bins.transform(data)
    np.testing.assert_array_equal(tree.predict(data), tree.predict(data, binned=encoded))
    changed = MixedData(data.values, [3, 2, 1], data.feature_names, data.feature_kinds)
    with pytest.raises(ValueError, match="identity"):
        tree.predict(changed, binned=encoded)
    other_bins = Binning.fit(data, bins=1)
    with pytest.raises(ValueError, match="identity"):
        tree.predict(data, binned=other_bins.transform(data))
    p = Problem(data, [[0, 0], [0, 0], [0, 0]], data.row_ids)
    state = initialize(RunContext("mixed", 0), p, p, [0, 0], score=lambda *_: 0.0)
    term = TreeTerm(tree, [[1, 0.5], [0, 1]])
    trial = propose_terms(state, (term,))
    assert len(state._encodings) == 0
    updated = resolve(state, trial, accept=True, score=lambda *_: 0.0)
    assert len(updated._encodings) == 1 and len(state._encodings) == 0
    with pytest.raises(TypeError):
        updated._encodings["invalid"] = encoded
    next_trial = propose_terms(updated, (term,))
    np.testing.assert_array_equal(
        preview_raw(updated, next_trial)[0], 2 * (tree.predict(data) @ term.mapping)
    )
