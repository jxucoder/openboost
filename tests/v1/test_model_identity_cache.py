"""Canonical byte oracles and ownership controls for immutable model identities."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from openboost import artifacts
from openboost.artifacts import ConstantTerm, Model, TreeTerm
from openboost.binning import Binning
from openboost.data import ClassSchema, NumericData
from openboost.tree import Tree


def canonical(model):
    raw = json.dumps(model.record(), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(len(raw).to_bytes(8, "big") + raw).hexdigest()


def specimen(width=2, categorical=False):
    names = ('x"\\雪', "β")
    binning = (
        Binning(names, ([], [-0.0, 2.0]), (("a", "雪"), None))
        if categorical
        else Binning(names, ([-0.0, 1.25], []))
    )
    tree = Tree(
        binning,
        [0, -1, -1],
        [0, -1, -1],
        [True, False, False],
        [1, -1, -1],
        [2, -1, -1],
        [[0.0, -0.0], [1e-100, -2.5], [3.0, 4.0]],
    )
    mapping = np.arange(2 * width, dtype=float).reshape(2, width) / 8
    mapping[0, 0] = -0.0
    return Model(
        names,
        [-0.0] * width,
        (
            ConstantTerm(np.arange(width) / 7, -0.125),
            TreeTerm(tree, mapping, 0.03),
            ConstantTerm([-1e-200] * width, 2.0),
        ),
    )


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_canonical_identity_including_unicode_negative_zero_and_prefix(width, categorical, size):
    model = specimen(width, categorical)
    classes = ClassSchema(("a", "雪")) if width in (1, 2) else ClassSchema((-3, 0, 7))
    model = replace(model, terms=model.terms[:size], classes=classes)
    assert model.identity == canonical(model)
    assert Model.from_record(model.record()).identity == model.identity


@pytest.mark.parametrize(
    "change",
    [
        "base",
        "coefficient",
        "mapping",
        "leaf",
        "value",
        "order",
        "append",
        "names",
        "classes",
        "signed_zero",
    ],
)
def test_distinct_semantic_and_byte_changes_do_not_reuse_identity(change):
    model = specimen()
    before = model.identity
    terms = model.terms
    if change == "base":
        other = replace(model, base=[1, 0])
    elif change == "coefficient":
        other = replace(model, terms=(replace(terms[0], coefficient=0.25), *terms[1:]))
    elif change == "mapping":
        other = replace(
            model, terms=(terms[0], replace(terms[1], mapping=np.ones((2, 2))), terms[2])
        )
    elif change == "leaf":
        tree = replace(terms[1].learner, value=[[0, 0], [5, 6], [3, 4]])
        other = replace(model, terms=(terms[0], replace(terms[1], learner=tree), terms[2]))
    elif change == "value":
        other = replace(model, terms=(replace(terms[0], value=[4, 5]), *terms[1:]))
    elif change == "order":
        other = replace(model, terms=tuple(reversed(terms)))
    elif change == "append":
        other = replace(model, terms=(*terms, ConstantTerm([0, 0])))
    elif change == "names":
        tree = replace(
            terms[1].learner, binning=replace(terms[1].learner.binning, feature_names=("a", "b"))
        )
        other = replace(
            model,
            feature_names=("a", "b"),
            terms=(terms[0], replace(terms[1], learner=tree), terms[2]),
        )
    elif change == "classes":
        other = replace(model, classes=ClassSchema((0, 1)))
    else:
        other = replace(model, base=[0.0, -0.0])
    assert other.identity == canonical(other)
    assert other.identity != before
    assert model.identity == canonical(model) == before


def test_cached_identity_encodes_nothing_on_second_access(monkeypatch):
    model = specimen()
    expected = canonical(model)
    calls = []
    original = json.dumps

    def counted(value, *args, **kwargs):
        calls.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(artifacts.json, "dumps", counted)
    assert model.identity == expected
    count = len(calls)
    assert count > 0
    assert model.identity == expected
    assert len(calls) == count


def test_appended_prefix_only_serializes_new_term_and_header(monkeypatch):
    model = specimen()
    before = model.identity
    extra = ConstantTerm([2, -3])
    extended = replace(model, terms=(*model.terms, extra))
    expected = canonical(extended)
    calls = []
    original = json.dumps

    def counted(value, *args, **kwargs):
        calls.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(artifacts.json, "dumps", counted)
    assert extended.identity == expected != before
    assert len(calls) == 2
    assert sum(v.get("kind") == "constant" for v in calls) == 1
    assert all(v.get("kind") != "tree" for v in calls)


def test_public_record_and_save_remain_fresh_and_exact(tmp_path):
    model = specimen()
    record = model.record()
    expected = json.dumps(record, allow_nan=False) + "\n"
    identity = model.identity
    record["base"][0] = 99
    record["terms"][1]["learner"]["value"][1][0] = 99
    record["terms"][1]["mapping"][0][0] = 99
    record["terms"].reverse()
    assert model.identity == canonical(model) == identity
    path = tmp_path / "model.json"
    model.save(path)
    assert path.read_text() == expected
    restored = Model.load(path)
    assert restored.identity == identity
    data = NumericData([[0, 1], [2, 3], [np.nan, 5]], [1, 2, 3], model.feature_names)
    assert restored.predict(data).tobytes() == model.predict(data).tobytes()


@pytest.mark.parametrize("kind", ["model", "constant", "tree"])
def test_cache_is_not_a_constructor_or_replacement_input(kind):
    model = specimen()
    obj = model if kind == "model" else model.terms[0 if kind == "constant" else 1]
    name = "_identity_cache" if kind == "model" else "_record_bytes_cache"
    with pytest.raises(ValueError, match="init=False"):
        replace(obj, **{name: "forged"})
    with pytest.raises(FrozenInstanceError):
        setattr(obj, name, "forged")
    kwargs = {n: getattr(obj, n) for n, f in obj.__dataclass_fields__.items() if f.init}
    with pytest.raises(TypeError):
        type(obj)(**kwargs, **{name: "forged"})
    assert name not in repr(obj)


def test_owned_arrays_and_replaced_terms_cannot_stale_a_cache():
    value, base = np.array([1.0]), np.array([2.0])
    term = ConstantTerm(value)
    model = Model(("x",), base, (term,))
    before = model.identity
    value[:] = 9
    base[:] = 9
    assert model.identity == canonical(model) == before
    for array in (term.value, model.base):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    other = replace(term, value=[3])
    assert other._record_bytes_cache is None
    replaced = replace(model, terms=(other,))
    assert replaced._identity_cache is None
    assert replaced.identity == canonical(replaced) != before


@pytest.mark.parametrize("kind", ["model", "constant", "term", "tree", "binning", "classes"])
def test_subclasses_keep_uncached_record_behavior(kind):
    model = specimen()
    if kind == "model":

        class Custom(Model):
            def record(self):
                result = super().record()
                result["custom"] = "extension"
                return result

        model = Custom(model.feature_names, model.base, model.terms)
    elif kind == "classes":

        class Custom(ClassSchema):
            pass

        model = replace(model, classes=Custom((0, 1)))
    elif kind == "constant":

        class Custom(ConstantTerm):
            pass

        model = replace(model, terms=(Custom([1, 2]),))
    else:
        term = model.terms[1]
        if kind == "term":

            class Custom(TreeTerm):
                pass

            term = Custom(term.learner, term.mapping)
        elif kind == "tree":

            class Custom(Tree):
                def record(self):
                    result = super().record()
                    result["custom"] = "extension"
                    return result

            tree = Custom(
                **{n: getattr(term.learner, n) for n in term.learner.__dataclass_fields__}
            )
            term = replace(term, learner=tree)
        else:

            class Custom(Binning):
                pass

            old = term.learner.binning
            tree = replace(
                term.learner, binning=Custom(old.feature_names, old.cuts, old.categories)
            )
            term = replace(term, learner=tree)
        model = replace(model, terms=(term,))
    assert model.identity == canonical(model)
    assert model._identity_cache is None
    assert all(term._record_bytes_cache is None for term in model.terms)


def test_finite_envelope_is_still_validated_after_prefix_hash():
    model = Model(("x",), [1e308], (ConstantTerm([1e307]),))
    assert model.identity == canonical(model)
    with pytest.raises(FloatingPointError):
        replace(model, terms=(*model.terms, ConstantTerm([1e308])))
