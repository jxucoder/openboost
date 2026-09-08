"""Independent new-semantic trajectories, without modifying the frozen old oracle."""

from itertools import product

import numpy as np
import pytest

from .reference import compared_normal as revised
from .reference import device_normal as historical
from .reference.normal_comparison import compare

SETTINGS = list(
    product(
        [
            ("weighted", 1, None),
            ("d2", 1, None),
            ("d2", 2, 1),
            ("conflict", 0, None),
            ("conflict", 2, None),
        ],
        [("ordinary", 0), ("natural", 0), ("natural", 0.25)],
        ["joint", "forward", "reverse"],
        [(True, 0.1), (False, 8.0)],
    )
)


def options(setting):
    (case, depth, minimum), (mode, damping), update, (fixed, rate) = setting
    return case, dict(
        depth=depth,
        minimum=minimum,
        mode=mode,
        damping=damping,
        update=update,
        fixed=fixed,
        rate=rate,
    )


def test_tiny_improvement_is_accepted_without_changing_reported_loss():
    before = np.array([[2.0**-30, 0]])
    target, offset, weight = np.array([0]), np.zeros((1, 2)), np.ones(1)
    after, attempts = revised.trial(before, -before, target, offset, weight, rate=1)
    np.testing.assert_array_equal(after, [[0, 0]])
    assert attempts[0][2] == "accepted"
    assert historical.geometry(before, target, offset, weight)[0] == attempts[0][1]


def test_reference_best_selection_advances_on_equal_reported_scores(monkeypatch):
    tiny = 2.0**-30
    data = dict(
        x=np.zeros((1, 1)),
        target=np.zeros(1),
        offset=np.zeros((1, 2)),
        weight=np.ones(1),
        information=np.ones((1, 2)),
        validation_x=np.zeros((1, 1)),
        validation_target=np.zeros(1),
        validation_offset=np.zeros((1, 2)),
        validation_weight=np.ones(1),
    )
    monkeypatch.setattr(revised, "fixture", lambda case: data)
    monkeypatch.setattr(revised, "base", lambda *args: np.array([tiny, 0]))
    _, steps = revised.rounds("controlled", depth=0, update="forward", rate=1, count=1)
    first = steps[0]
    assert first["accepted"] and first["best_terms"] == 1
    assert (
        first["best_score"]
        == historical.geometry(np.array([[tiny, 0]]), data["target"], data["offset"], data["weight"])[0]
    )
    np.testing.assert_array_equal(first["best_raw"], [[tiny / 2, 0]])


@pytest.mark.parametrize("setting", SETTINGS)
def test_all_original_settings_have_complete_independent_comparison_trajectories(setting):
    case, config = options(setting)
    f = historical.fixture(case)
    initial, steps = revised.rounds(case, **config)
    groups = 1 if config["update"] == "joint" else 2
    assert len(steps) == 3 * groups
    best = np.broadcast_to(initial, f["validation_offset"].shape).copy()
    best_terms = version = nterms = 0
    for index, s in enumerate(steps):
        assert s["round"] == index // groups
        change = compare(s["before"], s["raw"], f["target"], f["offset"], f["weight"])
        if s["accepted"]:
            assert config["fixed"] or change.improves()
            version += 1
            nterms += len(s["channels"])
            best_change = compare(
                best,
                s["validation_raw"],
                f["validation_target"],
                f["validation_offset"],
                f["validation_weight"],
            )
            if best_change.improves():
                best, best_terms = s["validation_raw"].copy(), nterms
        else:
            np.testing.assert_array_equal(s["raw"], s["before"])
            assert len(s["attempts"]) == 6
        assert (s["version"], s["nterms"], s["best_terms"]) == (version, nterms, best_terms)
        np.testing.assert_array_equal(s["best_raw"], best)
        assert (
            s["best_score"]
            == historical.geometry(
                best, f["validation_target"], f["validation_offset"], f["validation_weight"]
            )[0]
        )


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_original_failed_settings_have_explicit_new_reference_expectations(order):
    _, steps = revised.rounds("conflict", depth=0, mode="ordinary", update=order, rate=8)
    assert len(steps) == 6
    assert all(not s["accepted"] for s in steps)
    assert all(tuple(a[0] for a in s["attempts"]) == (8, 4, 2, 1, 0.5, 0.25) for s in steps)
