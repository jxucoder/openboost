"""F0.2 composition: offsets, two-stage outputs and accepted/best-state consistency."""

import numpy as np
import pytest

from .reference.coupled import normal, step
from .reference.integration import (
    Proposal,
    Track,
    TwoStage,
    advance,
    fit_positive,
    fit_quantiles,
    restore_best,
)
from .reference.positive import poisson_predict
from .reference.runs import derive_seed


def test_poisson_offset_once_two_rounds_and_new_exposure():
    x = [[0], [1], [2]]
    y, e = [0, 2, 5], [0.5, 1, 2]
    model, trace = fit_positive(x, y, kind="poisson", exposure=e, weight=[1, 2, 1])
    assert len(trace) == len(model.terms) == 2
    raw = np.full(3, model.base)
    for before, g, h, after in trace:
        np.testing.assert_allclose(before, raw)
        expected = np.array(e) * np.exp(raw)
        np.testing.assert_allclose(g, expected - y)
        np.testing.assert_allclose(h, expected)
        raw = np.array(after)
    np.testing.assert_allclose(model.raw(x), raw)
    normal_prediction = poisson_predict(model.raw(x), e)
    doubled = poisson_predict(model.raw(x), np.array(e) * 2)
    np.testing.assert_allclose(normal_prediction["rate"], doubled["rate"])
    np.testing.assert_allclose(doubled["count_mean"], 2 * normal_prediction["count_mean"])
    # Without this distinction an offset would compound each round.
    assert not np.allclose(model.raw(x), raw + 2 * np.log(e))


def test_two_stage_actual_training_and_independent_reconstruction():
    x = [[0], [1], [2]]
    model = TwoStage.fit(x, [0, 1, 2], [1, 0.5, 2], [[1], [2], [2]], [10, 20, 30])
    result = model.predict([[0], [1], [3]], [1, 2, 4])
    frequency = np.exp(model.frequency.raw([[0], [1], [3]]))
    severity = np.exp(model.severity.raw([[0], [1], [3]]))
    np.testing.assert_allclose(result["annualized"], frequency * severity)
    np.testing.assert_allclose(result["amount"], np.array([1, 2, 4]) * frequency * severity)
    assert len(model.frequency.terms) == len(model.severity.terms) == 2
    # Prediction has no dependence on future target/claim tables.
    assert all(np.isfinite(result["amount"]))


def test_three_quantile_ensembles_two_rounds_and_payloads():
    models = fit_quantiles([[0], [1], [2]], [0, 2, 10], weight=[2, 1, 2])
    assert tuple(models) == (0.1, 0.5, 0.9)
    for model in models.values():
        assert len(model.terms) == 2
        expected = np.full(2, model.base)
        for tree, coefficient in model.terms:
            expected += coefficient * tree.predict([[0], [3]])
        np.testing.assert_allclose(model.raw([[0], [3]]), expected)


def start():
    train, validation = [[0], [0]], [[0]]
    raw = np.zeros((2, 2))
    track = Track.initialize("run", 7, raw, np.zeros((1, 2)), score=lambda r: float(np.sum(r * r)))
    return train, validation, track


def proposal(track, train, round_index, channel, objective=normal):
    result = step(
        train,
        track.current.train_raw,
        [1, 3],
        objective,
        channels=(channel,),
        rates=tuple(0.1 * 0.5**j for j in range(6)),
    )
    return Proposal(track.run_id, track.current.version, (round_index, channel), result)


def test_ordered_state_two_rounds_restore_best_caches_coefficients_and_step():
    train, validation, track = start()
    for round_index in (1, 2):
        for channel in (0, 1):
            p = proposal(track, train, round_index, channel)
            track = advance(track, p, train, validation, score=lambda r: float(np.sum(r * r)))
    assert track.current.version == 4 and len(track.current.terms) == 4
    assert track.best.version == 0  # validation optimum was the initial raw
    future = proposal(track, train, 3, 0)
    restored = restore_best(track)
    assert restored.current.version == 5 and restored.current.step_id == (0, -1)
    assert restored.current.terms == ()
    np.testing.assert_array_equal(restored.current.train_raw, np.zeros((2, 2)))
    np.testing.assert_array_equal(restored.current.valid_raw, np.zeros((1, 2)))
    with pytest.raises(ValueError, match="stale"):
        advance(restored, future, train, validation, score=lambda r: 0.0)


def test_state_rejection_and_invalid_validation_are_atomic():
    train, validation, track = start()

    def reverse(r, y, weight=None):
        loss, g, metric = normal(r, y, weight=weight)
        return loss, -g, metric

    p = proposal(track, train, 1, 0, reverse)
    assert not p.update.accepted
    assert advance(track, p, train, validation, score=lambda r: 0.0) is track
    accepted = proposal(track, train, 1, 0)
    with pytest.raises(ValueError):
        advance(track, accepted, train, validation, score=lambda r: float("nan"))
    assert track.current.version == 0 and track.current.terms == ()
    retry = advance(track, accepted, train, validation, score=lambda r: float(np.sum(r * r)))
    assert retry.current.version == 1
    assert derive_seed(track.seed, track.run_id, 1, "parameter-0", "rows") == derive_seed(
        retry.seed, retry.run_id, 1, "parameter-0", "rows"
    )


def test_best_nonzero_snapshot_rebuilds_train_and_validation():
    train, validation, _ = start()

    def score(raw):
        return float((raw[0, 0] - 2 / 15) ** 2 + raw[0, 1] ** 2)

    track = Track.initialize("run", 7, np.zeros((2, 2)), np.zeros((1, 2)), score=score)
    first = proposal(track, train, 1, 0)
    track = advance(track, first, train, validation, score=score)
    saved = track.current
    second = proposal(track, train, 1, 1)
    next_key = derive_seed(track.seed, track.run_id, 1, "parameter-1", "rows")
    track = advance(track, second, train, validation, score=score)
    restored = restore_best(track)
    assert restored.current.terms == saved.terms
    assert restored.current.step_id == saved.step_id
    assert restored.current.train_raw == saved.train_raw
    assert restored.current.valid_raw == saved.valid_raw
    assert derive_seed(restored.seed, restored.run_id, 1, "parameter-1", "rows") == next_key
    train_raw, valid_raw = np.zeros((2, 2)), np.zeros((1, 2))
    for channel, tree, coefficient in restored.current.terms:
        train_raw[:, channel] += coefficient * tree.predict(train)
        valid_raw[:, channel] += coefficient * tree.predict(validation)
    np.testing.assert_allclose(train_raw, restored.current.train_raw)
    np.testing.assert_allclose(valid_raw, restored.current.valid_raw)


def test_cross_run_and_corrupted_proposals_cannot_commit():
    from dataclasses import replace

    train, validation, track = start()
    p = proposal(track, train, 1, 0)
    with pytest.raises(ValueError, match="another run"):
        advance(track, replace(p, run_id="foreign"), train, validation, score=lambda r: 0.0)
    wrong = replace(p.update, raw_after=((9.0, 9.0), (9.0, 9.0)))
    with pytest.raises(ValueError, match="terms"):
        advance(track, replace(p, update=wrong), train, validation, score=lambda r: 0.0)
    assert track.current.version == 0 and track.best.version == 0


def test_two_stage_join_to_model_uses_paid_records_not_raw_claimnb():
    from .reference.positive import policy_losses

    policies = [("a", 0, 1.0), ("b", 2, 0.5), ("c", 3, 2.0)]
    payments = [("b", 10.0), ("c", 20.0), ("c", 30.0)]
    rows, excluded = policy_losses(policies, payments)
    assert excluded == ()
    features = {"a": [0], "b": [1], "c": [2]}
    model = TwoStage.fit(
        [features[r[0]] for r in rows],
        [r[4] for r in rows],
        [r[1] for r in rows],
        [features[i] for i, _ in payments],
        [v for _, v in payments],
    )
    # Frequency base uses 3 positive payment records, not raw ClaimNb total 5.
    assert model.frequency.base == pytest.approx(np.log(3 / 3.5))
    assert model.severity.base == pytest.approx(np.log(20))
    assert np.all(model.predict([[1], [2]], [1, 1])["annualized"] > 0)


def test_commit_rejects_broadcastable_row_mismatches():
    from dataclasses import replace

    train, validation, track = start()
    p = proposal(track, train, 1, 0)
    with pytest.raises(ValueError, match="rows"):
        advance(track, p, [[0]], validation, score=lambda r: 0.0)
    wrong = replace(p.update, raw_after=(p.update.raw_after[0],))
    with pytest.raises(ValueError, match="shape"):
        advance(track, replace(p, update=wrong), train, validation, score=lambda r: 0.0)
