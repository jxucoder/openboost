"""Installed-wheel composition: public imports only, no repository helpers."""

import json
from pathlib import Path

import numpy as np
import pytest
from bounded_leaves import BoundedNewton
from normal_fisher import ChannelDecay, NormalFisher

from openboost.experimental import Booster, LevelWiseBuilder, TrainerConfig


def fit(rule=None, schedule=None, **kwargs):
    X = np.arange(4, dtype=np.float32)[:, None]
    y = np.array([-3, -1, 1, 3], np.float32)
    model = Booster(
        objective=NormalFisher(),
        tree_builder=LevelWiseBuilder(leaf_rule=rule),
        step_schedule=schedule,
        config=TrainerConfig(
            n_trees=2,
            max_depth=1,
            learning_rate=0.5,
            reg_lambda=1,
            min_child_weight=0,
            random_state=7,
        ),
    ).fit(X, y, **kwargs)
    return X, y, model


def test_two_round_reference_and_composition():
    X, y, combined = fit(BoundedNewton(0.1), ChannelDecay())
    _, _, default = fit()
    _, _, scheduled = fit(schedule=ChannelDecay())
    _, _, bounded = fit(rule=BoundedNewton(0.1))
    assert combined.coefficients_ == {"mu": [0.5, 0.25], "log_sigma": [0.25, 0.125]}
    assert all(v == [0.5, 0.5] for v in default.coefficients_.values())
    # Independent exhaustive depth-one reference on the original feature rows.
    raw = {"mu": np.zeros(4, np.float32), "log_sigma": np.full(4, 0.5 * np.log(5), np.float32)}
    second_grad = {}
    for r in range(2):
        residual = raw["mu"].astype(float) - y
        precision = np.exp(-2 * raw["log_sigma"].astype(float))
        stats = {
            "mu": (residual * precision, precision),
            "log_sigma": (1 - residual**2 * precision, np.full(4, 2.0)),
        }
        if r == 1:
            second_grad = {k: v[0].copy() for k, v in stats.items()}
        for channel, (g, h) in stats.items():
            # Production statistics are float32; emulate that public boundary only.
            g, h = g.astype(np.float32).astype(float), h.astype(np.float32).astype(float)
            scores = []
            for split in range(1, 4):
                score = (
                    g[:split].sum() ** 2 / (h[:split].sum() + 1)
                    + g[split:].sum() ** 2 / (h[split:].sum() + 1)
                    - g.sum() ** 2 / (h.sum() + 1)
                )
                scores.append(score)
            split = int(np.argmax(scores)) + 1
            groups = [np.arange(split), np.arange(split, 4)] if max(scores) > 0 else [np.arange(4)]
            pred = np.zeros(4, np.float32)
            for rows in groups:
                pred[rows] = np.clip(-g[rows].sum() / (h[rows].sum() + 1), -0.1, 0.1)
            np.testing.assert_allclose(
                combined.trees_[channel][r](combined.X_binned_), pred, atol=1e-7
            )
            raw[channel] += ([0.5, 0.25] if channel == "mu" else [0.25, 0.125])[r] * pred
    for k in raw:
        np.testing.assert_allclose(combined.predict_raw(X)[k], raw[k], atol=1e-7)
    assert not np.allclose(default.predict_raw(X)["mu"], scheduled.predict_raw(X)["mu"])
    assert not np.allclose(default.predict_raw(X)["mu"], bounded.predict_raw(X)["mu"])
    # Reconstruct unbounded first round through public tree/coefficient state.
    initial = NormalFisher().init_raw(y)
    first = {
        k: initial[k] + scheduled.coefficients_[k][0] * scheduled.trees_[k][0](scheduled.X_binned_)
        for k in raw
    }
    unbounded_next_mu = (first["mu"] - y) * np.exp(-2 * first["log_sigma"])
    assert not np.allclose(second_grad["mu"], unbounded_next_mu)
    out = Path("saved")
    out.mkdir(exist_ok=True)
    for name, model in [
        ("combined", combined),
        ("default", default),
        ("scheduled", scheduled),
        ("bounded", bounded),
    ]:
        model.save(out / (name + ".ob"))
        np.savez(out / (name + ".npz"), X=X, **model.predict_raw(X))


def test_early_stopping_and_roundtrip():
    X = np.arange(4, dtype=np.float32)[:, None]
    y = np.array([-3, -1, 1, 3], np.float32)
    model = Booster(
        objective=NormalFisher(),
        tree_builder=LevelWiseBuilder(),
        step_schedule=ChannelDecay(),
        config=TrainerConfig(n_trees=10, max_depth=1, learning_rate=0.5, min_child_weight=0),
    ).fit(X, y, eval_sets=[{"X": X, "y": -y}], early_stopping_rounds=1)
    assert model.best_iteration_ == 0
    assert model.coefficients_ == {"mu": [0.5], "log_sigma": [0.25]}
    assert all(len(t) == 1 for t in model.trees_.values())
    out = Path("saved")
    out.mkdir(exist_ok=True)
    model.save(out / "early.ob")
    np.savez(out / "early.npz", X=X, **model.predict_raw(X))
    with pytest.warns(UserWarning, match="trusted"):
        loaded = Booster.load(out / "early.ob")
    for k, v in model.predict_raw(X).items():
        np.testing.assert_array_equal(loaded.predict_raw(X)[k], v)
    Path("composition.json").write_text(
        json.dumps(
            {
                "models": 5,
                "early_best_iteration": model.best_iteration_,
                "channels": ["mu", "log_sigma"],
                "seed": 7,
                "samples": 4,
                "two_round_oracle": True,
            }
        )
    )
