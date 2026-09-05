"""Coefficient persistence and inference without training plugin objects."""

import joblib
import numpy as np
import pytest

import openboost as ob
from openboost.experimental import Booster, TrainerConfig
from tests.test_experimental_dispatch import Decay, PresetBuilder
from tests.test_experimental_objective import TwoSquared


class NoPickleObjective(TwoSquared):
    def __reduce__(self):
        raise AssertionError("Do not serialize the objective")


class NoPickleBuilder(PresetBuilder):
    def __reduce__(self):
        raise AssertionError("Do not serialize the builder")


class NoPickleSchedule(Decay):
    def __reduce__(self):
        raise AssertionError("Do not serialize the schedule")


def test_plugins_excluded_and_nonconstant_coefficients_roundtrip(tmp_path):
    X, y = np.zeros((4, 1)), np.ones(4)
    model = Booster(
        objective=NoPickleObjective(),
        tree_builder=NoPickleBuilder(),
        step_schedule=NoPickleSchedule(),
        config=TrainerConfig(n_trees=2, learning_rate=0.5),
    ).fit(X, y)
    path = tmp_path / "model.ob"
    model.save(path)
    state = joblib.load(path)
    assert not {"objective", "tree_builder", "step_schedule", "config"} & set(state)
    with pytest.warns(UserWarning, match="trusted"):
        loaded = Booster.load(path)
    assert loaded.coefficients_ == model.coefficients_
    for k, v in model.predict_raw(X).items():
        np.testing.assert_array_equal(loaded.predict_raw(X)[k], v)
    with pytest.raises(RuntimeError, match="inference"):
        loaded.fit(X, y)


def test_early_stop_restores_coefficients_and_checkpoint(tmp_path):
    from openboost._callbacks import ModelCheckpoint

    class Curve(TwoSquared):
        calls = 0

        def loss_value(self, raw, y, *args, **kwargs):
            if np.all(y < 0):
                result = self.calls
                self.calls += 1
                return float(result)
            return super().loss_value(raw, y, *args, **kwargs)

    X, y = np.zeros((4, 1)), np.ones(4)
    path = tmp_path / "best.ob"
    model = Booster(
        objective=Curve(),
        tree_builder=PresetBuilder(),
        step_schedule=Decay(),
        config=TrainerConfig(n_trees=5, learning_rate=0.5),
    )
    model.fit(
        X,
        y,
        eval_sets=[{"X": X, "y": -y}],
        callbacks=[ModelCheckpoint(str(path))],
        early_stopping_rounds=1,
    )
    assert model.best_iteration_ == 0
    assert all(len(t) == 1 for t in model.trees_.values())
    assert model.coefficients_ == {"a": [0.5], "b": [0.25]}
    assert model.fit_report_["tree_counts"] == {"a": 1, "b": 1}
    with pytest.warns(UserWarning, match="trusted"):
        loaded = Booster.load(path)
    for k, v in model.predict_raw(X).items():
        np.testing.assert_array_equal(loaded.predict_raw(X)[k], v)


@pytest.mark.parametrize("kind", ["numeric", "missing", "categorical"])
def test_default_tree_and_binner_state_roundtrip(tmp_path, kind):
    rng = np.random.default_rng(19)
    X = rng.normal(size=(64, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    if kind == "missing":
        X[::4, 0] = np.nan
    raw_X = X.copy()
    if kind == "categorical":
        X[:, 0] = np.arange(len(X)) % 3
        raw_X = X.copy()
        X = ob.array(X, categorical_features=[0])
    model = Booster(
        objective=TwoSquared(), step_schedule=Decay(), config=TrainerConfig(n_trees=3, max_depth=2)
    ).fit(X, y)
    path = tmp_path / "state.ob"
    model.save(path)
    with pytest.warns(UserWarning, match="trusted"):
        loaded = Booster.load(path)
    for k, v in model.predict_raw(raw_X).items():
        np.testing.assert_array_equal(loaded.predict_raw(raw_X)[k], v)


def test_legacy_missing_coefficients_and_invalid_counts(tmp_path):
    X, y = np.zeros((4, 1)), np.ones(4)
    model = Booster(objective=TwoSquared(), config=TrainerConfig(n_trees=2)).fit(X, y)
    state = model._to_state_dict()
    state.pop("coefficients_")
    path = tmp_path / "legacy.ob"
    joblib.dump(state, path)
    with pytest.warns(UserWarning, match="trusted"):
        loaded = Booster.load(path)
    assert loaded.coefficients_ == {"a": [0.1, 0.1], "b": [0.1, 0.1]}
    for k, v in model.predict_raw(X).items():
        np.testing.assert_array_equal(loaded.predict_raw(X)[k], v)
    state["coefficients_"] = {"a": [0.1], "b": [0.1]}
    joblib.dump(state, path)
    with pytest.warns(UserWarning, match="trusted"), pytest.raises(ValueError, match="Coefficient"):
        Booster.load(path)


def test_callback_round_begin_and_lr_mutation():
    from openboost._callbacks import Callback

    class Watch(Callback):
        def __init__(self):
            self.rounds = []

        def on_round_begin(self, state):
            self.rounds.append(state.round_idx)

    cb = Watch()
    Booster(objective=TwoSquared(), config=TrainerConfig(n_trees=2)).fit(
        np.zeros((4, 1)), np.ones(4), callbacks=[cb]
    )
    assert cb.rounds == [0, 1]


@pytest.mark.parametrize("hook", ["on_round_end", "on_train_end"])
def test_callback_cannot_silently_change_learning_rate(hook):
    from openboost._callbacks import Callback

    def mutate(self, state):
        state.model.learning_rate = 12
        return True

    cb = type("Mutate", (Callback,), {hook: mutate})()
    model = Booster(objective=TwoSquared(), config=TrainerConfig(n_trees=1))
    with pytest.raises(ValueError, match="StepSchedule"):
        model.fit(np.zeros((4, 1)), np.ones(4), callbacks=[cb])
    assert model.trees_ == {}


@pytest.mark.parametrize(
    "changes",
    [
        {"_experimental_version": 2},
        {"_serialization_version": 99},
        {"_serialization_version": 1, "_is_categorical": np.array([True])},
        {"coefficients_": {"a": [float("nan")], "b": [0.1]}},
    ],
)
def test_invalid_persistence_state_rejected(changes):
    model = Booster(objective=TwoSquared(), config=TrainerConfig(n_trees=1)).fit(
        np.zeros((4, 1)), np.ones(4)
    )
    state = model._to_state_dict()
    state.update(changes)
    with pytest.raises(ValueError):
        Booster.__new__(Booster)._from_state_dict(state)
