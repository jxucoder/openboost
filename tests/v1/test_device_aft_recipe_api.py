"""AFT recipe rejects unsupported options before any device work."""

from dataclasses import replace

import pytest

from openboost import device_recipes

from .test_device_aft_reference import fixture


@pytest.mark.parametrize("config, message", [
    ({"sigma": 0}, "scale"), ({"max_depth": -1}, "max_depth"),
    ({"max_trials": 7}, "trials"), ({"step": "unknown"}, "step"),
    ({"learning_rate": -1}, "nonnegative"), ({"patience": 0}, "patience"),
    ({"learner": lambda *_: None, "max_depth": 1}, "owns"),
])
def test_invalid_recipe_configuration_precedes_device_allocation(config, message):
    p = fixture()
    with pytest.raises(ValueError, match=message):
        device_recipes.aft(None, p, p, run_id="bad", seed=7, **config)


@pytest.mark.parametrize("invalid", ["target", "unknown"])
def test_target_scope_and_unknown_arguments_are_explicit(invalid):
    p = fixture()
    if invalid == "target":
        p = replace(p, target=p.target[:, :1], target_kind="numeric")
        with pytest.raises(ValueError, match="AFT"):
            device_recipes.aft(None, p, p, run_id="bad", seed=7)
    else:
        with pytest.raises(TypeError, match="unexpected"):
            device_recipes.aft(None, p, p, run_id="bad", seed=7, ignored=True)


@pytest.mark.parametrize("event", [False, True])
def test_actual_controlled_recipe_host_fixture_is_valid(event, monkeypatch):
    from openboost import device_aft

    from .test_device_aft_recipes_cuda import controlled

    p, binning, learner = controlled(monkeypatch, event=event, initial=0, values=(1e-20, 2e-20))
    device_aft.objective(.7).validate(p)
    assert p.raw_width == 1 and p.target_kind == "event_right"
    assert ((p.target[:, 0] == p.target[:, 1]) == event).all()
    assert callable(learner) and binning.feature_names == p.data.feature_names
