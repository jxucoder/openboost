"""Scalar consumer preflight does not require a device or silently select reporting loss."""

from dataclasses import replace

import pytest

from openboost import device_glm as glm
from openboost import device_recipes as recipes


@pytest.mark.parametrize("family", ["binary", "poisson"])
@pytest.mark.parametrize("missing", ["fields", "compare"])
def test_required_dependency_checked_before_preparation(family, missing, monkeypatch):
    configured = replace(getattr(glm, family)(), **{missing: None})
    monkeypatch.setattr(glm, family, lambda **kwargs: configured)
    with pytest.raises(NotImplementedError, match="fields and loss-change"):
        getattr(recipes, family)(None, None, None, run_id="preflight", seed=7)


@pytest.mark.parametrize("family", ["binary", "poisson"])
@pytest.mark.parametrize(
    "configuration",
    [
        dict(rounds=-1),
        dict(max_trials=7),
        dict(step="typo"),
        dict(learning_rate=-1),
        dict(max_depth=-1),
        dict(patience=0),
        dict(min_delta=1),
        dict(learner="wrong"),
        dict(learner=lambda *args: None, max_depth=1),
    ],
)
def test_invalid_recipe_configuration_fails_without_device(family, configuration):
    with pytest.raises(ValueError):
        getattr(recipes, family)(None, None, None, run_id="invalid", seed=7, **configuration)


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_unknown_parameter_is_not_ignored(family):
    with pytest.raises(TypeError, match="unexpected keyword"):
        getattr(recipes, family)(None, None, None, run_id="invalid", seed=7, ignored=True)
