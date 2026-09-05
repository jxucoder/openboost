"""Launcher parameters must reach model constructors, not only provenance."""

import sys
from types import ModuleType

import pytest
from benchmarks.scoringbench.run import _build_parser, _model_factories
from sklearn.tree import DecisionTreeRegressor


@pytest.fixture
def wrappers(monkeypatch):
    class Recorded:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    for module_name, class_name in (
        ("benchmarks.scoringbench.openboost_wrapper", "OpenBoostWrapper"),
        ("scoringbench.wrappers.ngboost_wrapper", "NGBoostWrapper"),
        ("scoringbench.wrappers.xgblss_wrapper", "XGBLSSWrapper"),
        ("scoringbench.wrappers.catboost_wrapper", "CatBoostQuantileWrapper"),
    ):
        module = ModuleType(module_name)
        setattr(module, class_name, Recorded)
        monkeypatch.setitem(sys.modules, module_name, module)
    learners = ModuleType("ngboost.learners")
    learners.default_tree_learner = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2)
    monkeypatch.setitem(sys.modules, "ngboost.learners", learners)
    return learners.default_tree_learner


def test_declared_seed_and_depth_reach_every_model(wrappers):
    args = _build_parser().parse_args(
        [
            "--models",
            "openboost_cpu,openboost_cuda,ngboost,xgblss,catboost_quantile",
            "--seed",
            "73",
            "--max-depth",
            "5",
            "--n-trees",
            "17",
            "--learning-rate",
            ".03",
        ]
    )
    models = {name: make().kwargs for name, make in _model_factories(args).items()}
    for name in ("openboost_cpu", "openboost_cuda"):
        assert models[name]["model_params"]["random_state"] == 73
        assert models[name]["max_depth"] == 5
        assert models[name]["n_trees"] == 17
    ngb = models["ngboost"]["ngb_params"]
    assert ngb["random_state"] == 73
    assert ngb["Base"].max_depth == 5 and ngb["Base"].random_state == 73
    assert ngb["Base"].min_samples_leaf == wrappers.min_samples_leaf
    assert ngb["Base"] is not wrappers
    assert wrappers.max_depth == 3 and wrappers.random_state is None
    assert models["xgblss"]["xgblss_params"]["seed"] == 73
    assert models["xgblss"]["xgblss_params"]["max_depth"] == 5
    assert models["catboost_quantile"]["catboost_params"]["random_seed"] == 73
    assert models["catboost_quantile"]["catboost_params"]["depth"] == 5


def test_ngboost_factories_do_not_share_mutable_base_learner(wrappers):
    args = _build_parser().parse_args(["--models", "ngboost"])
    make = _model_factories(args)["ngboost"]
    first, second = make().kwargs, make().kwargs
    assert first["ngb_params"]["Base"] is not second["ngb_params"]["Base"]


def test_configuration_records_base_learner_without_private_fitted_state():
    import json

    from benchmarks.scoringbench.run import _model_configuration

    class Wrapper:
        def __init__(self):
            self.ngb_params = {"Base": DecisionTreeRegressor(max_depth=5, random_state=73)}
            self._model = object()

    recorded = json.loads(json.dumps(_model_configuration(Wrapper())))
    params = recorded["parameters"]["ngb_params"]["Base"]["parameters"]
    assert params["max_depth"] == 5 and params["random_state"] == 73
    assert "_model" not in recorded["parameters"]


def test_factory_seed_controls_actual_openboost_fit(monkeypatch, wrappers):
    """Exercise real wrapper/core fit while isolating optional upstream metrics imports."""
    import importlib.util
    from pathlib import Path

    import numpy as np

    base = ModuleType("scoringbench.wrappers.base")
    base.DistributionPrediction = object
    base.ProbabilisticWrapper = object
    quantile = ModuleType("scoringbench.wrappers.quantile_based")
    quantile.quantiles_to_distribution = lambda *a, **k: None  # not used by fit/predict
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, quantile.__name__, quantile)
    path = Path(__file__).resolve().parents[1] / "benchmarks/scoringbench/openboost_wrapper.py"
    spec = importlib.util.spec_from_file_location("_scoringbench_fit_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "benchmarks.scoringbench.openboost_wrapper", module)

    args = _build_parser().parse_args(
        ["--models", "openboost_cpu", "--seed", "73", "--n-trees", "5"]
    )
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=80)).astype(np.float32)
    predictions = []
    global_state = np.random.get_state()
    try:
        for global_seed in (3, 991):
            np.random.seed(global_seed)
            model = _model_factories(args)["openboost_cpu"]()
            model.model_params["subsample"] = 0.65  # ensure the declared seed is exercised
            model.fit(X, y)
            assert model._model.random_state == 73
            predictions.append(model.predict(X))
        np.testing.assert_array_equal(*predictions)
        args.seed = 74
        other = _model_factories(args)["openboost_cpu"]()
        other.model_params["subsample"] = 0.65
        other.fit(X, y)
        assert not np.array_equal(other.predict(X), predictions[0])
    finally:
        np.random.set_state(global_state)


def test_manifest_persists_constructed_configuration(tmp_path, monkeypatch):
    import json

    from benchmarks.scoringbench import run

    args = _build_parser().parse_args(["--models", "openboost_cpu", "--seed", "73"])
    configured = {"openboost_cpu": {"parameters": {"model_params": {"random_state": 73}}}}
    monkeypatch.setattr(run, "_gpu_info", lambda: None)
    monkeypatch.setattr(run, "_git_state", lambda path: {"commit": "test", "dirty": False})
    path = run._write_provenance(tmp_path, tmp_path, args, [], 0, configured)
    manifest = json.loads(path.read_text())
    assert manifest["model_parameters"] == configured
    assert manifest["arguments"]["seed"] == 73
