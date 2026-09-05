"""Small real-device tests, run outside the repository against an installed wheel."""

import gc
import hashlib
import importlib.metadata
import json
import site
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_device_interop(checks):
    import cupy as cp
    from numba import cuda

    import openboost as ob

    assert cuda.is_available()
    manifest = json.loads(Path("manifest.json").read_text())
    wheel = Path(manifest["wheel"])
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == manifest["wheel_sha256"]
    module = Path(ob.__file__).resolve()
    assert any(module.is_relative_to(Path(p).resolve()) for p in site.getsitepackages())
    dist = importlib.metadata.distribution("openboost")
    verified = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("openboost/") and name.endswith(".py"):
                assert Path(dist.locate_file(name)).read_bytes() == archive.read(name)
                verified += 1
    checks["installed_files_verified"] = verified
    checks["installed_module"] = str(module)

    owner = cp.arange(32, dtype=cp.float32)
    view = cuda.as_cuda_array(owner)
    assert view.__cuda_array_interface__["data"][0] == owner.data.ptr
    del owner
    gc.collect()

    @cuda.jit
    def increment(values):
        i = cuda.grid(1)
        if i < values.size:
            values[i] += 1

    increment[1, 32](view)
    cuda.synchronize()
    np.testing.assert_array_equal(view.copy_to_host(), np.arange(32, dtype=np.float32) + 1)
    checks["interop"] = True


def test_normal_gpu_fit(checks, monkeypatch):
    from numba import cuda

    import openboost as ob
    import openboost._trainer as trainer
    from openboost._objectives import DistributionObjective

    rng = np.random.default_rng(31)
    X = rng.normal(size=(256, 4)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=256)).astype(np.float32)
    checks["dataset_sha256"] = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    calls = {"objective": 0, "tree": 0}
    original_step = DistributionObjective.step
    original_tree = trainer.fit_tree_gpu_native

    def step(self, raw, target, *args, **kwargs):
        assert all(hasattr(a, "__cuda_array_interface__") for a in raw.values())
        output = original_step(self, raw, target, *args, **kwargs)
        assert self._device_kernels_ok, "Objective silently fell back to host"
        assert all(hasattr(a, "__cuda_array_interface__") for pair in output.values() for a in pair)
        calls["objective"] += 1
        return output

    def tree(binned, grad, hess, **kwargs):
        assert all(hasattr(a, "__cuda_array_interface__") for a in (binned, grad, hess))
        output = original_tree(binned, grad, hess, **kwargs)
        calls["tree"] += 1
        return output

    monkeypatch.setattr(DistributionObjective, "step", step)
    monkeypatch.setattr(trainer, "fit_tree_gpu_native", tree)
    with ob.backend_context("cuda"):
        assert ob.is_cuda()
        model = ob.NaturalBoostNormal(n_trees=2, max_depth=2).fit(X, y)
        prediction = model.predict(X)
        cuda.synchronize()
        assert np.all(np.isfinite(prediction))
        assert all(len(trees) == 2 for trees in model.trees_.values())
        checks["nll"] = float(model.nll(X, y))
        assert np.isfinite(checks["nll"])
    checks["native_tree_calls"] = calls["tree"]
    checks["device_objective_calls"] = calls["objective"]
    assert calls == {"objective": 2, "tree": 4}
