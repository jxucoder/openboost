"""Installed independent packages, real GPU math and full shared-trainer fits."""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_installed_gpu_extensions(checks, monkeypatch):
    import bounded_leaves
    import cupy as cp
    import normal_fisher
    from bounded_leaves import BoundedNewton
    from normal_fisher import ChannelDecay, NormalFisher
    from scipy.special import ndtr

    from openboost.experimental import Booster, ExecutionContext, LevelWiseBuilder, TrainerConfig

    manifest = json.loads(Path("manifest.json").read_text())
    installed = {}
    for module, distname in [
        (normal_fisher, "openboost-example-normal-fisher"),
        (bounded_leaves, "openboost-example-bounded-leaves"),
    ]:
        location = Path(module.__file__)
        assert "site-packages" in location.parts
        dist = importlib.metadata.distribution(distname)
        wheel = next(
            n
            for n in manifest["extension_wheels"]
            if n.startswith(distname.replace("-", "_") + "-")
        )
        with zipfile.ZipFile(wheel) as archive:
            count = 0
            for name in archive.namelist():
                if name.endswith(".py"):
                    assert archive.read(name) == Path(dist.locate_file(name)).read_bytes()
                    count += 1
        assert count > 0
        installed[module.__name__] = {
            "version": dist.version,
            "verified_python_files": count,
            "path": str(location.relative_to(sys.prefix)),
        }
    obj = NormalFisher()
    ctx = ExecutionContext("cuda", cp, np.random.default_rng(7), 0)
    y = np.array([-1, 2, 4], np.float32)
    w = np.array([0, 0.5, 2], np.float32)
    raw = {
        "mu": np.array([0.2, 0.4, 0.1], np.float32),
        "log_sigma": np.array([-0.3, 0.2, 0.5], np.float32),
    }
    stats = obj.step(
        {k: cp.asarray(v) for k, v in raw.items()}, cp.asarray(y), cp.asarray(w), context=ctx
    )

    def nll(values):
        return w * (
            values["log_sigma"]
            + 0.5 * ((y - values["mu"]) * np.exp(-values["log_sigma"])) ** 2
            + 0.5 * np.log(2 * np.pi)
        )

    for k in raw:
        plus, minus = ({j: a.astype(float) for j, a in raw.items()} for _ in range(2))
        plus[k] += 1e-5
        minus[k] -= 1e-5
        np.testing.assert_allclose(
            cp.asnumpy(stats[k][0]), (nll(plus) - nll(minus)) / 2e-5, atol=1e-6, rtol=2e-6
        )
        assert all(isinstance(a, cp.ndarray) and a.dtype == cp.float32 for a in stats[k])
    np.testing.assert_allclose(
        cp.asnumpy(stats["mu"][1]), w * np.exp(-2 * raw["log_sigma"]), rtol=1e-6
    )
    np.testing.assert_array_equal(cp.asnumpy(stats["log_sigma"][1]), 2 * w)
    gpu_raw = {k: cp.asarray(v) for k, v in raw.items()}
    np.testing.assert_allclose(
        obj.loss_value(gpu_raw, cp.asarray(y), cp.asarray(w), context=ctx),
        nll({k: v.astype(float) for k, v in raw.items()}).sum() / w.sum(),
    )
    constrained = obj.constrain(gpu_raw)
    assert isinstance(constrained["sigma"], cp.ndarray)
    np.testing.assert_allclose(
        cp.asnumpy(constrained["sigma"]), np.exp(raw["log_sigma"].astype(float))
    )
    with pytest.raises(ValueError, match="device"):
        obj.step(raw, y, w, context=ctx)
    for logs in (-1000, 1000):
        with pytest.raises(ValueError):
            obj.step(
                {"mu": cp.ones(2, cp.float32), "log_sigma": cp.full(2, logs, cp.float32)},
                cp.ones(2, cp.float32),
                context=ctx,
            )

    class RecordedNormal(NormalFisher):
        def __init__(self):
            self.gradients = []

        def step(self, *args, **kwargs):
            out = super().step(*args, **kwargs)
            self.gradients.append(out["mu"][0].copy())
            return out

    copy_to_host = cp.asnumpy
    downloads = []
    records = []
    saved = Path("gpu_saved")
    saved.mkdir(exist_ok=True)
    for samples in (16, 4097):
        rng = np.random.default_rng(149)
        X = rng.normal(size=(samples, 3)).astype(np.float32)
        y = (1.2 * X[:, 0] + 0.4 * rng.normal(size=samples)).astype(np.float32)
        weights = rng.choice(np.array([0, 0.5, 1, 2], np.float32), samples)
        outputs, next_gradients = {}, {}
        cfg = TrainerConfig(n_trees=2, max_depth=2, learning_rate=0.2, n_bins=32, random_state=7)
        for bounded, scheduled in [(False, False), (False, True), (True, False), (True, True)]:
            result = []
            for device in ("cpu", "cuda"):
                objective = RecordedNormal()
                model = Booster(
                    objective=objective,
                    tree_builder=LevelWiseBuilder(
                        leaf_rule=BoundedNewton(0.1) if bounded else None
                    ),
                    step_schedule=ChannelDecay() if scheduled else None,
                    config=cfg,
                    device=device,
                )

                def compact_only(a, *args, **kwargs):
                    assert a.shape == (7,) and a.dtype in (cp.int32, cp.float32)
                    downloads.append(a.nbytes)
                    return copy_to_host(a, *args, **kwargs)

                with monkeypatch.context() as m:
                    if device == "cuda":
                        m.setattr(cp, "asnumpy", compact_only)
                    model.fit(X, y, sample_weight=weights)
                prediction = model.predict_raw(X)
                result.append(prediction)
                if device == "cuda":
                    assert model.fit_report_["actual_device"] == "cuda"
                    assert model.fit_report_["fallback_reason"] is None
                    if scheduled:
                        assert model.coefficients_ == {"mu": [0.2, 0.1], "log_sigma": [0.1, 0.05]}
                    if bounded:
                        assert all(
                            np.max(np.abs(tree.values)) <= 0.100001
                            for trees in model.trees_.values()
                            for tree in trees
                        )
                    next_gradients[bounded, scheduled] = copy_to_host(objective.gradients[1])
                    outputs[bounded, scheduled] = prediction
                    path = saved / f"{samples}-{bounded}-{scheduled}.ob"
                    model.save(path)
                    np.savez(path.with_suffix(".npz"), X=X, **prediction)
            for channel in result[0]:
                np.testing.assert_allclose(
                    result[1][channel], result[0][channel], atol=2e-5, rtol=2e-5
                )

            def metrics(pred, y=y, weights=weights):
                sigma = np.exp(pred["log_sigma"].astype(float))
                z = (y - pred["mu"]) / sigma
                return {
                    "nll": float(
                        np.average(
                            pred["log_sigma"] + 0.5 * z * z + 0.5 * np.log(2 * np.pi),
                            weights=weights,
                        )
                    ),
                    "crps": float(
                        np.average(
                            sigma
                            * (
                                z * (2 * ndtr(z) - 1)
                                + 2 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
                                - 1 / np.sqrt(np.pi)
                            ),
                            weights=weights,
                        )
                    ),
                }

            cpu_metrics, gpu_metrics = metrics(result[0]), metrics(result[1])
            for metric in cpu_metrics:
                np.testing.assert_allclose(
                    gpu_metrics[metric], cpu_metrics[metric], atol=2e-5, rtol=2e-5
                )
            records.append(
                {
                    "samples": samples,
                    "seed": 149,
                    "bounded": bounded,
                    "scheduled": scheduled,
                    "data_sha256": hashlib.sha256(
                        X.tobytes() + y.tobytes() + weights.tobytes()
                    ).hexdigest(),
                    "cpu": cpu_metrics,
                    "cuda": gpu_metrics,
                    "max_raw_error": max(
                        float(np.max(np.abs(result[0][k] - result[1][k]))) for k in result[0]
                    ),
                }
            )
        assert not np.allclose(outputs[False, False]["mu"], outputs[False, True]["mu"])
        assert not np.allclose(outputs[False, True]["mu"], outputs[True, True]["mu"])
        assert not np.allclose(next_gradients[False, True], next_gradients[True, True])
    assert len(downloads) == 160
    checks["installed_gpu_extensions"] = {
        "installed": installed,
        "math_oracle": True,
        "cases": records,
        "clipping_changes_next_gradient": True,
        "schedule_changes_prediction": True,
        "compact_transfer_calls": len(downloads),
        "compact_transfer_bytes": sum(downloads),
        "models_saved": 8,
    }
    demo = subprocess.run(
        [sys.executable, "extension_demo.py", "--device", "cuda"],
        text=True,
        capture_output=True,
    )
    assert demo.returncode == 0, demo.stdout + demo.stderr
    for suffix in ("ob", "npz"):
        Path("demo." + suffix).rename(saved / ("demo." + suffix))
    checks["installed_gpu_extensions"].update(demo=json.loads(demo.stdout), models_saved=9)
