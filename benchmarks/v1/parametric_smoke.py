"""Hand-check exposure-weighted GLM/composition controls and formula persistence."""

import hashlib
import importlib.metadata
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.v1.parametric import (
    fit_glm,
    fit_global_formula,
    fit_paid_composition,
    predict_glm,
    predict_global_formula,
    predict_paid_composition,
)

if __name__ == "__main__":
    x = np.ones((3, 2))
    counts = np.array([1, 2, 0])
    totals = np.array([10.0, 60.0, 0.0])
    e, w = np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0])
    index, amount = np.array([0, 1, 1]), np.array([10.0, 20.0, 40.0])
    rows = []
    for task, inputs, target, weight, exposure in [
        ("A7", x, counts, w, e),
        ("A8", x[index], amount, w[index], None),
        ("A9", x, totals, w, e),
    ]:
        saved = fit_glm(task, inputs, target, weight=weight, exposure=exposure, alpha=0.0)
        p = predict_glm(saved, inputs, exposure)
        replay = predict_glm(pickle.loads(pickle.dumps(saved)), inputs, exposure)
        for k in p:
            np.testing.assert_array_equal(p[k], replay[k])
        expected = np.dot(weight, target) / (
            weight.sum() if exposure is None else np.dot(weight, exposure)
        )
        actual = p["mean"] if task == "A8" else p["annualized"]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-8)
        rows.append(
            dict(task=task, status="pass", expected_mean=float(expected), actual=actual.tolist())
        )
    saved = fit_paid_composition(
        x, counts, totals, e, index, amount, weight=w, count_alpha=0.0, severity_alpha=0.0
    )
    p = predict_paid_composition(saved, x, e)
    replay = predict_paid_composition(pickle.loads(pickle.dumps(saved)), x, e)
    expected = np.dot(w, totals) / np.dot(w, e)
    np.testing.assert_allclose(p["annualized"], expected, rtol=1e-4)
    doubled = predict_paid_composition(saved, x, 2 * e)
    np.testing.assert_array_equal(doubled["annualized"], p["annualized"])
    np.testing.assert_array_equal(doubled["period"], 2 * p["period"])
    for k in p:
        np.testing.assert_array_equal(p[k], replay[k])
    rows.append(
        dict(
            task="A9 paid-count times severity",
            status="pass",
            expected_annualized=float(expected),
            prediction=p["annualized"].tolist(),
        )
    )
    age = np.linspace(0.1, 4, 40)
    saved = fit_global_formula(age, 3 * -np.expm1(-0.7 * age))
    reloaded = json.loads(json.dumps(saved))
    np.testing.assert_array_equal(
        predict_global_formula(saved, age), predict_global_formula(reloaded, age)
    )
    np.testing.assert_allclose([saved["amplitude"], saved["rate"]], [3.0, 0.7], rtol=1e-6)
    rows.append(dict(task="A12 global formula", status="pass", parameters=saved))
    result = dict(
        scope="synthetic CPU parametric controls, not real A9/A12 quality",
        cells=rows,
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        source_hashes={
            n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in ["parametric.py", "parametric_smoke.py"]
        },
    )
    Path("benchmarks/v1/evidence/parametric-cpu.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(rows)
