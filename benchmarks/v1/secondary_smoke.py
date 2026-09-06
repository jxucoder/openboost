"""Weighted probabilistic and parametric comparator smoke on installed CPU packages."""

import importlib.metadata
import json
import pickle
import platform
from pathlib import Path

import numpy as np


def run():
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import LogScore
    from scipy.optimize import least_squares, minimize
    from scipy.special import log_ndtr
    from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(73)
    x = rng.normal(size=(80, 3))
    y = x[:, 0] + rng.normal(size=80) * 0.3
    w = np.linspace(0.5, 2, 80)
    records = []
    model = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        Base=DecisionTreeRegressor(max_depth=2, random_state=73),
        n_estimators=4,
        random_state=73,
        verbose=False,
    )
    model.fit(x, y, sample_weight=w)
    dist = model.pred_dist(x)
    nll = np.log(dist.scale) + 0.5 * ((y - dist.loc) / dist.scale) ** 2 + 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(nll, -dist.logpdf(y), rtol=1e-10, atol=1e-10)
    loaded = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(loaded.pred_dist(x).loc, dist.loc)
    records.append(
        {
            "case": "NGBoost weighted Normal",
            "status": "pass",
            "mean_nll": float(np.average(nll, weights=w)),
        }
    )
    exposure = np.linspace(0.1, 2, 80)
    count = rng.poisson(exposure * np.exp(x[:, 0] / 3))
    positive = np.exp(y / 3)
    for name, klass, target, weight, kwargs in [
        ("Poisson rate/exposure", PoissonRegressor, count / exposure, w * exposure, {}),
        ("Gamma severity", GammaRegressor, positive, w, {}),
        (
            "Tweedie annualized",
            TweedieRegressor,
            count / exposure,
            w * exposure,
            {"power": 1.5, "link": "log"},
        ),
    ]:
        m = klass(alpha=0.1, max_iter=1000, **kwargs).fit(x, target, sample_weight=weight)
        before = m.predict(x)
        after = pickle.loads(pickle.dumps(m)).predict(x)
        assert np.isfinite(before).all() and np.all(before > 0)
        np.testing.assert_array_equal(before, after)
        records.append(
            {
                "case": name,
                "status": "pass",
                "reload_max_abs_error": float(np.max(np.abs(before - after))),
            }
        )
    # Fixed-sigma log-normal linear AFT, independent of tree implementations.
    design = np.column_stack([np.ones(80), x])
    observed = np.exp(y)
    event = np.arange(80) % 4 != 0

    def loss(beta):
        z = np.log(observed) - design @ beta
        terms = np.where(
            event, np.log(observed) + z * z / 2 + 0.5 * np.log(2 * np.pi), -log_ndtr(-z)
        )
        return float(np.average(terms, weights=w))

    fit = minimize(loss, np.zeros(4), method="BFGS")
    if not fit.success or not np.isfinite(fit.fun):
        raise ValueError("AFT optimizer failed: " + fit.message)
    records.append(
        {"case": "fixed-sigma linear lognormal AFT", "status": "pass", "mean_nll": float(fit.fun)}
    )
    age = np.linspace(0.1, 4, 80)
    truth = 3 * (-np.expm1(-0.7 * age))

    def residual(raw):
        a, b = np.logaddexp(0, raw)
        return np.sqrt(w) * (a * (-np.expm1(-b * age)) - truth)

    fit = least_squares(residual, [1.0, 0.0], max_nfev=2000)
    np.testing.assert_allclose(np.logaddexp(0, fit.x), [3, 0.7], rtol=1e-6, atol=1e-7)
    records.append(
        {
            "case": "global saturating formula",
            "status": "pass",
            "max_residual": float(np.max(np.abs(fit.fun))),
        }
    )
    return {
        "scope": "tiny synthetic comparator smoke, not real quality",
        "records": records,
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
    }


if __name__ == "__main__":
    import hashlib
    import subprocess

    result = run()
    result["source_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result["source_file_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"]))
    Path("benchmarks/v1/evidence/secondary-cpu.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(result["records"])
