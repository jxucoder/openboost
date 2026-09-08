"""Collect installed, designer-authored D1/D2 observations without oracle inputs."""

import argparse
import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from openboost import NumericData, Problem, RunContext
from openboost.binning import Binning
from openboost.objectives import Squared
from openboost.recipes import squared
from openboost.stats import newton
from openboost.tree import best_first, depthwise, symmetric


def observe(case, plugin, destination):
    data = NumericData(
        np.asarray(case["values"], dtype=float), np.arange(len(case["values"])), ("x",)
    )
    problem = Problem(
        data,
        np.array(case["target"])[:, None],
        data.row_ids,
        weight=case["weight"],
        offset=np.array(case["offset"])[:, None],
    )
    d1 = case["task"] == "D1"
    objective = plugin.Expectile(case["tau"]) if d1 else Squared()
    options = dict(
        context=RunContext(case["id"], 19),
        rounds=case["rounds"],
        learning_rate=case["learning_rate"],
        bins=case["bins"],
    )
    if d1:
        result = plugin.fit(problem, problem, tau=case["tau"], **options)
    else:
        grower = dict(depthwise=depthwise, best_first=best_first, symmetric=symmetric)[
            case["policy"]
        ]
        result = squared(
            problem,
            problem,
            learner=plugin.CohortLearner(problem, case["information"], grower=grower, max_depth=2),
            **options,
        )

    def geometry(raw):
        if d1:
            loss, g, h = objective.geometry(problem, raw)
        else:
            loss = objective.loss(problem, raw)
            g, h = objective.gradient(problem, raw), np.ones(len(problem.target))
        return dict(loss=loss, gradient=g.reshape(-1).tolist(), curvature=h.reshape(-1).tolist())

    trace = []
    # Replaying prefixes checks every persisted update as well as the live state.
    raw = np.broadcast_to(result.state.model.base, (len(data.row_ids), 1)).copy()
    for index in range(len(result.state.model.terms)):
        record = geometry(raw)
        record["before"] = raw[:, 0].tolist()
        raw = replace(result.state.model, terms=result.state.model.terms[: index + 1]).predict(data)
        record["raw"] = raw[:, 0].tolist()
        trace.append(record)
    payload = dict(
        base=float(result.state.model.base[0]),
        trace=trace,
        raw=result.state.train_raw[:, 0].tolist(),
    )
    if d1:
        payload["geometry"] = geometry(np.array(case["geometry_raw"])[:, None])
    else:
        fields = []

        def capture(binned, statistics, **kwargs):
            fields.append(statistics)
            return grower(binned, statistics, **kwargs)

        learner = plugin.CohortLearner(problem, case["information"], grower=capture, max_depth=1)
        tree = learner(
            Binning.fit(data, bins=case["bins"]).transform(data),
            newton(problem, case["probe_gradient"], np.ones(len(data.row_ids))),
        )
        payload["probe"] = dict(
            root_cut=None if tree.feature[0] == -1 else int(tree.threshold[0]),
            prediction=tree.predict(data)[:, 0].tolist(),
            information=fields[0].values[:, -2:].tolist(),
        )
    result.state.model.save(Path(destination) / f"{case['id']}.model.json")
    return payload


def collect(inputs, destination, task, plugin=None):
    if plugin is None:
        plugin = importlib.import_module("ob_expectile" if task == "D1" else "ob_cohort_splits")
        if "site-packages" not in plugin.__file__:
            raise ValueError("development control must use an installed extension")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cases = [case for case in json.loads(Path(inputs).read_text()) if case["task"] == task]
    if not cases:
        raise ValueError("unknown or empty task")
    payload = {case["id"]: observe(case, plugin, destination) for case in cases}
    (destination / f"{task}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("task", choices=("D1", "D2"))
    args = parser.parse_args()
    collect(args.inputs, args.destination, args.task)
