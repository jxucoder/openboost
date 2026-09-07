"""Current CPU matched-payment composition fit and inference-only replay."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from openboost import NumericData, RunContext
from openboost.composition import FrequencySeverity, paid_loss_problems


def fit(job, arrays):
    from openboost.recipes import gamma, poisson

    if (
        set(job) != {"seed", "config", "patience"}
        or type(job["seed"]) is not int
        or job["seed"] < 0
    ):
        raise ValueError("explicit composition seed/config/patience required")
    cfg = job["config"]
    if set(cfg) != {"rounds", "learning_rate", "bins", "max_depth", "reg_lambda"}:
        raise ValueError("unsupported composition configuration")
    if type(cfg["rounds"]) is not int or cfg["rounds"] <= 0:
        raise ValueError("positive round budget required")
    roles = {"x", "row_ids", "paid_count", "paid_total", "exposure"}
    if set(arrays) != {f"{r}_{p}" for r in roles for p in ("train", "validation")}:
        raise ValueError("matched training/validation roles only")
    problems = []
    names = None
    for part in ("train", "validation"):
        x, ids = arrays["x_" + part], arrays["row_ids_" + part]
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("nonempty finite encoded features required")
        names = tuple(f"x{i}" for i in range(x.shape[1])) if names is None else names
        data = NumericData(x, ids, names)
        problems.append(
            paid_loss_problems(
                data,
                arrays["paid_count_" + part],
                arrays["paid_total_" + part],
                arrays["exposure_" + part],
            )
        )
    if np.intersect1d(arrays["row_ids_train"], arrays["row_ids_validation"]).size:
        raise ValueError("policy partitions overlap")
    models, components = [], {}
    selection = "final" if job["patience"] is None else "best_component_validation"
    for i, (name, recipe) in enumerate((("frequency", poisson), ("severity", gamma))):
        result = recipe(
            problems[0][i],
            problems[1][i],
            context=RunContext(name, job["seed"]),
            patience=job["patience"],
            retention="summary",
            **cfg,
        )
        model = result.state.model if job["patience"] is None else result.state.best_model
        models.append(model)
        components[name] = dict(
            stop={**asdict(result.stop), "reason": result.stop.reason},
            selected_identity=model.identity,
            best_score=result.state.best_score,
            accepted_commits=result.state.version,
            diagnostic_retention="summary",
        )
    model = FrequencySeverity(*models)
    data = problems[1][0].data
    outputs = model.predict(data, data, arrays["exposure_validation"])
    return model, outputs, dict(selection=selection, joint_selection=False, components=components)


def replay(model_path, packet):
    if set(packet) != {"x", "row_ids", "exposure"}:
        raise ValueError("inference features, policy IDs and exposure only")
    model = FrequencySeverity.load(model_path)
    data = NumericData(packet["x"], packet["row_ids"], model.frequency.feature_names)
    return model.predict(data, data, packet["exposure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "predict"))
    parser.add_argument("spec", type=Path)
    parser.add_argument("packet", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    with np.load(args.packet, allow_pickle=False) as a:
        arrays = dict(a)
    if args.mode == "predict":
        if args.output is None:
            raise ValueError("prediction output path required")
        outputs = replay(args.spec, arrays)
        np.savez(args.output, row_ids=arrays["row_ids"], **outputs)
    else:
        if args.output is not None:
            raise ValueError("fit writes to its dedicated working directory")
        model, outputs, training = fit(json.loads(args.spec.read_text()), arrays)
        model.save("model.bin")
        np.savez("predictions.npz", row_ids=arrays["row_ids_validation"], **outputs)
        Path("training.json").write_text(json.dumps(training, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
