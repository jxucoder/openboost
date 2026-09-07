"""Count current CPU tree replay and retained trace arrays; not a timing benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openboost import NumericData, Problem, RunContext
from openboost.recipes import normal, squared
from openboost.tree import Tree


def run(retention="full"):
    repo = Path(__file__).resolve().parents[2]
    x = np.arange(96, dtype=float).reshape(32, 3) / 32
    data = NumericData(x, np.arange(32), ("x0", "x1", "x2"))
    validation = NumericData(x[:16] + 0.01, np.arange(32, 48), data.feature_names)
    cases = []
    for name, recipe, width in (("squared", squared, 1), ("normal", normal, 2)):
        train = Problem(data, (1 + np.sin(x[:, 0]))[:, None], data.row_ids, raw_width=width)
        valid = Problem(
            validation,
            (1 + np.cos(x[:16, 0]))[:, None],
            validation.row_ids,
            raw_width=width,
        )
        for rounds in (4, 8, 16, 32):
            calls = []
            original = Tree.predict

            def counted(tree, inputs, *, calls=calls, original=original, **kwargs):
                calls.append(len(inputs.values))
                return original(tree, inputs, **kwargs)

            with patch.object(Tree, "predict", counted):
                result = recipe(
                    train, valid, context=RunContext("runtime-audit", 63), rounds=rounds,
                    patience=None, step="fixed", bins=8, max_depth=1, retention=retention,
                )
            # Outside the counted interval: independently replay the final models.
            np.testing.assert_array_equal(result.state.train_raw, result.state.model.predict(data))
            np.testing.assert_array_equal(
                result.state.validation_raw, result.state.model.predict(validation)
            )
            arrays = {
                id(value): value
                for step in result.steps
                for field in fields(step)
                if isinstance(value := getattr(step, field.name), np.ndarray)
            }
            assert result.state.version == rounds
            assert len(calls) == 2 * width * rounds
            cases.append(dict(
                recipe=name, retention=retention, rounds=rounds, train_rows=32, validation_rows=16,
                parameters=width, tree_terms=len(result.state.model.terms),
                tree_predict_calls_during_fit=len(calls), tree_row_visits=sum(calls),
                distinct_step_arrays=len(arrays),
                step_array_logical_bytes=sum(value.nbytes for value in arrays.values()),
                final_raw_exact=True,
            ))
    sources = [*sorted((repo / "src/openboost").glob("*.py")), Path(__file__)]
    return dict(
        scope="Synthetic operation counts only; no wall-time, peak-memory, quality or E-gate claim",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        source_sha256={
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        argv=[sys.executable, *sys.argv],
        environment=dict(
            python=platform.python_version(), os=platform.platform(),
            numpy=importlib.metadata.version("numpy"), cpu_count=os.cpu_count(),
            threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            gpu=None, peak_memory=None, wall_time=None,
        ),
        cases=cases,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--retention", choices=("full", "summary"), default="full")
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.retention), indent=2, allow_nan=False) + "\n")
