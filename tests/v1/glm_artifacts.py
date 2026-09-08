"""Route GLM evidence into the existing bounded hardware collector."""

import base64
import os
from pathlib import Path

import numpy as np


def input_snapshot(problem):
    """Retain actual fixture bytes, including missing-value bit patterns."""
    arrays = dict(
        features=problem.data.values,
        row_ids=problem.data.row_ids,
        target=problem.target,
        offset=problem.offset,
        weight=problem.weight,
        **problem.structure,
    )
    result = {}
    for name, value in arrays.items():
        a = np.ascontiguousarray(value)
        result[name] = dict(
            dtype=a.dtype.str,
            shape=list(a.shape),
            data_base64=base64.b64encode(a.tobytes()).decode(),
        )
    return result


def directory(kind, temporary):
    names = {"comparisons": "COMPARISON", "recipes": "RECIPE"}
    explicit = os.environ.get(f"OPENBOOST_GLM_{names[kind]}_ARTIFACTS")
    retained = os.environ.get("OPENBOOST_NORMAL_ARTIFACTS")
    folder = (
        Path(explicit)
        if explicit
        else (Path(retained) / ("glm-" + kind) if retained else Path(temporary))
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder
