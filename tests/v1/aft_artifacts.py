"""Retain AFT inputs and comparison/recipe reports through the bounded collector."""

import os
from pathlib import Path

from .glm_artifacts import input_snapshot as input_snapshot


def directory(kind, temporary):
    if kind not in ("comparisons", "recipes"):
        raise ValueError("AFT comparison or recipe artifact kind required")
    explicit = os.environ.get("OPENBOOST_AFT_ARTIFACTS")
    retained = os.environ.get("OPENBOOST_NORMAL_ARTIFACTS")
    root = Path(explicit) if explicit else Path(retained) if retained else Path(temporary)
    folder = root / ("aft-" + kind)
    folder.mkdir(parents=True, exist_ok=True)
    return folder
