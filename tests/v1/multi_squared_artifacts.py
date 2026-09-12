"""Lossless multi-output evidence routed through the bounded device collector."""

import os
import sys
from pathlib import Path

from .glm_artifacts import input_snapshot as input_snapshot


def fresh_command(script, *arguments):
    """Use the installed CPU-only interpreter on hardware and deny device training."""
    guard = "\n".join(
        [
            "import sys",
            "for name in ('cupy', 'numba', 'openboost.device', 'openboost.device_multi_squared',",
            "             'openboost.device_runtime', 'openboost.device_recipes', 'openboost.recipes'):",
            "    sys.modules[name] = None",
        ]
    )
    return [
        os.environ.get("OPENBOOST_FRESH_CPU_PYTHON", sys.executable),
        "-I",
        "-c",
        guard + "\n" + script,
        *arguments,
    ]


def directory(kind, temporary):
    if kind not in ("comparisons", "recipes"):
        raise ValueError("comparison or recipe artifact kind required")
    root = os.environ.get("OPENBOOST_MULTI_SQUARED_ARTIFACTS") or os.environ.get(
        "OPENBOOST_NORMAL_ARTIFACTS"
    )
    path = Path(root or temporary) / ("multi-squared-" + kind)
    path.mkdir(parents=True, exist_ok=True)
    return path
