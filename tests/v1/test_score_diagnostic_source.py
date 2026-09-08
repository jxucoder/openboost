"""Integrity of the historical scoring control, without importing CUDA."""

import ast
import hashlib
from pathlib import Path


def test_archived_kernel_is_the_unchanged_run4_function():
    source = Path(__file__).with_name("run4_score_kernel.py").read_text()
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    segment = ast.get_source_segment(source, function)
    assert function.name == "scalar_scores"
    assert hashlib.sha256(segment.encode()).hexdigest() == (
        "d6fe21e575c54c16ffc590a57c95edb34dd3c0a95c705de18c0fcf8eaf7154a8"
    )
