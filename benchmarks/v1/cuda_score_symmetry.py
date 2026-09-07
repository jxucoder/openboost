"""CPU arithmetic diagnostic for run 4's score-symmetry hypothesis, not a CUDA backend."""

import hashlib
import json
import platform
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from tests.v1.reference.device_rounds import fixture, rounds
from tests.v1.reference.device_splits import enumerate_candidates


def _fraction(value):
    return Fraction(float(value))


def _fused(a, b, c):
    # These small binary32 inputs/products have exact binary64 representations.
    # Fraction makes the one-rounding and two-rounding examples explicit.
    return np.float32(float(_fraction(a) * _fraction(b) + _fraction(c)))


def arithmetic_report():
    data = fixture("weighted")
    base, steps = rounds("weighted")
    candidates, _ = enumerate_candidates(data["x"], steps[0]["fields"], range(8), minimum=0)
    raw = np.full(8, np.float32(base), dtype=np.float32)
    gradient = raw + data["offset"].astype(np.float32) - data["target"].astype(np.float32)
    weighted = gradient * data["weight"].astype(np.float32)
    records = []
    for key in ((0, 0, True), (0, 3, False)):
        candidate = next(c for c in candidates if c["key"] == key)
        summaries = []
        for rows in candidate["rows"]:
            summaries.append(
                [
                    float(
                        np.float32(float(sum((_fraction(weighted[r]) for r in rows), Fraction(0))))
                    ),
                    float(sum(data["weight"][r] for r in rows)),
                ]
            )
        factors = [
            (np.float32(0.5) * np.float32(g), np.float32(np.float32(g) / np.float32(h + 1)))
            for g, h in summaries
        ]
        (a, b), (c, d) = factors
        variants = dict(
            separate_products=np.float32(np.float32(a * b) + np.float32(c * d)),
            left_product_fused=_fused(a, b, np.float32(c * d)),
            right_product_fused=_fused(c, d, np.float32(a * b)),
        )
        records.append(
            dict(
                key=list(key),
                oracle_gain=float(candidate["gain"]),
                rows=candidate["rows"],
                reconstructed_float32_summaries=summaries,
                child_score_values={k: float(v) for k, v in variants.items()},
                child_score_bits={k: hex(int(v.view(np.uint32))) for k, v in variants.items()},
            )
        )
    return dict(
        status="cpu_arithmetic_diagnostic_only",
        limitation="The GPU run did not export these root candidate gains or PTX. These reconstructed arithmetic examples show a compatible mechanism, not the observed GPU instruction sequence.",
        source_run="c4157559c5982e3df849ae59129c48dc8e60b454",
        base=base,
        float32_base=float(np.float32(base)),
        weighted_gradient=weighted.tolist(),
        candidates=records,
    )


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    report = arithmetic_report()
    report["provenance"] = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)),
        python=platform.python_version(),
        numpy=np.__version__,
        os=platform.platform(),
        source_hashes={
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest()
            for p in (
                "benchmarks/v1/cuda_score_symmetry.py",
                "tests/v1/reference/device_rounds.py",
                "tests/v1/reference/device_splits.py",
                "tests/v1/reference/device_histogram.py",
            )
        },
        command="UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.cuda_score_symmetry",
    )
    print(json.dumps(report, indent=2))
