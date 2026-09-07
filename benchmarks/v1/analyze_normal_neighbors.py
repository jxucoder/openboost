"""CPU-only original-row study of adjacent states around a recorded device base."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from tests.v1.reference.device_normal import fixture, geometry
from tests.v1.reference.normal_acceptance import compare_precisions

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "benchmarks/v1/evidence/cuda-normal-090"
MODEL = "normal/d2-ordinary-0-forward-backtracking/model.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    assert digest(ARCHIVE / MODEL) == manifest["artifacts"][MODEL]
    f, d2 = fixture("conflict"), fixture("d2")
    assert all(np.array_equal(f[k], d2[k]) for k in ("x", "target", "offset", "weight"))
    base = np.array(json.loads((ARCHIVE / MODEL).read_text())["base"], np.float32)
    before = np.broadcast_to(base, f["offset"].shape)
    loss = geometry(before, f["target"], f["offset"], f["weight"])[0]
    rows = []
    for channel in range(2):
        for toward in (-np.inf, np.inf):
            candidate = base.copy()
            candidate[channel] = np.nextafter(base[channel], np.float32(toward))
            after = np.broadcast_to(candidate, before.shape)
            value = geometry(after, f["target"], f["offset"], f["weight"])[0]
            rows.append(
                dict(
                    channel=channel,
                    direction="down" if toward < 0 else "up",
                    raw=candidate.tolist(),
                    raw_float32_bits=candidate.view(np.uint32).tolist(),
                    float64_nll=value,
                    float64_nll_bits=int(np.float64(value).view(np.uint64)),
                    float64_nll_difference=value - loss,
                    high_precision=compare_precisions(
                        before, after, f["target"], f["offset"], f["weight"]
                    ),
                )
            )
    paths = [
        Path(__file__),
        ROOT / "tests/v1/reference/normal_acceptance.py",
        *(
            ROOT / "tests/v1/reference" / name
            for name in (
                "device_normal.py",
                "device_rounds.py",
                "device_splits.py",
                "device_histogram.py",
            )
        ),
    ]
    return dict(
        schema="openboost-normal-neighbor-study-v1",
        device_execution=False,
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        argv=sys.argv,
        environment=dict(
            python=platform.python_version(), numpy=np.__version__, os=platform.platform()
        ),
        analysis_sources={str(p.relative_to(ROOT)): digest(p) for p in paths},
        input_manifest_sha256=digest(ARCHIVE / "manifest.json"),
        input_model=dict(
            path=MODEL, sha256=digest(ARCHIVE / MODEL), dispatch_revision=manifest["revision"]
        ),
        fixture="conflict; training arrays equal d2",
        original_inputs={k: f[k].tolist() for k in ("target", "offset", "weight")},
        base=base.tolist(),
        base_float32_bits=base.view(np.uint32).tolist(),
        float64_nll=loss,
        float64_nll_bits=int(np.float64(loss).view(np.uint64)),
        neighbors=rows,
        limitation="A passing D2 base, not a failed GPU state trace. CPU high-precision mathematics, not CUDA execution/emulation or a repaired acceptance policy.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.parse_args().output.write_text(json.dumps(analyze(), indent=2, allow_nan=False) + "\n")
