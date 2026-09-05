"""Build an allowlisted upload bundle from a clean committed source tree."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "build" / "foundation"
IMAGE = "nvidia/cuda@sha256:14c54fad24b376ab78a70e1ef6595a2b7c8cdbf187e4f9b76de99a926fb62460"
COMMAND = ["uv", "run", "modal", "run", "benchmarks/foundation/modal_app.py::foundation_smoke"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(suite="smoke"):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if git("status", "--porcelain"):
        raise RuntimeError("Commit the implementation before preparing GPU evidence")
    BUNDLE.mkdir(parents=True, exist_ok=True)
    # Only delete generated wheels in this dedicated build directory.
    for old in BUNDLE.glob("openboost-*.whl"):
        old.unlink()
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(BUNDLE)], cwd=ROOT, check=True)
    wheels = list(BUNDLE.glob("openboost-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one freshly built wheel")
    sources = {
        "test_smoke.py": ROOT / "tests/foundation/test_smoke.py",
        "conftest.py": ROOT / "tests/foundation/conftest.py",
        "pytest.ini": ROOT / "tests/foundation/pytest.ini",
        "requirements.txt": ROOT / "benchmarks/foundation/requirements.txt",
    }
    if suite in ("correctness", "boundaries"):
        sources["test_correctness.py"] = ROOT / "tests/foundation/test_correctness.py"
    if suite == "boundaries":
        sources["test_boundaries.py"] = ROOT / "tests/foundation/test_boundaries.py"
    for name, source in sources.items():
        shutil.copyfile(source, BUNDLE / name)
    manifest = {
        "schema_version": 1,
        "suite": suite,
        "test_files": [name for name in sources if name.startswith("test_")],
        "source_sha": git("rev-parse", "HEAD"),
        "source_dirty": False,
        "wheel": wheels[0].name,
        "wheel_sha256": sha256(wheels[0]),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "files": {name: sha256(BUNDLE / name) for name in sources},
        "base_image": IMAGE,
        "python": "3.12",
        "uv_version": "0.12.1",
        "command": ["uv", "run", "--no-sync", "modal", "run", f"benchmarks/foundation/modal_app.py::foundation_{suite}"],
        "gpu": "T4",
        "timeout_s": 300,
        "retries": 0,
        "dataset": {
            "generator": "numpy.default_rng",
            "seed": 31,
            "shape": [256, 4],
            "split": "smoke uses training data; no held-out quality claim",
        },
    }
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared wheel {manifest['wheel_sha256']} from {manifest['source_sha']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "correctness", "boundaries"), default="smoke")
    prepare(parser.parse_args().suite)
