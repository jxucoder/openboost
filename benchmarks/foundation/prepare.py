"""Build an allowlisted upload bundle from a clean committed source tree."""

import argparse
import ast
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
    if suite in ("correctness", "boundaries", "baseline"):
        sources["test_correctness.py"] = ROOT / "tests/foundation/test_correctness.py"
    if suite in ("boundaries", "baseline"):
        sources["test_boundaries.py"] = ROOT / "tests/foundation/test_boundaries.py"
    if suite in ("histograms", "splits", "leaves", "builder", "trainer"):
        sources["test_histograms.py"] = ROOT / "tests/foundation/test_histograms.py"
        sources["histogram_oracle.py"] = ROOT / "tests/test_batch_histograms.py"
    if suite in ("splits", "leaves", "builder", "trainer"):
        sources["test_splits.py"] = ROOT / "tests/foundation/test_splits.py"
        sources["split_oracle.py"] = ROOT / "tests/test_batch_splits.py"
    if suite in ("leaves", "builder", "trainer"):
        sources["test_leaves.py"] = ROOT / "tests/foundation/test_leaves.py"
        sources["leaf_oracle.py"] = ROOT / "tests/test_batch_leaves.py"
    if suite in ("builder", "trainer"):
        sources["test_builder.py"] = ROOT / "tests/foundation/test_builder.py"
        sources["builder_oracle.py"] = ROOT / "tests/test_levelwise_builder.py"
    if suite == "trainer":
        sources["test_trainer.py"] = ROOT / "tests/foundation/test_trainer.py"
    extension_wheels, extension_sources = {}, {}
    if suite == "extensions":
        sources["test_extensions.py"] = ROOT / "tests/foundation/test_extensions.py"
        sources["check_extension_inference.py"] = ROOT / "tests/foundation/check_extension_inference.py"
        sources["extension_demo.py"] = ROOT / "examples/extensions/demo.py"
        for package in ("normal_fisher", "bounded_leaves"):
            project = ROOT / "examples/extensions" / package
            for source in sorted(project.rglob("*")):
                if source.is_file() and source.suffix in (".py", ".toml", ".md"):
                    extension_sources[str(source.relative_to(ROOT))] = sha256(source)
                    if source.suffix == ".py" and "src" in source.parts:
                        for node in ast.walk(ast.parse(source.read_text())):
                            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names] if isinstance(node, ast.Import) else []
                            if any(name.startswith("openboost") and name not in {"openboost", "openboost.experimental"} for name in names):
                                raise ValueError("Private OpenBoost import in extension package")
            for old in BUNDLE.glob(f"openboost_example_{package}-*.whl"):
                old.unlink()
            subprocess.run(["uv", "build", "--wheel", "--out-dir", str(BUNDLE), str(project)], cwd=ROOT, check=True)
            wheel = next(BUNDLE.glob(f"openboost_example_{package}-*.whl"))
            extension_wheels[wheel.name] = sha256(wheel)
    if suite == "baseline":
        from .dataset import describe

        archive = ROOT / "build/foundation_data/cal_housing.tgz"
        expected = json.loads((ROOT / "benchmarks/foundation/housing.json").read_text())
        if describe(archive) != expected:
            raise RuntimeError("Frozen dataset/split mismatch")
        sources.update({name: ROOT / "benchmarks/foundation" / name for name in ("dataset.py", "baseline_worker.py", "housing.json")})
        sources["test_baseline.py"] = ROOT / "tests/foundation/test_baseline.py"
        sources["cal_housing.tgz"] = archive
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
        "timeout_s": 1800 if suite == "baseline" else 300,
        "retries": 0,
        "dataset": {
            "generator": "numpy.default_rng",
            "seed": 31,
            "shape": [256, 4],
            "split": "smoke uses training data; no held-out quality claim",
        },
    }
    if suite == "extensions":
        manifest["extension_wheels"] = extension_wheels
        manifest["extension_sources"] = extension_sources
    if suite == "baseline":
        manifest["dataset"] = expected
        manifest["baseline_protocol"] = {"seeds": [0, 1, 2], "modes": ["resident", "eval"], "backends": ["cpu", "cuda"], "repeats_per_cell": 2, "nll_abs_tolerance": "0.01 * max(1, abs(cpu_nll))", "crps_max_ratio": 1.01, "coverage90_abs_tolerance": 0.01}
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared wheel {manifest['wheel_sha256']} from {manifest['source_sha']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "correctness", "boundaries", "baseline", "histograms", "splits", "leaves", "builder", "trainer", "extensions"), default="smoke")
    prepare(parser.parse_args().suite)
