"""Check the baseline builder locally in an isolated NumPy/Hatchling environment.

Run with uv run --offline --isolated --no-project --with numpy==2.3.5 --with
hatchling python -m benchmarks.v1.check_validation_install OUTPUT.json.
This does not install or validate CUDA packages.
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check(output):
    if output.exists():
        raise ValueError("a new local evidence destination is required")
    sys.path.insert(0, str(ROOT / "src"))
    from benchmarks.v1.build_validation_baseline import build
    from benchmarks.v1.performance_evidence import digest

    protocol = json.loads((ROOT / "v1-sprints/105-validation-run11.json").read_text())
    protocol["packages"] = [
        f"{name}=={importlib.metadata.version(name)}" for name in ("numpy", "hatchling")
    ]
    with tempfile.TemporaryDirectory(prefix="openboost-105-baseline-install-") as directory:
        result = build(ROOT, Path(directory), protocol)
        code = """import json, pathlib, sys
sys.path.insert(0,sys.argv[1])
from benchmarks.v1.performance_evidence import verify_report
root=pathlib.Path(sys.argv[1]); folder=root/'v1-sprints/105-input-replay-local'
inputs=json.loads((folder/'inputs.json').read_text())
report=json.loads((folder/'result.json').read_text())
print(json.dumps(verify_report(report,inputs,inputs['sha256'],expected_repetitions=2)))
"""
        command = [result["python"], "-I", "-c", code, str(ROOT)]
        completed = subprocess.run(
            command, cwd=directory, text=True, capture_output=True, check=True, timeout=20
        )
        result.update(
            scope="Local baseline builder/install and saved CPU replay only. Two local packages checked; the eighteen Linux/CUDA package identities and real device execution remain unverified.",
            replay=json.loads(completed.stdout),
            replay_argv=command,
            invocation=sys.argv,
            revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
            support_sources={
                name: digest((ROOT / "benchmarks/v1" / name).read_bytes())
                for name in (
                    "check_validation_install.py",
                    "build_validation_baseline.py",
                    "performance_evidence.py",
                    "performance_checkpoint.py",
                )
            },
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        return dict(
            installed=result["passed"], source_files=len(result["sources"]), replay=result["replay"]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(check(parser.parse_args().output)))
