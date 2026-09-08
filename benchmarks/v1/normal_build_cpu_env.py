"""Build an isolated CPU replay environment during the bounded Modal image build."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def build(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    wheels = destination / "wheels"
    python = destination / "venv/bin/python"
    report = {"commands": [], "passed": False, "builder_python": sys.version}

    def run(argv):
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        report["commands"].append(
            dict(argv=argv, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        )
        result.check_returncode()

    try:
        run(
            [
                "uv",
                "build",
                "--python",
                sys.executable,
                "--wheel",
                "--no-build-isolation",
                "--out-dir",
                str(wheels),
                str(root),
            ]
        )
        (wheel,) = wheels.glob("openboost-*.whl")
        run(["uv", "venv", "--python", sys.executable, str(destination / "venv")])
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "numpy==2.3.5",
                str(wheel),
            ]
        )
        report.update(passed=True, wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    finally:
        (destination / "build.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    build("/snapshot", "/opt/openboost-normal-cpu")
