"""Local NumPy/core-only replay of the 105 evidence format, with generation disabled."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmarks.v1.performance_checkpoint import CONFIG, workload
from benchmarks.v1.performance_evidence import digest, input_record, write_json

CHILD = """import importlib.util, json, pathlib, sys, openboost
from benchmarks.v1.performance_evidence import run, verify_report, write_json
import benchmarks.v1.performance_checkpoint as old
def forbidden(*args, **kwargs): raise AssertionError("generator called")
old.workload = forbidden
assert "site-packages" in openboost.__file__
assert importlib.util.find_spec("cupy") is None and importlib.util.find_spec("numba") is None
root = pathlib.Path(sys.argv[1])
inputs = json.loads((root / "inputs.json").read_text())
report = run(inputs, "cpu", root / "result.json", repetitions=2)
verified = verify_report(report, inputs, inputs["sha256"], expected_repetitions=2)
write_json(root / "verification.json", dict(**verified, installed_core=True,
    generator_disabled=True, cupy_available=False, numba_available=False,
    input_sha256=inputs["sha256"]))
print(json.dumps(verified))
"""


def main(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    commands = []
    metadata = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)),
        sources={
            str(p.relative_to(repo)): digest(p.read_bytes())
            for p in (
                *sorted((repo / "src/openboost").glob("*.py")),
                *(
                    repo / "benchmarks/v1" / name
                    for name in (
                        "performance_checkpoint.py",
                        "performance_evidence.py",
                        "check_performance_install.py",
                    )
                ),
            )
        },
    )
    with tempfile.TemporaryDirectory(prefix="openboost-105-replay-") as temporary:
        work = Path(temporary)

        def command(argv):
            p = subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=120)
            commands.append(
                dict(argv=argv, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
            )
            write_json(
                output / "installation.json",
                dict(
                    **metadata,
                    scope="Local isolated CPU evidence-format verification; not a performance or CUDA result.",
                    commands=commands,
                ),
            )
            if p.returncode:
                raise RuntimeError(p.stderr)

        command(
            ["uv", "build", "--offline", "--wheel", "--out-dir", str(work / "wheels"), str(repo)]
        )
        command(["uv", "venv", "--python", sys.executable, str(work / "venv")])
        python, wheel = work / "venv/bin/python", next((work / "wheels").glob("*.whl"))
        command(
            [
                "uv",
                "pip",
                "install",
                "--offline",
                "--no-deps",
                "--python",
                str(python),
                "numpy==2.3.5",
                str(wheel),
            ]
        )
        for name in ("performance_checkpoint.py", "performance_evidence.py"):
            destination = work / "benchmarks/v1" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / "benchmarks/v1" / name, destination)
        previous = dict(CONFIG)
        try:
            CONFIG.update(rounds=2, bins=4, max_depth=1)
            write_json(output / "inputs.json", input_record(*workload(33, "normal"), "normal"))
        finally:
            CONFIG.clear()
            CONFIG.update(previous)
        command([str(python), "-c", CHILD, str(output)])
    print((output / "verification.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
