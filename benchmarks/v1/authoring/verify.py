"""Retain an installed D1/D2 development smoke; no author/model invocation."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .export import ROOT, export


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema="openboost-d1-d2-installed-development-v1",
        passed=False,
        dispatch_ready=False,
        attempts=[],
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        os=platform.platform(),
        python=platform.python_version(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        device="cpu",
        threads=1,
        commands=[],
        claim="standalone development verification; no OS isolation or author benefit claim",
        sources={
            str(p.relative_to(ROOT)): digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(command, cwd, *, succeeds=True):
        record = dict(argv=command, cwd=str(cwd), timeout_s=120, exit_code=None)
        report["commands"].append(record)
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120
        )
        record.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if (result.returncode == 0) != succeeds:
            raise RuntimeError(f"unexpected exit {result.returncode}: {result.stderr}")
        return result.stdout

    try:
        bundle, observations = output / "evaluator", output / "observations"
        export(bundle)
        wheels = output / "wheels"
        for project in (
            ROOT,
            ROOT / "examples/v1_extensions/expectile",
            ROOT / "examples/v1_extensions/cohort_splits",
        ):
            run(["uv", "build", "--wheel", "--offline", "--out-dir", str(wheels)], project)
        paths = sorted(wheels.glob("*.whl"))
        report["wheels"] = {p.name: digest(p) for p in paths}
        with tempfile.TemporaryDirectory(prefix="openboost-d1-d2-development-") as temp:
            work = Path(temp)
            shutil.copyfile(Path(__file__).with_name("development.py"), work / "development.py")
            # Only inputs are delivered to the development collector's working directory.
            # Both environments are controlled by us; this is not a hostile-code sandbox.
            shutil.copyfile(bundle / "inputs.json", work / "inputs.json")
            interpreters = {}
            for role in ("collector", "judge"):
                run(["uv", "venv", "--python", sys.executable, str(work / role)], work)
                python = str(work / role / "bin/python")
                interpreters[role] = python
                selected = (
                    paths
                    if role == "collector"
                    else [p for p in paths if p.name.startswith("openboost-")]
                )
                run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--offline",
                        "--python",
                        python,
                        "numpy==2.3.5",
                        *map(str, selected),
                    ],
                    work,
                )
            for task in ("D1", "D2"):
                run(
                    [
                        interpreters["collector"],
                        "-I",
                        str(work / "development.py"),
                        str(work / "inputs.json"),
                        str(observations),
                        task,
                    ],
                    work,
                )
            # Retain every installed distribution file digest, including NumPy's
            # platform binary dependencies, rather than relying on version labels.
            report["judge_runtime"] = json.loads(
                run(
                    [
                        interpreters["judge"],
                        "-I",
                        "-c",
                        "import hashlib,importlib.metadata as m,importlib.util as u,json; "
                        "assert all(u.find_spec(n) is None for n in ['ob_expectile','ob_cohort_splits','tests']); "
                        "print(json.dumps({n:{'version':m.version(n),'files':{str(p):hashlib.sha256("
                        "m.distribution(n).locate_file(p).read_bytes()).hexdigest() for p in m.files(n) "
                        "if p.suffix != '.pyc'}} for n in ['numpy','openboost']}))",
                    ],
                    work,
                )
            )
            for task in ("D1", "D2"):
                command = [
                    interpreters["judge"],
                    "-I",
                    str(bundle / "judge.py"),
                    str(bundle),
                    str(observations),
                    task,
                ]
                result = run(command, work)
                (output / f"{task}-result.json").write_text(result)
            # Exercise a failing *standalone command*, not only an in-process assertion.
            negative = output / "negative-observations"
            shutil.copytree(observations, negative)
            model_path = negative / "d1-0.8.model.json"
            model = json.loads(model_path.read_text())
            model["base"][0] += 1
            model_path.write_text(json.dumps(model) + "\n")
            run(
                [
                    interpreters["judge"],
                    "-I",
                    str(bundle / "judge.py"),
                    str(bundle),
                    str(negative),
                    "D1",
                ],
                work,
                succeeds=False,
            )
            if "saved_model" not in report["commands"][-1]["stderr"]:
                raise RuntimeError("negative command failed for an unexpected reason")
            report["negative_model_rejected"] = True
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["artifacts"] = {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "manifest.json"
        }
        (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
