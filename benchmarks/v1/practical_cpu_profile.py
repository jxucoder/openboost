"""Freeze and execute the eight-case Sprint 066 CPU diagnostic, without test labels."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.v1 import housing

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(archive, directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    x, y, member = housing.load_archive(archive)
    frozen = json.loads((ROOT / "benchmarks/v1/datasets/housing.json").read_text())
    if housing.digest(x.tobytes() + y.tobytes()) != frozen["arrays_sha256"]:
        raise ValueError("Housing arrays differ from frozen source")
    train, valid, _ = housing.split_indices(len(y), 0)
    if [housing.digest(ids.tobytes()) for ids in (train, valid)] != frozen["split_sha256"]["0"][:2]:
        raise ValueError("Housing fold-zero row order differs")
    if len(train) < 8192 or len(valid) < 1024:
        raise ValueError("frozen prefixes unavailable")
    train, valid = train[:8192], valid[:1024]
    np.savez(
        root / "input.npz",
        x_train=x[train],
        y_train=y[train],
        train_ids=train,
        x_validation=x[valid],
        y_validation=y[valid],
        validation_ids=valid,
    )
    cases = [
        dict(id=f"{recipe}-{rows}-{rounds}", recipe=recipe, train_rows=rows, rounds=rounds)
        for rows, rounds in ((8192, 4), (2048, 32), (8192, 32), (8192, 128))
        for recipe in ("squared", "normal")
    ]
    record = dict(
        schema="practical-cpu-066",
        input_sha256=digest(root / "input.npz"),
        features=housing.FEATURES,
        archive_sha256=digest(archive),
        member_sha256=member,
        source_freeze_sha256=digest(ROOT / "benchmarks/v1/datasets/housing.json"),
        training_prefix_sha256=housing.digest(train.tobytes()),
        small_training_prefix_sha256=housing.digest(train[:2048].tobytes()),
        validation_prefix_sha256=housing.digest(valid.tobytes()),
        cases=cases,
        seed=0,
        round_options=dict(
            bins=32, max_depth=2, learning_rate=0.1, reg_lambda=1.0, patience=None, step="fixed"
        ),
        budgets=dict(
            worker_seconds=120,
            profile_soft_seconds=60,
            worker_address_bytes=8 * 1024**3,
            numerical_threads=2,
            container_memory_mib=[8192, 8192],
            function_seconds=1500,
            retries=0,
            max_uninstrumented_fits=8,
            max_instrumented_fits=2,
        ),
        profile_policy="after first failed fit, profile only that case; if all pass, profile the 8192/128 case for each recipe",
        stop_policy="stop remaining uninstrumented fits at first non-pass; retain pending cases as not_run",
        scope="eight original numeric Housing features, raw target units, training/validation only; no exporter imputation/indicator columns; diagnostic not E3/E4",
    )
    (root / "protocol.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def execute(directory, profile_only=None, baseline_wheel=None):
    import modal

    root = Path(directory).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if digest(root / "input.npz") != protocol["input_sha256"]:
        raise ValueError("frozen input changed")
    if profile_only is not None and profile_only not in {c["id"] for c in protocol["cases"]}:
        raise ValueError("profile case is absent from frozen protocol")
    baseline = None
    if baseline_wheel is not None:
        if profile_only is not None:
            raise ValueError("paired comparison and profile-only are separate modes")
        baseline_wheel = Path(baseline_wheel).resolve()
        baseline = json.loads(
            (ROOT / "benchmarks/v1/evidence/practical-cpu-066/sweep/manifest.json").read_text()
        )
        if digest(baseline_wheel) != baseline["wheel_sha256"]:
            raise ValueError("baseline wheel differs from retained Sprint 066 sweep")
        if protocol != baseline["protocol"]:
            raise ValueError("paired comparison must use unchanged baseline protocol")
    output = root / "run"
    output.mkdir(exist_ok=False)
    wheels = output / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(wheels)], cwd=ROOT, check=True
    )
    (wheel,) = wheels.glob("*.whl")
    source_paths = [
        Path(__file__),
        ROOT / "benchmarks/v1/cpu_profile_worker.py",
        ROOT / "benchmarks/v1/resource_preflight.py",
        ROOT / "benchmarks/v1/profile_worker.py",
        ROOT / "benchmarks/v1/housing.py",
        *sorted((ROOT / "src/openboost").glob("*.py")),
    ]
    manifest = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        argv=sys.argv,
        sources={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        protocol_sha256=digest(root / "protocol.json"),
        wheel_sha256=digest(wheel),
        modal_version=modal.__version__,
        protocol=protocol,
        profile_only=profile_only,
        paired_baseline=baseline,
        comparison_policy=(
            dict(
                max_fits=12,
                selected_cases=protocol["cases"][:6],
                order="baseline then candidate for each case",
                profiles=0,
                scope="separate same-container amendment; original per-worker caps",
            )
            if baseline
            else None
        ),
        status="running",
        cases=[],
        profiles=[],
    )
    path = output / "manifest.json"

    def save():
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    save()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install("numpy==2.3.5", "threadpoolctl==3.6.0", uv_version="0.12.1")
        .add_local_file(wheel, "/opt/" + wheel.name, copy=True)
        .uv_pip_install("/opt/" + wheel.name, uv_version="0.12.1")
    )
    for name in ("cpu_profile_worker.py", "resource_preflight.py", "profile_worker.py"):
        image = image.add_local_file(ROOT / "benchmarks/v1" / name, "/opt/" + name, copy=True)
    image = image.add_local_file(root / "input.npz", "/opt/input.npz", copy=True)
    if baseline_wheel is not None:
        image = image.add_local_file(baseline_wheel, "/opt/baseline.whl", copy=True)
    app = modal.App("openboost-v1-practical-cpu-profile")

    @app.function(
        image=image,
        cpu=(2, 2),
        memory=(8192, 8192),
        timeout=1500,
        max_containers=1,
        serialized=True,
        include_source=False,
        is_generator=True,
    )
    def remote(protocol, profile_only, paired):
        import hashlib
        import importlib.metadata
        import json
        import platform
        import runpy
        import sys
        import tempfile
        from pathlib import Path

        resources = runpy.run_path("/opt/resource_preflight.py")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            preflight = resources["worker_probe"](work / "preflight")
            yield dict(
                kind="preflight",
                result=preflight,
                environment=dict(
                    python=platform.python_version(),
                    os=platform.platform(),
                    packages={
                        n: importlib.metadata.version(n)
                        for n in ("numpy", "threadpoolctl", "openboost")
                    },
                ),
            )
            if not preflight["passed"]:
                return

            if paired:
                import zipfile

                with zipfile.ZipFile("/opt/baseline.whl") as archive:
                    archive.extractall(work / "baseline")

            def run_case(case, instrumented=False, fixtures=False, variant=None):
                name = "fixtures" if fixtures else case["id"] + ("-profile" if instrumented else "")
                if variant is not None:
                    name = variant + "-" + name
                out = work / name
                command = [sys.executable, "/opt/cpu_profile_worker.py"]
                if fixtures:
                    command.append("--fixtures")
                else:
                    job = dict(
                        case,
                        features=protocol["features"],
                        input_path="/opt/input.npz",
                        input_sha256=protocol["input_sha256"],
                        instrumented=instrumented,
                    )
                    job_path = work / (name + ".json")
                    job_path.write_text(json.dumps(job))
                    command.append(str(job_path))
                if variant == "baseline":
                    bootstrap = "import sys,runpy; sys.path.insert(0,'/opt'); sys.path.insert(0,sys.argv.pop(1)); sys.argv.pop(0); runpy.run_path(sys.argv[0],run_name='__main__')"
                    command = [
                        sys.executable,
                        "-c",
                        bootstrap,
                        str(work / "baseline"),
                        *command[1:],
                    ]
                execution = resources["bounded"](
                    command,
                    seconds=30 if fixtures else 120,
                    address_bytes=protocol["budgets"]["worker_address_bytes"],
                    directory=out,
                )
                artifacts = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
                if (
                    execution["status"] == "pass"
                    and ("fixtures.json" if fixtures else "result.json") not in artifacts
                ):
                    execution["status"] = "error"
                    execution["reason"] = "missing required result"
                return dict(
                    kind="fixtures" if fixtures else "profile" if instrumented else "case",
                    id=name,
                    case=case,
                    variant=variant,
                    execution=execution,
                    artifacts=artifacts,
                    hashes={n: hashlib.sha256(b).hexdigest() for n, b in artifacts.items()},
                )

            fixture = run_case({}, fixtures=True)
            yield fixture
            if fixture["execution"]["status"] != "pass":
                return
            if paired:
                failed = False
                # Only the six completed baseline cases, in original order.
                for case in protocol["cases"][:6]:
                    for variant in ("baseline", "candidate"):
                        if failed:
                            yield dict(
                                kind="case",
                                id=variant + "-" + case["id"],
                                case=case,
                                variant=variant,
                                artifacts={},
                                hashes={},
                                execution=dict(status="not_run", reason="prior paired fit failed"),
                            )
                            continue
                        result = run_case(case, variant=variant)
                        yield result
                        failed = result["execution"]["status"] != "pass"
                return
            if profile_only is not None:
                case = next(c for c in protocol["cases"] if c["id"] == profile_only)
                yield run_case(case, instrumented=True)
                return
            failed = None
            for case in protocol["cases"]:
                if failed is not None:
                    yield dict(
                        kind="case",
                        id=case["id"],
                        case=case,
                        execution=dict(status="not_run", reason="prior fit failed"),
                        artifacts={},
                        hashes={},
                    )
                    continue
                result = run_case(case)
                yield result
                if result["execution"]["status"] != "pass":
                    failed = case
            profiles = [failed] if failed else protocol["cases"][-2:]
            for case in profiles:
                yield run_case(case, instrumented=True)

    try:
        with modal.enable_output(), app.run():
            for item in remote.remote_gen(protocol, profile_only, baseline is not None):
                if item["kind"] == "preflight":
                    if "preflight" in manifest:
                        raise RuntimeError(
                            "restarted generator detected; stopping repeated execution"
                        )
                    manifest["preflight"] = item
                    print("preflight", item["result"]["passed"], flush=True)
                else:
                    folder = output / item["id"]
                    folder.mkdir()
                    for name, content in item.pop("artifacts").items():
                        if Path(name).name != name:
                            raise ValueError("invalid artifact basename")
                        (folder / name).write_bytes(content)
                    if item["kind"] == "fixtures":
                        manifest["fixtures"] = item
                    else:
                        manifest["profiles" if item["kind"] == "profile" else "cases"].append(item)
                    print(item["id"], item["execution"]["status"], flush=True)
                manifest["image_id"] = image.object_id
                save()
        complete = (
            len(manifest["profiles"]) == 1
            if profile_only
            else len(manifest["cases"]) == (12 if baseline else 8)
        )
        manifest["status"] = "complete" if complete else "incomplete"
    except Exception as exc:
        manifest.update(status="error", error=str(exc))
        raise
    finally:
        save()
    if manifest["status"] != "complete" or any(
        c["execution"]["status"] != "pass" for c in manifest["cases"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--paired-baseline-wheel",
        type=Path,
        help="Pair six completed baseline cases in one container",
    )
    parser.add_argument("--profile-only", help="Profile one frozen case after an interrupted sweep")
    parser.add_argument(
        "--archive", type=Path, default=Path("build/foundation_data/cal_housing.tgz")
    )
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.archive, args.directory)
    else:
        execute(args.directory, args.profile_only, args.paired_baseline_wheel)
