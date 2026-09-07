"""Two frozen real A6 resource probes; no test data, no full search."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def profile_complete(execution, record):
    """A retained soft deadline is diagnostic completion, never a successful fit."""
    return (
        execution.get("status") == "error"
        and execution.get("exit_code") == 124
        and record.get("status") == "deadline"
        and record.get("soft_limit_s") == 60
        and bool(record.get("functions"))
    )


def comparator_jobs(plan, *, config_index=0):
    if config_index not in (0, 5):
        raise ValueError("only frozen comparator configurations 00 and 05 are preflighted")
    expected = [f"{library}:0:{config_index:02}" for library in ("xgboost", "lightgbm", "catboost")]
    jobs = [job for name in expected for job in plan["jobs"] if job["id"] == name]
    if [job["id"] for job in jobs] != expected:
        raise ValueError("three unique frozen comparator probes required")
    return jobs


def probe(spec, *, profile=False, package_root=None):
    import runpy

    import numpy as np

    source = Path("/snapshot/benchmarks/v1")
    comparator = spec["library"] != "openboost"
    worker = "baseline_worker.py" if comparator else "openboost_worker.py"
    predictor = "baseline_predict.py" if comparator else "openboost_predict.py"
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="a6-probe-", dir="/tmp"))
    root.chmod(0o755)
    job = {k: v for k, v in spec.items() if k not in ("id", "fold")}
    job["input_npz"] = "/input/worker-input.npz"
    path = root / "job.json"
    path.write_text(json.dumps(job))
    path.chmod(0o444)
    execute = runpy.run_path(str(source / "process_runner.py"))["execute"]
    if profile:
        fit = root / "profile"
        result = execute(
            [
                sys.executable,
                str(source / "profile_worker.py"),
                str(path),
                "--seconds",
                "60",
                "--no-stacks",
            ],
            fit,
            timeout_s=90,
            threads=1,
            address_limit_bytes=8 * 1024**3,
            unprivileged=True,
        )
        record = (
            json.loads((fit / "profile.json").read_text())
            if (fit / "profile.json").exists()
            else {}
        )
        root.chmod(0o700)
        return dict(
            id=spec["id"],
            passed=profile_complete(result, record),
            outcome_kind="instrumented_diagnostic_only",
            execution=result,
            profile_status=record.get("status"),
            artifacts={
                str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
            },
        )
    wrapper = """import json,resource,runpy,sys,importlib.metadata
from pathlib import Path
sys.argv=sys.argv[1:]
sys.path.insert(0,str(Path(sys.argv[0]).parent))
try:
    runpy.run_path(sys.argv[0],run_name='__main__')
finally:
    Path('resources.json').write_text(json.dumps(dict(packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, package_file=getattr(sys.modules.get('openboost'),'__file__',None))))
"""
    if package_root is not None:
        wrapper = f"import sys; sys.path.insert(0, {package_root!r})\n" + wrapper
    fit = root / "fit"
    result = execute(
        [sys.executable, "-c", wrapper, str(source / worker), str(path)],
        fit,
        timeout_s=1800,
        threads=1,
        address_limit_bytes=8 * 1024**3,
        unprivileged=True,
    )
    passed = result["status"] == "pass"
    replay_status = "not_run"
    replay_wall_s = None
    if passed:
        with np.load("/input/worker-input.npz", allow_pickle=False) as arrays:
            np.savez(
                root / "features.npz",
                x=arrays["x_validation"],
                row_ids=arrays["validation_row_ids"],
            )
        replay_started = time.monotonic()
        with (root / "replay.log").open("wb") as log:
            try:
                replay = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys,runpy; sys.path.insert(0, '/snapshot/benchmarks/v1'); "
                        + (f"sys.path.insert(0, {package_root!r}); " if package_root else "")
                        + "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
                        str(source / predictor),
                        str(fit / "model.bin"),
                        str(root / "features.npz"),
                        str(root / "replay.npz"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                    env=dict(
                        os.environ,
                        OMP_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                    ),
                )
            except subprocess.TimeoutExpired:
                replay = None
        replay_wall_s = time.monotonic() - replay_started
        replay_status = "timeout" if replay is None else "error"
        if replay is not None and replay.returncode == 0:
            with np.load(fit / "predictions.npz") as a, np.load(root / "replay.npz") as b:
                replay_status = (
                    "exact" if all(np.array_equal(a[k], b[k]) for k in a.files) else "mismatch"
                )
        passed = replay_status == "exact"
    root.chmod(0o700)
    return dict(
        id=spec["id"],
        passed=passed,
        execution=result,
        replay=replay_status,
        replay_wall_s=replay_wall_s,
        environment=dict(
            python=platform.python_version(), os=platform.platform(), numpy=np.__version__
        ),
        artifacts={
            str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
        },
    )


def pair_matches(results):
    if len(results) != 2 or not all(r["passed"] and r["replay"] == "exact" for r in results):
        return False
    return all(
        bool(results[0]["artifacts"].get(name))
        and results[0]["artifacts"].get(name) == results[1]["artifacts"].get(name)
        for name in ("fit/model.bin", "fit/predictions.npz", "fit/training.json", "replay.npz")
    )


def paired_probe(spec):
    results = []
    artifacts = {}
    for label, root in (("baseline", "/baseline"), ("current", None)):
        result = probe(spec, package_root=root)
        result["variant"] = label
        results.append(result)
        artifacts.update({label + "/" + k: v for k, v in result["artifacts"].items()})
        if not result["passed"]:
            break
    return dict(
        id=spec["id"],
        passed=pair_matches(results),
        outcome_kind="same_container_real_fit_pair",
        variants=[{k: v for k, v in result.items() if k != "artifacts"} for result in results],
        artifacts=artifacts,
    )


def main(output, packets, *, profile=False, paired=False, comparators=False, comparator_config=0):
    if sum((profile, paired, comparators)) > 1:
        raise ValueError("profile, paired and comparator modes are separate experiments")
    if comparator_config not in (0, 5) or (comparator_config != 0 and not comparators):
        raise ValueError("comparator configuration requires comparator mode and index 00 or 05")
    import modal

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    repo = source.parents[2]
    names = [
        "a6_resource_preflight",
        "profile_worker",
        "openboost_worker",
        "openboost_predict",
        "process_runner",
        "preprocessing",
        "ranking",
    ]
    if comparators:
        names += ["baseline_worker", "baseline_predict"]
    paths = (
        sorted((repo / "src/openboost").rglob("*.py"))
        + [source.with_name(name + ".py") for name in names]
        + [
            repo / name
            for name in (
                "benchmarks/__init__.py",
                "benchmarks/v1/__init__.py",
                "pyproject.toml",
                "README.md",
                "LICENSE",
            )
        ]
    )
    plan_path = repo / "v1-sprints/070-a6-resource-plan.json"
    plan = json.loads(plan_path.read_text())
    for name, expected in plan["input_files"].items():
        if hashlib.sha256((source.parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError("changed frozen resource plan input")
    from benchmarks.v1.a6_preflight_plan import compile_plan

    current = compile_plan(json.loads(source.with_name("search-design.json").read_text()))
    if {k: v for k, v in plan.items() if k != "input_files"} != current:
        raise ValueError("resource plan differs from compiler")
    if comparators:
        from benchmarks.v1.a6_search_plan import validate_plan

        plan_path = repo / "v1-sprints/070-a6-cpu-search-plan-bins.json"
        frozen = json.loads(plan_path.read_text())
        for name, expected in frozen["input_files"].items():
            if hashlib.sha256((source.parent / name).read_bytes()).hexdigest() != expected:
                raise ValueError("changed frozen comparator plan input")
        validate_plan(
            frozen["plan"], json.loads(source.with_name("search-design.json").read_text())
        )
        plan = {**frozen["plan"], "input_files": frozen["input_files"]}
    packet_manifest = json.loads((packets / "manifest.json").read_text())
    if packet_manifest["application"] != "A6" or packet_manifest["dataset"] != "parkinsons":
        raise ValueError("A6 Parkinsons packet required")
    if packet_manifest["source_freeze_sha256"] != plan["input_files"]["datasets/parkinsons.json"]:
        raise ValueError("source freeze mismatch")
    if (
        packet_manifest["preprocessing_freeze_sha256"]
        != plan["input_files"]["datasets/preprocessing.json"]
    ):
        raise ValueError("preprocessing freeze mismatch")
    fold = packet_manifest["folds"][0]
    if fold["seed"] != 0:
        raise ValueError("fold zero required")
    descriptor = fold["artifacts"]["worker-input"]
    packet = packets / descriptor["path"]
    if hashlib.sha256(packet.read_bytes()).hexdigest() != descriptor["sha256"]:
        raise ValueError("changed worker packet")
    import numpy as np

    with np.load(packet, allow_pickle=False) as arrays:
        if set(arrays.files) != {
            "x_train",
            "y_train",
            "x_validation",
            "y_validation",
            "validation_row_ids",
        }:
            raise ValueError("train/validation packet only")
    jobs = (
        comparator_jobs(plan, config_index=comparator_config)
        if comparators
        else [j for trial in plan["first_probe_ids"] for j in plan["jobs"] if j["id"] == trial]
    )
    if profile or paired:
        jobs = jobs[:1]
    baseline_revision = "17ab9de"
    baseline_files = []
    baseline_hashes = {}
    if paired:
        baseline_revision = subprocess.check_output(
            ["git", "rev-parse", baseline_revision], text=True
        ).strip()
        tracked = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", baseline_revision, "src/openboost"], text=True
        ).splitlines()
        for name in tracked:
            if not name.endswith(".py"):
                continue
            raw = subprocess.check_output(["git", "show", baseline_revision + ":" + name])
            relative = Path(name).relative_to("src")
            path = output / "baseline-source" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            baseline_files.append((path, str(relative)))
            baseline_hashes[name] = hashlib.sha256(raw).hexdigest()
    manifest = dict(
        baseline_revision=baseline_revision if paired else None,
        baseline_sources=baseline_hashes,
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        sources={
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        argv=sys.argv,
        requested=dict(
            cpu=[2, 2],
            memory_mib=[8192, 8192],
            function_seconds=120 if profile else 3800 if paired else 1900,
            retries=0,
        ),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        packet=descriptor,
        fold_metadata=fold["metadata"],
        packet_manifest=packet_manifest,
        jobs=jobs,
        mode="comparators"
        if comparators
        else "profile"
        if profile
        else "paired"
        if paired
        else "resource",
        status="running",
        modal_version=modal.__version__,
    )

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    app = modal.App("openboost-v1-a6-resource-preflight")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .uv_pip_install("numpy==2.3.5", "pytest==9.0.2", "hatchling==1.27.0", uv_version="0.12.1")
        .env({"PYTHONDONTWRITEBYTECODE": "1"})
    )
    if comparators:
        image = image.apt_install("libgomp1").uv_pip_install(
            "xgboost==3.4.1",
            "lightgbm==4.7.0",
            "catboost==1.2.10",
            "scipy==1.16.3",
            "scikit-learn==1.8.0",
            uv_version="0.12.1",
        )
    for path in paths:
        image = image.add_local_file(path, "/snapshot/" + str(path.relative_to(repo)), copy=True)
    image = image.uv_pip_install(
        "/snapshot", extra_options="--no-deps --no-build-isolation", uv_version="0.12.1"
    )

    image = image.add_local_file(packet, "/input/worker-input.npz", copy=True)

    for path, relative in baseline_files:
        image = image.add_local_file(path, "/baseline/" + relative, copy=True)

    @app.function(
        image=image,
        cpu=(2, 2),
        memory=(8192, 8192),
        timeout=120 if profile else 3800 if paired else 1900,
        retries=0,
        max_containers=1,
        serialized=True,
        include_source=False,
    )
    def remote(job):
        import runpy

        functions = runpy.run_path("/snapshot/benchmarks/v1/a6_resource_preflight.py")
        return (
            functions["paired_probe"](job) if paired else functions["probe"](job, profile=profile)
        )

    try:
        results = []
        with modal.enable_output(), app.run():
            for job in jobs:
                result = remote.remote(job)
                results.append({k: v for k, v in result.items() if k != "artifacts"})
                for name, data in result.pop("artifacts").items():
                    path = output / job["id"].replace(":", "-") / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                manifest.update(results=results, image_id=image.object_id)
                save()
                if not result["passed"]:
                    break
        manifest.update(
            status="complete",
            passed=len(results) == len(jobs) and all(r["passed"] for r in results),
            artifacts={
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
    except Exception as error:
        manifest.update(status="error", error=str(error))
        raise
    finally:
        save()
    print(json.dumps(dict(passed=manifest["passed"], results=manifest["results"]), indent=2))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("packets", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--comparators", action="store_true")
    parser.add_argument("--comparator-config", type=int, choices=(0, 5), default=0)
    args = parser.parse_args()
    main(
        args.output,
        args.packets,
        profile=args.profile,
        paired=args.paired,
        comparators=args.comparators,
        comparator_config=args.comparator_config,
    )
