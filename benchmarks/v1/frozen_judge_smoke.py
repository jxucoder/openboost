"""Synthetic evaluator-freeze fault smoke; never application quality evidence."""

import argparse
import hashlib
import json
import platform
import subprocess
from copy import deepcopy
from pathlib import Path

from benchmarks.v1.judge import SCHEMA, cache_key, judge


def run(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema=SCHEMA,
        protocol_sha256="a" * 64,
        provenance=dict(code_sha="b" * 40, dirty=False, environment=dict(kind="synthetic fixture")),
        expected=[],
    )
    for application, fold in [(f"A{i}", 0) for i in range(1, 14)] + [("A1", 1)]:
        manifest["expected"].append(
            dict(
                id=f"{application}/cpu/{fold}",
                application=application,
                required=True,
                backend="cpu",
                dataset_sha256="c" * 64,
                split_sha256="d" * 64,
                preprocessing_sha256="e" * 64,
                config=dict(depth=2),
                seed=0,
                model="fixture",
                fold=str(fold),
            )
        )
    artifacts = {}
    for role, payload in dict(
        predictions=b"[1.0, 2.0]", model=b"not a trained model", log=b"synthetic smoke"
    ).items():
        path = root / (role + ".json")
        path.write_bytes(payload)
        artifacts[role] = dict(path=path.name, sha256=hashlib.sha256(payload).hexdigest())
    records = [
        dict(
            id=c["id"],
            status="pass",
            cache_key=cache_key(manifest, c),
            backend="cpu",
            fallback=False,
            exit_code=0,
            artifacts=artifacts,
            metrics=dict(loss=0.2),
            reason="",
        )
        for c in manifest["expected"]
    ]
    cases = []
    for attack in (
        "valid",
        "rehashed_fold_omission",
        "rehashed_code",
        "missing_record",
        "duplicate_record",
        "wrong_backend",
        "worker_failure",
        "timeout",
        "nonfinite_metric",
    ):
        producer, outputs = deepcopy(manifest), deepcopy(records)
        if attack == "rehashed_fold_omission":
            producer["expected"].pop()
            outputs.pop()
        elif attack == "rehashed_code":
            producer["provenance"]["code_sha"] = "f" * 40
        elif attack == "missing_record":
            outputs.pop()
        elif attack == "duplicate_record":
            outputs.append(deepcopy(outputs[0]))
        elif attack == "wrong_backend":
            outputs[0]["backend"] = "cuda"
        elif attack == "worker_failure":
            outputs[0]["exit_code"] = 2
        elif attack == "timeout":
            outputs[0].update(status="timeout", exit_code=-9, reason="injected timeout status")
        elif attack == "nonfinite_metric":
            outputs[0]["metrics"]["loss"] = float("nan")
        for record in outputs:
            cell = next(c for c in producer["expected"] if c["id"] == record["id"])
            record["cache_key"] = cache_key(producer, cell)
        declared = judge(producer, outputs, root)
        frozen = judge(producer, outputs, root, frozen_manifest=manifest)
        assert frozen["integrity_pass"] is (attack == "valid")
        assert frozen["gate_results"] == {}
        cases.append(dict(attack=attack, declared=declared, frozen=frozen))
    repo = Path(__file__).resolve().parents[2]
    result = dict(
        scope="Synthetic fault injection into integrity judge; no real worker/resource/access isolation or quality claim",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        environment=dict(python=platform.python_version(), os=platform.platform()),
        sources={
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("judge.py"))
        },
        frozen_manifest=manifest,
        cases=cases,
        passed=True,
    )
    (root / "smoke.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    run(args.directory)
