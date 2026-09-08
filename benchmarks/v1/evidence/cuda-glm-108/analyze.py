"""Offline run-12 provenance, stored-input comparison and saved-model audit."""

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))

from benchmarks.v1.cuda_aggregation_preflight import judge_run  # noqa: E402
from tests.v1.reference.glm_comparison import direct_difference  # noqa: E402
from tests.v1.reference.glm_recipe import fit  # noqa: E402
from tests.v1.test_device_glm_reference import original_rows  # noqa: E402

from openboost import ClassSchema, LossChange, NumericData, Problem  # noqa: E402
from openboost.artifacts import Model  # noqa: E402
from openboost.stopping import StopState  # noqa: E402


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def comparison(family, arrays, record, retained_exact):
    exact = direct_difference(family, *arrays)
    check(exact == Decimal(retained_exact), "retained direct likelihood differs")
    change = LossChange(**record)
    return dict(
        status=change.status,
        encloses=change.lower is not None
        and Decimal(change.lower) <= exact <= Decimal(change.upper),
        identity_matches=change.unchanged == np.array_equal(arrays[0], arrays[1]),
        method_matches=change.method == family + "-convex-taylor18-interval-v1",
    )


def decode_inputs(record, family):
    values = {}
    for name, item in record.items():
        raw = base64.b64decode(item["data_base64"], validate=True)
        value = np.frombuffer(raw, dtype=item["dtype"]).reshape(item["shape"])
        check(value.tobytes() == raw, "retained array bytes differ")
        values[name] = value
    expected = {"features", "row_ids", "target", "offset", "weight"}
    if family == "poisson":
        expected.add("exposure")
    check(values.keys() == expected, "input snapshot fields differ")
    data = NumericData(values["features"], values["row_ids"], ("x",))
    return Problem(
        data,
        values["target"],
        data.row_ids,
        offset=values["offset"],
        weight=values["weight"],
        classes=ClassSchema(("no", "yes")) if family == "binary" else None,
        structure={"exposure": values["exposure"]} if family == "poisson" else None,
    )


def recipe(report):
    family, depth, step, rate = report["case"].split("/")
    train, validation = (
        decode_inputs(report["inputs"][name], family) for name in ("train", "validation")
    )
    expected = fit(
        family,
        original_rows(train),
        original_rows(validation),
        depth=int(depth),
        step=step,
        rate=float(rate),
    )
    models = {name: Model.from_record(report[name]) for name in ("model", "best")}
    for name, model in models.items():
        check(model.record() == report[name], "model round trip differs")
        check(model.classes == train.classes, "persisted classes differ")
        check(
            len(model.terms) == expected["terms" if name == "model" else "best_terms"],
            "model prefix differs",
        )
        np.testing.assert_allclose(
            model.predict(validation.data)[:, 0],
            expected["val" if name == "model" else "best"],
            rtol=1e-4,
            atol=1e-5,
        )
    np.testing.assert_allclose(
        models["model"].predict(train.data)[:, 0], expected["raw"], rtol=1e-4, atol=1e-5
    )
    check(len(report["steps"]) == len(expected["history"]), "trajectory length differs")
    for actual, reference in zip(report["steps"], expected["history"], strict=True):
        observed = [
            (t["coefficient"], t["accepted"], t["failure"] is not None) for t in actual["trials"]
        ]
        required = [(t["coefficient"], t["accepted"], t["failure"]) for t in reference["trials"]]
        check(observed == required, "trial decisions differ")
        check(actual["after_version"] == reference["terms"], "trajectory prefix differs")
        np.testing.assert_allclose(
            [actual["loss"], actual["validation_score"], actual["best_score"]],
            [reference["loss"], reference["score"], reference["best_score"]],
            rtol=1e-3,
            atol=1e-3,
        )
    stop = StopState(**report["stop"])
    check(
        stop.reason == expected["reason"] and stop.stale_rounds == expected["stale"],
        "stop state differs",
    )
    audits = [
        comparison(family, row["inputs"], row["result"], row["decimal220"])
        for row in report["comparisons"]
    ]
    check(
        all(
            row["encloses"] and row["identity_matches"] and row["method_matches"] for row in audits
        ),
        "recipe comparison audit failed",
    )
    return dict(
        case=report["case"],
        comparisons=len(audits),
        models_replayed=2,
        terms=len(models["model"].terms),
        best_terms=len(models["best"].terms),
        rounds=stop.completed_rounds,
        stop=stop.reason,
    )


def analyze():
    manifest, verdict = read(ROOT / "manifest.json"), read(ROOT / "verdict.json")
    protocol, result = manifest["protocol"], manifest["result"]
    check(manifest["dirty"] is False, "dirty dispatch")
    check(
        protocol["authorization"] == protocol["upload_authorization"] == "approved",
        "dispatch not approved",
    )
    check(len(manifest["sources"]) == protocol["upload_file_count"] == 85, "source closure differs")
    protocol_name = "v1-sprints/108-glm-run12.json"
    for name, expected in manifest["sources"].items():
        raw = subprocess.check_output(["git", "show", manifest["revision"] + ":" + name], cwd=REPO)
        check(digest(raw) == expected, "dispatch source differs: " + name)
        if name == protocol_name:
            check(json.loads(raw) == protocol, "embedded protocol differs")
        elif name.endswith(".py"):
            # Use executed core and oracle code for replay, even after future construction.
            check(
                digest((REPO / name).read_bytes()) == expected,
                "local audit source differs: " + name,
            )
    check(
        protocol["frozen_sources"]
        == {p: h for p, h in manifest["sources"].items() if p != protocol_name},
        "prefrozen sources differ",
    )
    retained = result.get("retained_artifact_hashes", {})
    check(set(retained) <= set(protocol["retained_artifacts"]), "undeclared artifact")
    check(
        set(manifest["artifacts"]) == set(retained) | {"pytest.log", "junit.xml", "verdict.json"},
        "artifact index differs",
    )
    for name, expected in manifest["artifacts"].items():
        check(digest((ROOT / name).read_bytes()) == expected, "raw artifact changed: " + name)
    check(
        retained == {name: manifest["artifacts"][name] for name in retained},
        "artifact hashes disagree",
    )
    total_bytes = sum((ROOT / name).stat().st_size for name in retained)
    check(total_bytes <= protocol["retained_artifact_bytes"], "artifact budget exceeded")
    check(
        judge_run(result, (ROOT / "junit.xml").read_text(), protocol, manifest["sources"])
        == verdict,
        "raw verdict rejudgment differs",
    )
    check(manifest["status"] == ("pass" if verdict["passed"] else "fail"), "raw status differs")
    provenance = (
        "installed_sources_match",
        "versions_match",
        "snapshot_sources_match",
        "cpu_environment_built",
    )
    check(
        all(verdict[name] for name in provenance), "installed source/package/CPU provenance failed"
    )
    numeric, recipes, ptx = [], [], []
    for name in sorted(retained):
        row = read(ROOT / name)
        if name.endswith("-ptx.json"):
            required = [
                op + "." + mode + ".f64" for op in ("add", "mul", "div") for mode in ("rm", "rp")
            ]
            check(row["required_instructions"] == required, "PTX instruction contract differs")
            ptx.append(
                dict(
                    family=row["family"],
                    instructions_present=all(op in row["ptx"] for op in required),
                    bytes=len(row["ptx"].encode()),
                )
            )
        elif "/glm-comparisons/" in name:
            case = row["case"]
            audit = comparison(case["family"], case["arrays"], row["comparison"], row["decimal220"])
            check(
                Path(name).stem == digest(case["id"].encode())[:16],
                "numeric artifact binding differs",
            )
            audit.update(
                case=case["id"],
                expected_status_matches=case["status"] is None or case["status"] == audit["status"],
            )
            check(row["status"] == audit["status"], "retained status disagrees with bounds")
            numeric.append(audit)
        else:
            check(
                Path(name).stem == digest(row["case"].encode())[:16],
                "recipe artifact binding differs",
            )
            recipes.append(recipe(row))
    numeric_failures = [
        r["case"]
        for r in numeric
        if not all(
            r[k]
            for k in ("encloses", "identity_matches", "method_matches", "expected_status_matches")
        )
    ]
    if verdict["passed"]:
        check(
            len(numeric) == 59 and len(recipes) == 16 and len(ptx) == 2,
            "passing artifact counts differ",
        )
        check(
            not numeric_failures and all(r["instructions_present"] for r in ptx),
            "passing numerical/PTX audit failed",
        )
    groups = {}
    for group, files in protocol["groups"].items():
        names = {p[:-3].replace("/", ".") for p in files}
        selected = [r for r in verdict["cases"] if r["case"].split("::")[0] in names]
        groups[group] = dict(Counter(row["status"] for row in selected))
    failures = [
        dict(case=node.get("classname") + "::" + node.get("name"), message=child.get("message", ""))
        for node in ET.parse(ROOT / "junit.xml").getroot().iter("testcase")
        for child in node
        if child.tag in ("failure", "error", "skipped")
    ]
    return dict(
        manifest_sha256=digest((ROOT / "manifest.json").read_bytes()),
        execution_revision=manifest["revision"],
        raw_verdict=verdict["passed"],
        dispatch_sources_verified=85,
        installed_core_files=len(result["installed_sources"]),
        packages_verified=len(result["packages"]),
        raw_artifacts_verified=len(manifest["artifacts"]),
        retained_artifacts=len(retained),
        retained_bytes=total_bytes,
        missing_artifacts=sorted(set(protocol["retained_artifacts"]) - set(retained)),
        groups=groups,
        failures=failures,
        numerical_cases=len(numeric),
        numerical_failures=numeric_failures,
        comparison_audits=len(numeric) + sum(r["comparisons"] for r in recipes),
        models_replayed=sum(r["models_replayed"] for r in recipes),
        recipes=recipes,
        ptx=ptx,
        worker_seconds=result["worker_wall_seconds"],
        dispatch_seconds=(
            datetime.fromisoformat(manifest["finished"])
            - datetime.fromisoformat(manifest["started"])
        ).total_seconds(),
        scope="Bounded correctness and stored-input replay. No full R1/R4, Normal, quality, speed, E4 or adoption claim.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    if args.check:
        check(result == read(ROOT / "analysis.json"), "derived analysis differs")
    else:
        (ROOT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "raw_verdict",
                    "groups",
                    "retained_artifacts",
                    "comparison_audits",
                    "models_replayed",
                )
            },
            indent=2,
        )
    )
