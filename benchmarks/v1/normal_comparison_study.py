"""Freeze 092-A's historical mapping and independent CPU comparison evidence."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from tests.v1.reference.device_normal import fixture, predict, rounds
from tests.v1.reference.normal_acceptance import compare_precisions, loss_difference
from tests.v1.reference.normal_comparison import compare

from benchmarks.v1.normal_acceptance_trace import pack, unpack

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "benchmarks/v1/evidence/cuda-acceptance-091"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapping():
    """One explicit disposition per original node, without inventing new outcomes."""
    protocol = json.loads((ROOT / "v1-sprints/090-normal-run6.json").read_text())
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    outcomes = {}
    for case in ET.parse(ARCHIVE / "junit.xml").getroot().iter("testcase"):
        node = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        outcomes[node] = "failed" if case.find("failure") is not None else "passed"
    categories = {
        "test_device_normal_runtime_cuda.py": "mapped-transactions",
        "test_device_normal_recipe_cuda.py": "normal-recipes",
        "test_device_normal_extension_cuda.py": "installed-d2-inference",
    }
    rows = []
    for node in protocol["expected_cases"]:
        source = node.split("::")[0]
        category = categories.get(Path(source).name, "unchanged-operations")
        rows.append(
            dict(
                historical_node=node,
                historical_source_sha256=manifest["sources"][source],
                historical_outcome=outcomes[node],
                disposition=category,
                planned_requirement=f"092/{category}/{node}",
                predicate_supersession=outcomes[node] == "failed",
                revised_status="not_implemented_or_run",
            )
        )
    return dict(
        schema="openboost-092-historical-mapping-v1",
        historical_revision=manifest["revision"],
        historical_manifest_sha256=digest(ARCHIVE / "manifest.json"),
        original_protocol_sha256=digest(ROOT / "v1-sprints/090-normal-run6.json"),
        original_tolerances={key: protocol[key] for key in ("float32", "normal_float32")},
        interpretation="Planned requirements are not collected test IDs or passing revised cases. Historical failures remain failures.",
        counts=dict(Counter(row["disposition"] for row in rows)),
        cases=rows,
    )


def evaluate(identifier, args, **metadata):
    result = compare(*args)
    oracle = compare_precisions(*args)
    values = [Decimal(oracle[f"decimal{p}"]) for p in (60, 100)]
    if not oracle["estimates_agree"]:
        values += [loss_difference(*args, precision=p) for p in (160, 220)]
        with localcontext() as context:
            context.prec = 240
            agreement = abs(values[-1] - values[-2]) <= max(map(abs, values[-2:])) * Decimal(
                "1e-40"
            )
        oracle["higher_precision"] = dict(
            decimal160=str(values[-2]), decimal220=str(values[-1]), estimates_agree=agreement
        )
        if not agreement:
            raise AssertionError(f"oracle precision unresolved: {identifier}")
    enclosed = None
    if result.lower is not None:
        enclosed = all(
            Decimal.from_float(result.lower) <= v <= Decimal.from_float(result.upper)
            for v in values
        )
        if not enclosed:
            raise AssertionError(f"oracle outside enclosure: {identifier}")
    return dict(
        id=identifier,
        **metadata,
        inputs={
            key: pack(np.asarray(value, dtype=np.float64))
            for key, value in zip(
                ("before", "after", "target", "offset", "weight"), args, strict=True
            )
        },
        comparison=asdict(result),
        high_precision=oracle,
        oracle_enclosed=enclosed,
    )


def observations():
    rows = []
    for order in ("forward", "reverse"):
        path = ARCHIVE / f"normal/acceptance/{order}.json"
        trace = json.loads(path.read_text())
        for step in trace["steps"]:
            for trial in step["trials"]:
                for dataset in ("training", "validation"):
                    data = trace["inputs"][dataset]
                    args = (
                        unpack(step["before"][f"{dataset}_raw"]),
                        unpack(trial["proposal"][f"{dataset}_raw"]),
                        unpack(data["target"]).ravel(),
                        unpack(data["offset"]),
                        unpack(data["weight"]),
                    )
                    rows.append(
                        evaluate(
                            f"run7/{order}/channel{step['channels'][0]}/alpha{trial['coefficient']}/{dataset}",
                            args,
                            origin="stored_gpu_inputs_reanalyzed_on_cpu",
                            trace_sha256=digest(path),
                            original_accepted=trial["accepted"],
                        )
                    )
    return rows


def reference_proposals():
    """Actual proposals of the frozen CPU oracle, explicitly not a CUDA trajectory."""
    rows = []
    for case in ("weighted", "d2"):
        f = fixture(case)
        for mode, damping in (("ordinary", 0), ("natural", 0), ("natural", 0.25)):
            for update in ("joint", "forward", "reverse"):
                _, steps = rounds(
                    case,
                    mode=mode,
                    damping=damping,
                    update=update,
                    count=1,
                    depth=1,
                    minimum=1 if case == "d2" else None,
                    rate=8,
                )
                for index, step in enumerate(steps):
                    delta = np.zeros_like(step["before"])
                    for channel, nodes in zip(step["channels"], step["nodes"], strict=True):
                        delta[:, channel] = predict(nodes, f["x"])
                    for coefficient, _, decision in step["attempts"]:
                        after = step["before"] + coefficient * delta
                        args = (step["before"], after, f["target"], f["offset"], f["weight"])
                        rows.append(
                            evaluate(
                                f"cpu-reference/{case}/{mode}/{damping}/{update}/step{index}/alpha{coefficient}",
                                args,
                                origin="frozen_original_row_cpu_proposal",
                                original_decision=decision,
                            )
                        )
    return rows


def analytic_cases():
    rows = []
    for name, before, after, target in (
        ("tiny-improvement", [[2.0**-30, 0]], [[0, 0]], [0]),
        ("scale-cancellation", [[0, 0]], [[0, 2.0**-52]], [1]),
        ("unchanged", [[1, 0]], [[1, 0]], [0]),
        ("clear-worsening", [[0, 0]], [[1, 0]], [0]),
        ("outside-exponent-range", [[0, 33]], [[1, 33]], [0]),
    ):
        rows.append(
            evaluate(
                f"analytic/{name}",
                (before, after, target, [[0, 0]], [1]),
                origin="declared_analytic_case",
            )
        )
    return rows


def study():
    rows = observations() + analytic_cases() + reference_proposals()
    paths = [Path(__file__), ROOT / "benchmarks/v1/normal_acceptance_trace.py"]
    paths += [
        ROOT / "tests/v1/reference" / f"{name}.py"
        for name in (
            "normal_comparison",
            "normal_acceptance",
            "device_normal",
            "device_rounds",
            "device_splits",
            "device_histogram",
        )
    ]
    return dict(
        schema="openboost-092-comparison-study-v1",
        device_execution=False,
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        argv=sys.argv,
        environment=dict(
            python=platform.python_version(), numpy=np.__version__, os=platform.platform()
        ),
        sources={str(p.relative_to(ROOT)): digest(p) for p in paths},
        historical_mapping_sha256=hashlib.sha256(
            json.dumps(mapping(), sort_keys=True).encode()
        ).hexdigest(),
        counts=dict(Counter(row["comparison"]["status"] for row in rows)),
        cases=rows,
        limitation="Independent CPU mathematics. Not public components, new device evidence, revised conformance, performance or author benefit.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("mapping", "study"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = mapping() if args.kind == "mapping" else study()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
