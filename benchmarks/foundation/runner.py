"""Validate returned smoke evidence; usable offline without Modal installed."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REQUIRED_TESTS = {"test_device_interop", "test_normal_gpu_fit"}


def validate_result(manifest, result):
    if result.get("returncode") != 0 or result.get("timed_out", True):
        raise ValueError("Remote test process failed or timed out")
    if manifest.get("source_dirty") is not False:
        raise ValueError("Source must be committed and clean")
    for key in ("wheel_sha256", "source_sha"):
        if not manifest.get(key) or result.get(key) != manifest[key]:
            raise ValueError(f"Provenance mismatch: {key}")
    try:
        root = ET.fromstring(result.get("junit", ""))
    except ET.ParseError as exc:
        raise ValueError("Missing or invalid JUnit report") from exc
    cases = list(root.iter("testcase"))
    names = [case.get("name") for case in cases]
    if len(names) != len(REQUIRED_TESTS) or set(names) != REQUIRED_TESTS:
        raise ValueError("Required GPU cases missing or duplicated")
    if any(list(root.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("GPU cases failed, errored, or skipped")
    env = result.get("environment", {})
    checks = result.get("checks", {})
    if not env.get("cuda_available") or not env.get("gpu_name"):
        raise ValueError("No real CUDA environment evidence")
    if not (
        checks.get("interop") is True
        and checks.get("native_tree_calls") == 4
        and checks.get("device_objective_calls") == 2
        and checks.get("installed_files_verified", 0) > 0
        and checks.get("dataset_sha256")
    ):
        raise ValueError("Missing installed-wheel/device-path checks (possible fallback)")


def main():
    directory = Path(sys.argv[1])
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        result = json.loads((directory / "results.json").read_text())
        validate_result(manifest, result)
    except (OSError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Foundation smoke evidence passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
