"""Metadata-only interpreter admission checks; never import/launch a Modal app."""

import ast
import copy
import json
import os
import runpy
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).with_name("modal_validate_checkpoint.py")
if not PATH.exists():
    PATH = Path(__file__).with_name("runner-correction.py")
MODULE = runpy.run_path(str(PATH), run_name="pr27_corrected_snapshot")


def protocol(phase="cpu312"):
    return dict(schema=MODULE["SCHEMA"], repository=MODULE["REPOSITORY"], source_commit="a" * 40,
                inventory_sha256="b" * 64, phase=phase, python="3.10" if phase == "cpu310" else "3.12",
                build_packages=MODULE["BUILD_PACKAGES"], uv="0.12.1",
                resources=dict(cpu=2, memory_mib=8192, timeout_seconds=2100 if phase == "gpu" else 1200,
                               work_seconds=1800 if phase == "gpu" else 1080, max_containers=1, retries=0),
                maximum_return_bytes=MODULE["RETURN_LIMIT"], maximum_log_bytes=16 * 1024**2,
                policy=dict(path=".github/modal-validation-policy.json", sha256="c" * 64),
                expected_core_modules=63, allowed_skips=MODULE["allowed_job_skips"](phase),
                gpu_tests=["tests/v1/test_fixture_cuda.py"], expected_gpu_cases=1)


def cases(*, skips=True, failure=False):
    root = ET.Element("testsuite")
    ET.SubElement(root, "testcase", classname="tests.v1.test_fixture", name="test_pass")
    if failure:
        ET.SubElement(ET.SubElement(root, "testcase", classname="tests.v1.test_fixture", name="test_fail"), "failure")
    if skips:
        for record in MODULE["CPU_SKIPS"]:
            classname, name = record["id"].split("::")
            ET.SubElement(ET.SubElement(root, "testcase", classname=classname, name=name), "skipped", message=record["reason"])
    return list(root)


class InterpreterPolicyTests(unittest.TestCase):
    def test_cpu310_controller_and_child_are_explicitly_distinct(self):
        self.assertEqual(MODULE["interpreter_policy"]("cpu310"),
                         dict(controller="3.12", child="3.10.17", managed_install="3.10.17"))

    def test_cpu312_uses_controller_interpreter_without_download(self):
        self.assertEqual(MODULE["interpreter_policy"]("cpu312"),
                         dict(controller="3.12", child="3.12", managed_install=None))

    def test_gpu_preserves_python312(self):
        self.assertEqual(MODULE["interpreter_policy"]("gpu"),
                         dict(controller="3.12", child="3.12", managed_install=None))

    def test_unknown_phase_rejected(self):
        with self.assertRaises(ValueError):
            MODULE["interpreter_policy"]("cpu311")

    def test_compatible_serialized_controller_accepted(self):
        MODULE["validate_controller"]((3, 12, 12), "3.12")

    def test_original_mismatch_rejected_before_build(self):
        with self.assertRaisesRegex(ValueError, "controller and image"):
            MODULE["validate_controller"]((3, 12, 12), "3.10")

    def test_wrong_local_controller_minor_rejected(self):
        for version in ((3, 10, 17), (3, 13, 0)):
            with self.subTest(version=version), self.assertRaises(ValueError):
                MODULE["validate_controller"](version, "3.12")

    def test_exact_managed_cpu310_patch_required(self):
        self.assertEqual(MODULE["validate_child_version"]("cpu310", "3.10.17"), "3.10.17")
        for version in ("3.10", "3.10.18", "3.12.10"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                MODULE["validate_child_version"]("cpu310", version)

    def test_actual_cpu312_child_minor_required(self):
        for phase in ("cpu312", "gpu"):
            self.assertEqual(MODULE["validate_child_version"](phase, "3.12.10"), "3.12.10")
            with self.assertRaises(ValueError):
                MODULE["validate_child_version"](phase, "3.10.17")

    def test_only_controller_fallback_references_sys_executable(self):
        tree = ast.parse(PATH.read_text())
        nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                 and isinstance(node.value, ast.Name) and node.value.id == "sys"
                 and node.attr == "executable"]
        self.assertEqual(len(nodes), 1)
        self.assertIn('else sys.executable', PATH.read_text().splitlines()[nodes[0].lineno - 1])

    def test_main_rejects_incompatible_host_before_creating_output(self):
        packet = protocol()
        with tempfile.TemporaryDirectory(prefix="pr27-interpreter-metadata-") as name:
            root = Path(name)
            source, output = root / "protocol.json", root / "result"
            source.write_text(json.dumps(packet))
            with patch.object(sys, "argv", [str(PATH), str(source), str(output)]), \
                    patch.object(sys, "version_info", (3, 10, 17)), \
                    patch.dict(os.environ, MODAL_PROFILE="edamame-labs"), \
                    self.assertRaisesRegex(ValueError, "controller and image"):
                MODULE["main"]()
            self.assertFalse(output.exists())


class ExactSkipTests(unittest.TestCase):
    def test_exact_two_cpu_skips_remain_separate_from_passes(self):
        value = MODULE["junit_observations"](cases())
        self.assertEqual({key: value[key] for key in ("cases", "passed", "failures", "skipped")},
                         dict(cases=3, passed=1, failures=0, skipped=2))
        for name in ("cpu310", "cpu312"):
            MODULE["validate_junit_observations"](value, name, 3)

    def test_skip_reason_must_match_exactly(self):
        value = MODULE["junit_observations"](cases())
        value["observed_skips"][0]["reason"] += " changed"
        with self.assertRaises(ValueError):
            MODULE["validate_junit_observations"](value, "cpu312", 3)

    def test_skip_id_requires_full_classname(self):
        value = MODULE["junit_observations"](cases())
        value["observed_skips"][0]["id"] = value["observed_skips"][0]["id"].split(".")[-1]
        with self.assertRaises(ValueError):
            MODULE["validate_junit_observations"](value, "cpu312", 3)

    def test_missing_expected_skip_is_not_silently_accepted(self):
        value = MODULE["junit_observations"](cases(skips=False))
        with self.assertRaises(ValueError):
            MODULE["validate_junit_observations"](value, "cpu312", 1)

    def test_extra_skip_is_rejected(self):
        entries = cases()
        extra = ET.Element("testcase", classname="tests.v1.test_fixture", name="test_extra")
        ET.SubElement(extra, "skipped", message="another environment restriction")
        value = MODULE["junit_observations"]([*entries, extra])
        with self.assertRaises(ValueError):
            MODULE["validate_junit_observations"](value, "cpu312", 4)

    def test_gpu_and_prerequisites_require_zero_skips(self):
        for name in ("gpu", "gpu-prerequisites"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                MODULE["validate_junit_observations"](MODULE["junit_observations"](cases()), name, 3)
            MODULE["validate_junit_observations"](MODULE["junit_observations"](cases(skips=False)), name, 1)

    def test_original_failure_cannot_be_excused_by_allowed_skips(self):
        value = MODULE["junit_observations"](cases(failure=True))
        self.assertEqual((value["cases"], value["passed"], value["failures"], value["skipped"]), (4, 1, 1, 2))
        with self.assertRaises(ValueError):
            MODULE["validate_junit_observations"](value, "cpu312", 4)

    def test_protocol_requires_exact_cpu_skip_records(self):
        for phase in ("cpu310", "cpu312", "gpu"):
            value = protocol(phase)
            MODULE["validate_protocol"](value)
            value["allowed_skips"] = [] if phase != "gpu" else copy.deepcopy(MODULE["CPU_SKIPS"])
            with self.assertRaises(ValueError):
                MODULE["validate_protocol"](value)

    def test_non_cpu_jobs_do_not_inherit_main_cpu_skips(self):
        for name in ("build", "docs", "lint", "gpu-prerequisites", "gpu"):
            self.assertEqual(MODULE["allowed_job_skips"](name), [])

    def test_skip_policy_return_does_not_mutate_constant(self):
        value = MODULE["allowed_job_skips"]("cpu312")
        value[0]["reason"] = "changed"
        self.assertNotEqual(MODULE["CPU_SKIPS"][0]["reason"], "changed")


if __name__ == "__main__":
    unittest.main()
