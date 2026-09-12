"""Tiny stdlib-only receipt controls: no validation jobs or numerical imports."""

import ast
import copy
import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("gate", Path(__file__).with_name("check_modal_validation.py"))
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class Check(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.commit = "a" * 40
        self.inventory = {"src/openboost/example.py": dict(mode="100644", type="blob", oid="b" * 40)}
        self.identity = gate.digest(self.inventory)
        policy_path = Path(__file__).with_name("modal-validation-policy.json")
        if not policy_path.is_file():
            policy_path = Path(__file__).resolve().parent.parent / "modal-validation-policy.json"
        policy = json.loads(policy_path.read_text())
        self.actual_policy = copy.deepcopy(policy)
        policy.update(gpu_tests=["tests/v1/test_case.py"], expected_gpu_cases=1)
        for rule in policy["jobs"].values():
            rule["allowed_skips"] = []
        self.put(gate.POLICY, policy)
        self.write("src/openboost/example.py", b"x = 1\n")
        protocol = dict(source_commit=self.commit, inventory_sha256=self.identity,
                        policy=dict(path=gate.POLICY, sha256=self.descriptor(gate.POLICY)["sha256"]),
                        gpu_tests=policy["gpu_tests"], expected_gpu_cases=1)
        self.results = {}
        for phase, names in gate.PHASES.items():
            prefix = gate.EVIDENCE + phase + "/"
            xml = b'<testsuites><testsuite tests="1" failures="0" errors="0"><testcase classname="tests.v1.test_case" name="test_ok"/></testsuite></testsuites>'
            jobs, artifact_names = {}, []
            for name in names:
                argv = gate.expected_command(name, protocol)
                collection = name + "-collection.json"
                names_for_job = [name + ".stdout.txt", name + ".stderr.txt", name + ".xml", collection]
                self.write(prefix + names_for_job[0], b"original process log\n")
                self.write(prefix + names_for_job[1], b"")
                self.write(prefix + names_for_job[2], xml)
                self.put(prefix + collection, dict(case_count=1, nodeids=["tests/v1/test_case.py::test_ok"],
                         command=[value for value in argv if not value.startswith("--junitxml=")] + ["--collect-only"]))
                artifact_names.extend(names_for_job)
                jobs[name] = dict(status="pass", exit_code=0, source_commit=self.commit,
                                 inventory_sha256=self.identity, command=argv, wall_seconds=1,
                                 artifacts=names_for_job, junit=name + ".xml", collection=collection, cases=1)
            artifacts = {name: self.descriptor(prefix + name) for name in artifact_names}
            self.results[phase] = dict(
                schema="openboost-pr27-modal-validation-v1", source_commit=self.commit,
                inventory=self.inventory, inventory_sha256=self.identity, phase=phase,
                passed=True, artifacts=artifacts, jobs=jobs, protocol=protocol,
                protocol_sha256=gate.digest(protocol), policy_sha256=protocol["policy"]["sha256"],
                platform=dict(python="3.10.9" if phase == "cpu310" else "3.12.1", provider="Modal", os="Linux"),
                collection_complete=True, git_clean=True, gpu=dict(hardware="Tesla T4"),
                installed_sources={"src/openboost/example.py":self.descriptor("src/openboost/example.py")["sha256"]},
            )

    def write(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def put(self, name, value):
        self.write(name, encoded(value))

    def descriptor(self, name):
        raw = (self.root / name).read_bytes()
        return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

    def publish(self):
        phases = {}
        for phase, result in self.results.items():
            name = gate.EVIDENCE + phase + "/manifest.json"
            self.put(name, result)
            phases[phase] = self.descriptor(name)
        self.put(gate.EVIDENCE + "report.json", dict(
            schema="openboost-pr27-modal-validation-index-v1", source_commit=self.commit,
            inventory_sha256=self.identity, phases=phases,
        ))

    def verify(self, changed=False):
        def read(path, maximum=gate.MAX_JSON):
            raw = (self.root / gate.safe(path)).read_bytes()
            gate.require(len(raw) <= maximum, "byte limit")
            return raw
        def inventory(revision):
            value = copy.deepcopy(self.inventory)
            if changed and revision == "HEAD":
                value["src/openboost/example.py"]["oid"] = "c" * 40
            return value
        with patch.object(gate, "blob", read), patch.object(gate, "inventory", inventory):
            return gate.verify()

    def test_complete_bound_pass(self):
        self.publish()
        self.assertEqual(self.verify()["testcase_counts"], dict(cpu310=1, cpu312=1, gpu=1, **{"gpu-prerequisites": 1}))

    def test_actual_policy_all_thirteen_selectors_are_accepted(self):
        selectors = self.actual_policy["gpu_tests"]
        self.assertEqual(len(selectors), 13)
        self.assertEqual(sum("::" in selector for selector in selectors), 2)
        self.assertEqual(gate.validate_gpu_tests(selectors), selectors)

    def test_gpu_selector_rejects_unsafe_or_nonfunction_suffix(self):
        for selector in ("../test_x.py", "tests/v1/test_x.py::f/../x", "tests/v1/test_x.py::",
                         "tests/v1/test_x.py::f::g", "tests/v1/test_x.py[a]", True):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                gate.validate_gpu_tests([selector])

    def test_exact_gpu_option_order_matches_actual_runner_source(self):
        here = Path(__file__).resolve().parent
        runner = here / "modal_validate_checkpoint.py"
        if not runner.is_file():
            runner = here.parent / "openboost-pr-validation/runner.py"
        lines = [line for line in runner.read_text().splitlines() if 'job("gpu",' in line]
        self.assertEqual(len(lines), 1)
        observed = re.findall(r"--(?:basetemp|junitxml)=", lines[0])
        argv = gate.expected_command("gpu", self.actual_policy)
        expected = [value.split("=", 1)[0] + "=" for value in argv
                    if value.startswith(("--basetemp=", "--junitxml="))]
        self.assertEqual(observed, expected)

    def test_all_pinned_executables_are_outside_checkout(self):
        for name in set().union(*gate.PHASES.values()):
            with self.subTest(name=name):
                argv = gate.expected_command(name, self.actual_policy)
                executable = argv[argv.index("--python") + 1] if "--python" in argv else argv[0]
                self.assertTrue(executable.startswith("/tmp/pr27-environment/bin/"))
                self.assertFalse(Path(executable).is_relative_to("/tmp/pr27-source"))

    def test_actual_runner_pytest_commands_match_checked_external_environment(self):
        runner = Path(__file__).with_name("modal_validate_checkpoint.py")
        node = next(node for node in ast.walk(ast.parse(runner.read_text()))
                    if isinstance(node, ast.FunctionDef) and node.name == "pytest_command")
        namespace = dict(repo=Path("/tmp/pr27-source"), environment=Path("/tmp/pr27-environment"))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-pytest-command", "exec"), namespace)
        for name in ("cpu310", "cpu312", "gpu", "gpu-prerequisites"):
            with self.subTest(name=name):
                expected = gate.expected_command(name, self.actual_policy)
                start = expected.index("-o")
                targets = expected[5:start]
                extra = expected[start + 5:]
                self.assertEqual(namespace["pytest_command"](targets, extra), expected)

    def test_changed_source_fails(self):
        self.publish()
        with self.assertRaisesRegex(ValueError, "candidate changed"):
            self.verify(changed=True)

    def test_failed_phase_is_not_promoted(self):
        self.results["gpu"]["passed"] = False
        self.publish()
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.verify()

    def test_nonzero_job_fails(self):
        self.results["cpu310"]["jobs"]["cpu310"]["exit_code"] = 7
        self.publish()
        with self.assertRaisesRegex(ValueError, "process result"):
            self.verify()

    def test_changed_raw_log_fails(self):
        self.publish()
        self.write(gate.EVIDENCE + "gpu/gpu.stdout.txt", b"changed")
        with self.assertRaisesRegex(ValueError, "artifact bytes"):
            self.verify()

    def test_missing_result_fails(self):
        with self.assertRaises(FileNotFoundError):
            self.verify()


if __name__ == "__main__":
    unittest.main()
