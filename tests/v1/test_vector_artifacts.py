"""119 exact inputs, actual host constructors and fresh installed inference guards."""

import base64
import json
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from openboost import device_multi_squared as objective
from openboost.artifacts import Model
from openboost.multioutput import MultiOutputModel, TargetScale

from .multi_squared_artifacts import fresh_command, input_snapshot
from .reference import multi_squared as ref
from .test_multi_squared_reference import prepared


@pytest.mark.parametrize("width", [1, 2, 4])
def test_lossless_recipe_inputs_and_geometry(width):
    for problem in prepared(width)[:2]:
        snapshot = json.loads(json.dumps(input_snapshot(problem), allow_nan=False))
        originals = dict(
            features=problem.data.values,
            row_ids=problem.data.row_ids,
            target=problem.target,
            offset=problem.offset,
            weight=problem.weight,
        )
        for name, original in originals.items():
            item = snapshot[name]
            data = base64.b64decode(item["data_base64"], validate=True)
            restored = np.frombuffer(data, dtype=item["dtype"]).reshape(item["shape"])
            assert restored.dtype == original.dtype and restored.shape == original.shape
            assert data == np.ascontiguousarray(original).tobytes()
        target, offset = objective._host_arrays(problem)
        np.testing.assert_array_equal(target, problem.target)
        np.testing.assert_array_equal(offset, problem.offset)
        assert np.isnan(problem.data.values).any()


class DeviceBoundaryReached(Exception):
    pass


def stop_before_device():
    raise DeviceBoundaryReached


def test_actual_numerical_host_constructors_reach_device_boundary(monkeypatch, tmp_path):
    from . import test_device_multi_squared_cuda as gpu

    monkeypatch.setattr(gpu, "ExecutionContext", stop_before_device)
    for case in ref.CASES:
        with pytest.raises(DeviceBoundaryReached):
            gpu.test_comparison_enclosure_reverse_and_snapshot_ownership(
                case, monkeypatch, tmp_path
            )


def test_actual_recipe_host_constructors_reach_device_boundary(monkeypatch, tmp_path):
    from . import test_device_multi_squared_recipes_cuda as gpu

    monkeypatch.setattr(gpu, "ExecutionContext", stop_before_device)
    for width in (1, 2, 4):
        for mode in ("independent", "shared", "projected"):
            for depth in (1, 2):
                for step, rate in (("fixed", 0.5), ("backtracking", 8)):
                    with monkeypatch.context() as patch, pytest.raises(DeviceBoundaryReached):
                        gpu.test_two_round_recipe_and_every_actual_comparison(
                            width, mode, depth, step, rate, patch, tmp_path
                        )
    with monkeypatch.context() as patch, pytest.raises(DeviceBoundaryReached):
        gpu.test_joint_driver_separate_anchors_and_noops(patch)
    for step in ("fixed", "backtracking"):
        with pytest.raises(DeviceBoundaryReached):
            gpu.test_reporting_tie_cannot_supply_acceptance_or_best(step)
    for mode in ("independent", "shared"):
        with pytest.raises(DeviceBoundaryReached):
            gpu.test_scaling_constant_target_and_original_unit_fresh_inference(mode, tmp_path)


def test_actual_vector_tree_host_constructors_reach_device_boundary(monkeypatch, tmp_path):
    from . import test_device_vector_tree_cuda as gpu

    monkeypatch.setattr(gpu, "ExecutionContext", stop_before_device)
    for width in (1, 2, 4):
        for projected in (False, True):
            for depth in (0, 1, 2):
                with pytest.raises(DeviceBoundaryReached):
                    gpu.test_shared_topology_projected_splits_full_leaves(
                        width, projected, depth, tmp_path
                    )


def test_fresh_command_honors_explicit_cpu_python(monkeypatch):
    monkeypatch.setenv("OPENBOOST_FRESH_CPU_PYTHON", "/explicit/cpu/python")
    command = fresh_command("print('inference')", "argument")
    assert command[:3] == ["/explicit/cpu/python", "-I", "-c"]
    assert command[-1] == "argument"


@pytest.mark.parametrize(
    "module", ["cupy", "numba", "openboost.device_runtime", "openboost.device_recipes"]
)
def test_fresh_inference_denies_cuda_and_device_training(module):
    result = subprocess.run(fresh_command(f"import {module}"), capture_output=True, text=True)
    assert result.returncode != 0 and "ModuleNotFoundError" in result.stderr


def test_scaled_fresh_core_inference_preserves_metadata_and_offsets(tmp_path):
    train, validation, _ = prepared(2)
    train = replace(train, target=np.column_stack((train.target[:, 0], np.full(8, 7.0))))
    scale = TargetScale.fit(train)
    assert scale.constant == (False, True)
    model = MultiOutputModel(Model(("a", "b"), [1, 2]), scale)
    path, inputs, offsets, output = [tmp_path / n for n in ("m.json", "x.npy", "o.npy", "p.npy")]
    model.save(path)
    np.save(inputs, validation.data.values)
    np.save(offsets, validation.offset)
    code = "\n".join(
        [
            "import sys, numpy as np",
            "from openboost.multioutput import MultiOutputModel",
            "from openboost.data import NumericData",
            "x=np.load(sys.argv[2]); data=NumericData(x,np.arange(len(x)),('a','b'))",
            "m=MultiOutputModel.load(sys.argv[1])",
            "assert m.target_scale.constant == (False, True)",
            "np.save(sys.argv[4],m.predict(data,offset=np.load(sys.argv[3])))",
        ]
    )
    subprocess.run(
        fresh_command(code, str(path), str(inputs), str(offsets), str(output)), check=True
    )
    np.testing.assert_array_equal(
        np.load(output), model.predict(validation.data, offset=validation.offset)
    )
