"""Hand-derived partition and leaf-region controls, not CUDA execution."""

import json
import subprocess

import numpy as np
import pytest

from .grouped_tree_artifacts import FRESH, snapshot
from .multi_squared_artifacts import fresh_command
from .reference.grouped_tree import partition, predict

CODES = np.array([[0, 1, 2, 0, 1], [2, 0, 1, 1, 0]])
MISSING = np.array([[False, False, False, True, False], [False, True, False, False, False]])


@pytest.mark.parametrize('missing_left, expected', [(False, ([4, 1], [3, 2])), (True, ([4, 3, 1], [2]))])
def test_threshold_inclusive_missing_direction_and_original_order(missing_left, expected):
    assert partition(CODES, MISSING, [4, 3, 1, 2], (0, 1, missing_left)) == expected


@pytest.mark.parametrize('rows', [[], [3], [2, 3]])
def test_empty_children_preserve_selected_rows(rows):
    left, right = partition(CODES, MISSING, rows, (0, 0, False))
    assert left == [] and right == rows


@pytest.mark.parametrize('width', [1, 2, 3])
def test_variable_depth_leaf_regions_ignore_internal_node_values(width):
    topology = ((0, 0, True, 1, 2), (-1, -1, False, -1, -1),
                (1, 0, False, 3, 4), (-1, -1, False, -1, -1), (-1, -1, False, -1, -1))
    values = np.array([[999] * width, [2] * width, [-999] * width, [5] * width, [-7] * width])
    expected = np.array([[2] * width, [-7] * width, [-7] * width, [2] * width, [5] * width], np.float32)
    np.testing.assert_array_equal(predict(CODES, MISSING, topology, values), expected)
    np.testing.assert_array_equal(predict(CODES, MISSING, ((-1, -1, False, -1, -1),), [[-0.] * width]),
                                  np.full((5, width), -0., np.float32))


@pytest.mark.parametrize('width', [1, 2, 3])
def test_fresh_scalar_vector_artifact_uses_lossless_missing_inputs(width, tmp_path):
    from openboost import NumericData
    from openboost.binning import Binning
    from openboost.tree import Tree

    data = NumericData([[0], [1], [np.nan]], [4, 2, 7], ('x',))
    tree = Tree(Binning(('x',), (np.array([0.5]),)), [0, -1, -1], [0, -1, -1],
                [True, False, False], [1, -1, -1], [2, -1, -1], [[99] * width, [2] * width, [-3] * width])
    path = tmp_path / 'fresh.json'
    path.write_text(json.dumps(dict(
        input=snapshot(data), jobs=[dict(run_id='run-1', model=tree.record())],
        schedules=[dict(observed=[dict(run_id='run-1', result=[[2] * width, [-3] * width, [2] * width])])],
    ), allow_nan=False))
    result = subprocess.run(fresh_command(FRESH, str(path)), check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == dict(models=1, predictions_matched=True, training_imports_denied=True)
