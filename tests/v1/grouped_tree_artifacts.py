"""Lossless grouped prediction inputs and fresh core-only artifact replay."""

import base64

import numpy as np


def snapshot(data):
    result = {}
    for name in ('values', 'row_ids'):
        a = np.ascontiguousarray(getattr(data, name))
        result[name] = dict(dtype=a.dtype.str, shape=list(a.shape), data_base64=base64.b64encode(a.tobytes()).decode())
    return dict(feature_names=list(data.feature_names), arrays=result)


FRESH = """
import sys
for name in ('openboost.device_group_tree', 'openboost._device_group_tree_kernels',
             'openboost.device_groups', 'openboost._device_group_kernels',
             'openboost.device_tree', 'openboost.device_runs', 'openboost.device_inputs',
             'openboost.runs'):
    sys.modules[name] = None
import base64, json
from pathlib import Path
import numpy as np
from openboost import NumericData
from openboost.tree import Tree
record = json.loads(Path(sys.argv[1]).read_text())
arrays = {name: np.frombuffer(base64.b64decode(a['data_base64']), dtype=a['dtype']).reshape(a['shape'])
          for name, a in record['input']['arrays'].items()}
data = NumericData(arrays['values'], arrays['row_ids'], tuple(record['input']['feature_names']))
expected = {o['run_id']: o['result'] for o in record['schedules'][0]['observed']}
for job in record['jobs']:
    tree = Tree.from_record(job['model'])
    np.testing.assert_array_equal(tree.predict(data), expected[job['run_id']])
print(json.dumps(dict(models=len(record['jobs']), predictions_matched=True, training_imports_denied=True)))
"""
