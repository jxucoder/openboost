"""Fresh CPU tree replay of lossless grouped-growth evidence."""

FRESH = """
import sys
for name in ('openboost.device_group_growth', 'openboost.device_group_tree',
             'openboost.device_tree', 'openboost.device_groups', 'openboost.device_inputs'):
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
count = 0
for schedule in record['schedules']:
    for result in schedule['observed']:
        tree = Tree.from_record(result['model'])
        np.testing.assert_array_equal(tree.predict(data), result['prediction'])
        count += 1
print(json.dumps(dict(models=count, predictions_matched=True, training_imports_denied=True)))
"""
