"""Preregistered near-tie diagnostic, explicitly not repaired structural parity."""

import json

import numpy as np
import pytest

from openboost import device_tree as trees
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext
from openboost.ops import candidates, histogram, score
from openboost.stats import RowFields
from openboost.tree import depthwise

from .reference.device_splits import enumerate_candidates
from .test_device_normal_cuda import close
from .test_device_normal_reference import prepared_fixture

pytestmark = pytest.mark.gpu


def test_retained_near_tie_records_sums_choices_and_prediction_differences():
    train, _, binned, _ = prepared_fixture("weighted_ties")
    gradient = np.array(
        [
            0,
            0.042141210703726076,
            0.07697524756812128,
            0.044312313692044324,
            0,
            2.0064142014852084,
            -2.0121120875434078,
            0.7221702433864421,
        ]
    )
    original = np.column_stack((gradient, train.weight))
    stored = original.astype(np.float32)
    names, roles = ("gradient", "curvature"), ("training", "training")
    cpu_fields = RowFields(train.identity, train.data.identity, names, original, roles)
    cpu_candidates = {
        c.key: c for c in candidates(histogram(binned, cpu_fields, np.arange(len(gradient))))
    }
    keys = ((0, 0, True), (0, 3, False))
    options, _ = enumerate_candidates(train.data.values, stored, range(len(gradient)))
    by_key = {c["key"]: c for c in options}
    cpu_prediction = depthwise(binned, cpu_fields, max_depth=2).predict(train.data)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, train)
        buffer = context.upload(stored)
        fields = ops.fields(data, buffer, names=names, roles=roles)
        batch = ops.candidates(ops.histogram(data, fields, ops.rows(data)))
        scores = ops.newton_scores(batch)
        selected = ops.choose(batch, scores, ops.feasible(batch))
        indices = [next(i for i in range(batch.size) if batch.key(i) == key) for key in keys]
        sums, gains = context.export(batch.values), context.export(scores.values)
        tree = trees.depthwise(ops, data, fields, binning=binned.binning, max_depth=2)
        prediction = context.export(trees.predict(ops, tree, data))
        diagnostic = dict(
            status="known structural parity limitation; diagnostic only",
            keys=keys,
            original_fields=original.tolist(),
            stored_fields=stored.tolist(),
            original_row_sums=[by_key[k]["sums"].tolist() for k in keys],
            original_row_gains=[by_key[k]["gain"] for k in keys],
            cpu_histogram_gains=[score(cpu_candidates[k]) for k in keys],
            device_sums=sums[indices].tolist(),
            device_gains=gains[indices].tolist(),
            device_gain_bits=gains[indices].view(np.uint32).tolist(),
            selected_key=selected.key if selected else None,
            row_partitions=[by_key[k]["rows"] for k in keys],
            device_topology=tree.topology,
            cpu_prediction=cpu_prediction.tolist(),
            device_prediction=prediction.tolist(),
            prediction_difference=(prediction - cpu_prediction).tolist(),
        )
        print("NORMAL_NEAR_TIE_DIAGNOSTIC=" + json.dumps(diagnostic), flush=True)
        # Check measurement validity, not a cross-backend structural winner.
        for i, key in zip(indices, keys, strict=True):
            close(sums[i], by_key[key]["sums"])
            close(gains[i], by_key[key]["gain"])
        assert by_key[keys[0]]["rows"] != by_key[keys[1]]["rows"]
        assert np.all(np.isfinite(prediction)) and selected is not None
