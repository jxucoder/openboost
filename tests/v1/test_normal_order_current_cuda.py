"""Current CUDA split order versus exact stored fields; measurements precede assertions."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from openboost import device_tree as trees
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.normal_precision import ATOL, RTOL
from .reference.normal_split_order import candidates, exact_tree, predict, winner
from .test_normal_order_current_reference import prepared, topology

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("case", [
    "captured", "p24-negative", "p24-zero", "p24-positive",
    "p54-negative", "p54-zero", "p54-positive",
    "p100-negative", "p100-zero", "p100-positive",
    "p149-negative", "p149-zero", "p149-positive",
])
def test_current_exact_winner_topology_leaves_and_all_rows(case, tmp_path):
    problem, binned, host_fields, x, exact = prepared(case)
    options = candidates(x, exact, range(8))
    expected = winner(options)
    wanted = exact_tree(x, exact, depth=2)
    stored = host_fields.values.astype(np.float32)
    root = Path(os.environ.get("OPENBOOST_NORMAL_ARTIFACTS", tmp_path)) / "current-normal-order"
    root.mkdir(parents=True, exist_ok=True)
    record = dict(case=case, stage="inputs", x=x, row_ids=problem.data.row_ids.tolist(),
                  stored_fields=stored.tolist(), stored_field_bits=stored.view(np.uint32).tolist(),
                  exact_candidates=[dict(key=c["key"], rows=c["rows"], legal=c["legal"],
                                         sums=[[str(v) for v in s] for s in c["sums"]],
                                         parent=[str(v) for v in c["parent"]], gain=str(c["gain"])) for c in options],
                  expected_key=expected["key"], expected_topology=topology(wanted),
                  expected_values=[str(n["value"]) for n in wanted],
                  expected_prediction=[str(v) for v in predict(wanted, x)],
                  tolerance=dict(rtol=RTOL, atol=ATOL), expected_node_rows=[n["rows"] for n in wanted])

    def save():
        (root / (case + ".json")).write_text(json.dumps(record, indent=2) + "\n")

    save()
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        buffer = context.upload(stored)
        fields = ops.fields(data, buffer, names=host_fields.names, roles=host_fields.roles)
        batch = ops.candidates(ops.histogram(data, fields, ops.rows(data)))
        scores = ops.newton_scores(batch)
        mask = ops.feasible(batch)
        selected = ops.choose(batch, scores, mask)
        gains = context.export(scores.values)
        record.update(stage="root", device_keys=[batch.key(i) for i in range(batch.size)],
                      device_sums=context.export(batch.values).tolist(), device_gains=gains.tolist(),
                      device_gain_bits=gains.view(np.uint32).tolist(),
                      selected_key=None if selected is None else selected.key)
        save()
        tree = trees.depthwise(ops, data, fields, binning=binned.binning, max_depth=2)
        exported = trees.export(ops, tree)
        prediction = context.export(trees.predict(ops, tree, data))
        record.update(stage="complete", device_topology=tree.topology,
                      device_tree=exported.record(), device_prediction=prediction.tolist(),
                      metrics=dict(context.metrics))
        save()
        # A mismatch fails the raw verdict, but cannot erase any completed measurement.
        assert selected is not None and selected.key == expected["key"]
        assert list(tree.topology) == topology(wanted)
        np.testing.assert_allclose(exported.value[:, 0], [float(n["value"]) for n in wanted], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(prediction.reshape(-1), [float(v) for v in predict(wanted, x)], rtol=RTOL, atol=ATOL)
