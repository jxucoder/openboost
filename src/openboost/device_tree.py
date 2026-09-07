"""Experimental composable resident scalar trees; bounded T4 correctness verified."""

from dataclasses import dataclass

import numpy as np

from .binning import Binning
from .device import DeviceData, DeviceFields, _atomic, _parameter, _workspace
from .tree import Tree


@dataclass(frozen=True, eq=False)
class DeviceTree:
    """Opaque owned tree, independent of training scratch and prepared data lifetime."""

    binning: Binning
    topology: tuple[tuple[int, int, bool, int, int], ...]

    @property
    def n_nodes(self):
        return len(self.topology)


def _check_binning(data, binning):
    if not isinstance(binning, Binning) or data.binning_identity != binning.identity:
        raise ValueError("prediction/training fitted binning identity differs")


def _cpu_tree(binning, topology, values):
    f, t, missing, left, right = zip(*topology, strict=True)
    return Tree(binning, f, t, np.array(missing, bool), left, right, values)


@_atomic
def depthwise(
    ops,
    data,
    fields,
    *,
    binning,
    max_depth=2,
    reg_lambda=1.0,
    min_child_h=0.0,
    split_penalty=0.0,
    scoring=None,
    legality=None,
    leaf=None,
):
    """Grow via public operations; supplied callbacks replace their respective policy.

    Callbacks receive (ops, candidates), or (ops, histogram) for leaf. They may
    compose resident operations; new callback workspace is released on return.
    Existing caller buffers are borrowed. Every chosen split has nonempty children.
    Topology and control are host metadata; leaf values and predictions stay on GPU.
    """
    ops._get(data, DeviceData)
    ops._get(fields, DeviceFields)
    _check_binning(data, binning)
    if fields.data is not data:
        raise ValueError("fields belong to different prepared data")
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("nonnegative integer max_depth required")
    regularization, minimum, penalty = map(_parameter, (reg_lambda, min_child_h, split_penalty))
    for callback in (scoring, legality, leaf):
        if callback is not None and not callable(callback):
            raise ValueError("callable device operation required")
    if (
        (scoring is not None and (reg_lambda != 1 or split_penalty != 0))
        or (leaf is not None and reg_lambda != 1)
        or (legality is not None and min_child_h != 0)
    ):
        raise ValueError("supplied operations own their policy parameters")
    context = ops.execution
    with _workspace(ops) as retained:
        nodes, values = [], []
        queue = [(ops.rows(data), 0)]
        index = 0
        while index < len(queue):
            rows, level = queue[index]
            node = [-1, -1, False, -1, -1]
            with _workspace(ops) as node_outputs:
                histogram = ops.histogram(data, fields, rows)
                value = (
                    leaf(ops, histogram)
                    if leaf is not None
                    else ops.leaf(histogram, reg_lambda=regularization)
                )
                ops._validate(ops._float(value, (1,)).reshape(1, 1))
                value = context.copy(value)
                values.append(value)
                node_outputs.add(value)
                if level < max_depth:
                    batch = ops.candidates(histogram)
                    scores = (
                        scoring(ops, batch)
                        if scoring is not None
                        else ops.newton_scores(
                            batch, reg_lambda=regularization, split_penalty=penalty
                        )
                    )
                    mask = (
                        legality(ops, batch)
                        if legality is not None
                        else ops.feasible(batch, min_child_h=minimum)
                    )
                    if legality is not None:
                        mask = ops.mask_and(mask, ops.nonempty(batch))
                    split = ops.choose(batch, scores, mask)
                    if split is not None:
                        children = ops.partition(rows, split)
                        node = [*split.key, len(queue), len(queue) + 1]
                        queue.extend((child, level + 1) for child in children)
                        node_outputs.update(children)
            nodes.append(tuple(node))
            ops.release(rows)
            index += 1
        topology = tuple(nodes)
        _cpu_tree(binning, topology, np.zeros(len(nodes)))  # validate metadata, no inference
        device_topology = context.upload(np.asarray(topology, np.int32))
        packed = context._empty((len(nodes),), np.float32)
        for i, value in enumerate(values):
            ops._launch("pack_leaf", 1, context._array(value), i, context._array(packed))
        handles = (device_topology, packed)
        result = ops._record(DeviceTree(binning, topology), handles, handles)
        retained.add(result)
        return result


@_atomic
def copy(ops, tree):
    """Independent owned tree snapshot; no aliases to learner workspace."""
    ops._get(tree, DeviceTree)
    handles = tuple(ops.execution.copy(h) for h in ops._records[tree][0])
    return ops._record(DeviceTree(tree.binning, tree.topology), handles, handles)


@_atomic
def predict(ops, tree, data):
    """New resident scalar predictions [N, 1], without base or observation offset."""
    ops._get(tree, DeviceTree)
    ops._get(data, DeviceData)
    _check_binning(data, tree.binning)
    context = ops.execution
    topology, values = (context._array(h) for h in ops._records[tree][0])
    output = context._empty((data.n_rows, 1), np.float32)
    ops._launch(
        "scalar_tree_predict",
        data.n_rows,
        context._array(data.codes),
        context._array(data.missing),
        topology,
        values,
        context._array(output),
    )
    ops._validate(context._array(output))
    return output


def export(ops, tree):
    """Explicit CPU inference artifact; exports only packed node values."""
    ops._get(tree, DeviceTree)
    values = ops.execution.export(ops._records[tree][0][1])
    return _cpu_tree(tree.binning, tree.topology, values)
