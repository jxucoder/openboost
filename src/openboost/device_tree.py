"""Composable resident trees; scalar T4 evidence, vector construction pending CUDA."""

from dataclasses import dataclass

import numpy as np

from .binning import Binning
from .device import DeviceData, DeviceFields, DeviceSplit, _atomic, _parameter, _workspace
from .tree import Tree


@dataclass(frozen=True, eq=False)
class DeviceTree:
    """Opaque owned tree, independent of training scratch and prepared data lifetime."""

    binning: Binning
    topology: tuple[tuple[int, int, bool, int, int], ...]
    output_width: int = 1

    def __post_init__(self):
        _check_width(self.output_width)

    @property
    def n_nodes(self):
        return len(self.topology)


def _check_width(width):
    if type(width) is not int or width < 1:
        raise ValueError("positive integer tree output width required")


def _check_binning(data, binning):
    if not isinstance(binning, Binning) or data.binning_identity != binning.identity:
        raise ValueError("prediction/training fitted binning identity differs")


def _cpu_tree(binning, topology, values):
    f, t, missing, left, right = zip(*topology, strict=True)
    return Tree(binning, f, t, np.array(missing, bool), left, right, values)


@_atomic
def assemble(ops, *, binning, topology, values, output_width=1):
    """Pack a validated topology and borrowed resident node values into an owned tree.

    Values is an explicit tuple of live float32 [output_width] buffers, one per
    node including internal nodes. Their lifetime remains with the caller; the
    returned tree owns independent packed values and topology. No row data or
    learner scratch is needed after assembly.
    """
    _check_width(output_width)
    if not isinstance(binning, Binning):
        raise ValueError("fitted Binning required")
    if (
        not isinstance(topology, tuple)
        or not 0 < len(topology) <= np.iinfo(np.int32).max
        or output_width > np.iinfo(np.int32).max
        or any(not isinstance(row, tuple) or len(row) != 5 or
               any(type(v) is not t for v, t in zip(row, (int, int, bool, int, int), strict=True))
               for row in topology)
    ):
        raise ValueError("explicit int32-sized topology with integer keys and boolean routing required")
    if not isinstance(values, tuple) or len(values) != len(topology):
        raise ValueError("one explicit node value buffer per topology node required")
    shape = (len(topology),) if output_width == 1 else (len(topology), output_width)
    _cpu_tree(binning, topology, np.zeros(shape))
    # Check all buffer identities/shapes before any device validation allocation.
    arrays = [ops._float(value, (output_width,)) for value in values]
    context = ops.execution
    with _workspace(ops) as retained:
        for array in arrays:
            ops._validate(array.reshape(1, output_width))
        device_topology = context.upload(np.asarray(topology, np.int32))
        packed = context._empty(shape, np.float32)
        for i, array in enumerate(arrays):
            ops._launch("pack_leaf" if output_width == 1 else "pack_vector_leaf",
                        output_width, array, i, context._array(packed))
        handles = (device_topology, packed)
        result = ops._record(DeviceTree(binning, topology, output_width), handles, handles)
        retained.add(result)
        return result


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
    ordering=None,
    field_leaf=None,
    leaf_fields=None,
    output_width=1,
):
    """Grow via public operations; supplied callbacks replace their respective policy.

    Callbacks receive (ops, candidates), or (ops, histogram) for leaf. They may
    compose resident operations; new callback workspace is released on return.
    Existing caller buffers are borrowed. Every chosen split has nonempty children.
    Topology and control are host metadata; leaf values and predictions stay on GPU.
    Leaf callbacks must return [output_width]. Separate leaf_fields use the same
    original routed rows; callbacks select vector policies explicitly.
    ordering replaces scoring and legality, returning a live split from that
    exact candidate batch or None. field_leaf replaces the histogram leaf and
    receives (ops, leaf_fields, rows), without constructing a leaf histogram.
    Each supplied operation owns its numerical policy parameters.
    """
    ops._get(data, DeviceData)
    ops._get(fields, DeviceFields)
    _check_binning(data, binning)
    if fields.data is not data:
        raise ValueError("fields belong to different prepared data")
    leaf_fields = fields if leaf_fields is None else ops._get(leaf_fields, DeviceFields)
    if leaf_fields.data is not data:
        raise ValueError("leaf fields belong to different prepared data")
    _check_width(output_width)
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("nonnegative integer max_depth required")
    regularization, minimum, penalty = map(_parameter, (reg_lambda, min_child_h, split_penalty))
    for callback in (scoring, legality, leaf, ordering, field_leaf):
        if callback is not None and not callable(callback):
            raise ValueError("callable device operation required")
    if (
        (scoring is not None and (reg_lambda != 1 or split_penalty != 0))
        or (leaf is not None and reg_lambda != 1)
        or (legality is not None and min_child_h != 0)
        or (ordering is not None and (
            scoring is not None or legality is not None
            or reg_lambda != 1 or min_child_h != 0 or split_penalty != 0
        ))
        or (field_leaf is not None and (leaf is not None or reg_lambda != 1))
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
                histogram = None
                if field_leaf is not None:
                    value = field_leaf(ops, leaf_fields, rows)
                else:
                    histogram = ops.histogram(data, leaf_fields, rows)
                    value = (
                        leaf(ops, histogram)
                        if leaf is not None
                        else ops.leaf(histogram, reg_lambda=regularization)
                    )
                ops._validate(ops._float(value, (output_width,)).reshape(1, output_width))
                value = context.copy(value)
                values.append(value)
                node_outputs.add(value)
                if level < max_depth:
                    if histogram is None or leaf_fields is not fields:
                        histogram = ops.histogram(data, fields, rows)
                    batch = ops.candidates(histogram)
                    if ordering is not None:
                        split = ordering(ops, batch)
                        if split is not None:
                            ops._get(split, DeviceSplit)
                            if split.candidates is not batch:
                                raise ValueError("ordering split belongs to different candidate batch")
                    else:
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
                        if any(child.positions.shape[0] == 0 for child in children):
                            raise ValueError("ordering split must have nonempty children")
                        node = [*split.key, len(queue), len(queue) + 1]
                        queue.extend((child, level + 1) for child in children)
                        node_outputs.update(children)
            nodes.append(tuple(node))
            ops.release(rows)
            index += 1
        topology = tuple(nodes)
        shape = (len(nodes),) if output_width == 1 else (len(nodes), output_width)
        _cpu_tree(binning, topology, np.zeros(shape))  # validate metadata, no inference
        device_topology = context.upload(np.asarray(topology, np.int32))
        packed = context._empty(shape, np.float32)
        for i, value in enumerate(values):
            ops._launch(
                "pack_leaf" if output_width == 1 else "pack_vector_leaf",
                output_width,
                context._array(value),
                i,
                context._array(packed),
            )
        handles = (device_topology, packed)
        result = ops._record(DeviceTree(binning, topology, output_width), handles, handles)
        retained.add(result)
        return result


@_atomic
def copy(ops, tree):
    """Independent owned tree snapshot; no aliases to learner workspace."""
    ops._get(tree, DeviceTree)
    handles = tuple(ops.execution.copy(h) for h in ops._records[tree][0])
    return ops._record(DeviceTree(tree.binning, tree.topology, tree.output_width), handles, handles)


@_atomic
def predict(ops, tree, data):
    """New resident predictions [N, L], without base or observation offset."""
    ops._get(tree, DeviceTree)
    ops._get(data, DeviceData)
    _check_binning(data, tree.binning)
    context = ops.execution
    topology, values = (context._array(h) for h in ops._records[tree][0])
    output = context._empty((data.n_rows, tree.output_width), np.float32)
    ops._launch(
        "scalar_tree_predict" if tree.output_width == 1 else "vector_tree_predict",
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
