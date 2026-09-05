"""Small numeric level-wise builder composed from verified device primitives."""

import numpy as np

from .._array import BinnedArray
from .._core._batch_leaf import NewtonLeafRule, leaf_values
from .._core._batch_primitives import build_histograms
from .._core._batch_split import _array, _namespace, find_splits, partition
from ._builders import BuiltTree, TreeStructure, snapshot_tree
from ._contracts import ExecutionContext


class LevelWiseBuilder:
    """Numeric L2 full-sample builder, depth 0..8, NumPy or CuPy/default stream.

    CPU Booster can select this explicitly. Direct CUDA build returns host tree
    arrays plus an owned CuPy training prediction; GPU Booster dispatch is a
    separate integration gate. The budget covers one histogram batch, not the
    inputs, prediction cache, compact tree arrays or transient validation masks.
    """

    supported_devices = frozenset({"cpu", "cuda"})

    def __init__(self, *, leaf_rule=None, memory_budget_bytes=256 * 1024**2):
        self.leaf_rule = NewtonLeafRule() if leaf_rule is None else leaf_rule
        self.memory_budget_bytes = memory_budget_bytes

    def build(self, binned, grad, hess, *, config, context):
        from ._booster import validate_config

        validate_config(config)
        if config.reg_alpha != 0 or config.subsample != 1 or config.colsample_bytree != 1:
            raise ValueError("LevelWiseBuilder supports L2 and full row/feature sampling only")
        if not isinstance(binned, BinnedArray):
            raise TypeError("Expected BinnedArray")
        xp = _namespace(binned.data)
        device = "cpu" if xp is np else "cuda"
        if (
            not isinstance(context, ExecutionContext)
            or context.device != device
            or context.xp is not xp
            or binned.device != device
        ):
            raise ValueError("Binner, context and arrays must use the same device")
        if xp is not np and xp.cuda.get_current_stream().ptr != 0:
            raise ValueError("LevelWiseBuilder currently requires the default CUDA stream")
        if (
            not 1 <= binned.n_features <= np.iinfo(np.int32).max
            or not 1 <= binned.n_samples <= np.iinfo(np.int32).max
        ):
            raise ValueError("Require nonempty binned features/samples")
        _array(binned.data, xp, np.uint8, (binned.n_features, binned.n_samples))
        _array(grad, xp, np.float32, (binned.n_samples,))
        _array(hess, xp, np.float32, (binned.n_samples,))
        for name in ("has_missing", "is_categorical"):
            flags = getattr(binned, name)
            if (
                not isinstance(flags, np.ndarray)
                or flags.dtype != np.bool_
                or flags.shape not in ((0,), (binned.n_features,))
            ):
                raise ValueError("Binner flags must be host bool feature metadata")
            if flags.any():
                raise ValueError("LevelWiseBuilder rejects missing/categorical features")
        if any(m is not None for m in binned.category_maps) or bool(xp.any(binned.data == 255)):
            raise ValueError("LevelWiseBuilder rejects missing/categorical features")
        if (
            not isinstance(getattr(self.leaf_rule, "supported_devices", None), frozenset)
            or device not in self.leaf_rule.supported_devices
            or not callable(getattr(self.leaf_rule, "values", None))
        ):
            raise ValueError("Leaf rule does not support the requested device")
        budget = self.memory_budget_bytes
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("Histogram budget must be a nonnegative integer")
        slots = 2 ** (config.max_depth + 1) - 1
        required = slots * (binned.n_features * 256 * 8 + 5) if config.max_depth else 0
        if required > budget:
            raise MemoryError(f"Histogram requires {required} bytes; budget is {budget}")
        ids = xp.zeros(binned.n_samples, xp.int32)
        frontier = xp.zeros(slots, xp.bool_)
        frontier[0] = True
        leaves = frontier.copy()
        feature, threshold, left, right = (xp.full(slots, -1, xp.int32) for _ in range(4))
        node = xp.arange(slots, dtype=xp.int32)
        parent = xp.maximum((node - 1) // 2, 0)
        nonroot = node != 0
        for _ in range(config.max_depth):
            hist = build_histograms(
                binned.data, grad, hess, ids, frontier, memory_budget_bytes=budget
            )
            splits = find_splits(
                hist,
                reg_lambda=config.reg_lambda,
                min_child_weight=config.min_child_weight,
                min_gain=config.min_gain,
            )
            valid = splits.valid
            # Keep fixed-size arrays instead of compacting boolean selections.
            # Every nonroot slot is active iff its fixed parent split this round.
            feature = xp.where(valid, splits.feature, feature)
            threshold = xp.where(valid, splits.threshold, threshold)
            left = xp.where(valid, splits.left_child, left)
            right = xp.where(valid, splits.right_child, right)
            ids = partition(binned.data, ids, splits)
            frontier = valid[parent] & nonroot
            leaves = (leaves & ~valid) | frontier
            # Do not retain the previous histogram while allocating the next.
            del hist, splits, valid
        values = leaf_values(
            grad, hess, ids, leaves, leaf_rule=self.leaf_rule, config=config, context=context
        )
        prediction = values[ids]  # advanced indexing owns the sample-sized result
        compact = (feature, threshold, left, right, values)
        host = [a.copy() if xp is np else xp.asnumpy(a) for a in compact]
        tree = TreeStructure(
            *host, n_nodes=slots, depth=config.max_depth, n_features=binned.n_features
        )
        return BuiltTree(snapshot_tree(tree, binned.n_features), prediction)
