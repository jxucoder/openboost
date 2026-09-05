"""CPU tree builder and explicit update schedule contracts."""

from dataclasses import dataclass, replace

import numpy as np

from .._core._growth import TreeStructure
from .._core._tree import fit_tree
from ._contracts import ExecutionContext, exact_keys, readonly, vector


@dataclass(frozen=True)
class BuiltTree:
    tree: TreeStructure
    train_prediction: object | None = None


class ConstantSchedule:
    def coefficients(self, round_idx, channel_names, base_learning_rate):
        return dict.fromkeys(channel_names, base_learning_rate)


class CPUHistogramBuilder:
    supported_devices = frozenset({'cpu'})

    def build(self, binned, grad, hess, *, config, context):
        if config.reg_lambda == 0 and float(np.sum(hess, dtype=np.float64)) == 0:
            if float(np.sum(grad, dtype=np.float64)) != 0:
                raise ValueError('Zero curvature denominator with nonzero gradient')
            tree = TreeStructure(
                features=np.array([-1], dtype=np.int32), thresholds=np.array([-1], dtype=np.int32),
                left_children=np.array([-1], dtype=np.int32), right_children=np.array([-1], dtype=np.int32),
                values=np.zeros(1, dtype=np.float32), n_nodes=1, depth=0, n_features=binned.n_features)
        else:
            tree = fit_tree(binned, grad, hess, max_depth=config.max_depth,
                            min_child_weight=config.min_child_weight, reg_lambda=config.reg_lambda,
                            reg_alpha=config.reg_alpha, min_gain=config.min_gain,
                            subsample=config.subsample, colsample_bytree=config.colsample_bytree,
                            rng=context.rng)
        return BuiltTree(tree)


def snapshot_tree(tree, n_features):
    # Only package-owned standard scalar trees enter storage. Copy O(nodes),
    # not O(samples), so a builder can safely reuse its compact scratch buffers.
    if type(tree) is not TreeStructure or tree.is_symmetric:
        raise TypeError('P3 builders must return a standard scalar TreeStructure')
    n = tree.n_nodes
    if not isinstance(n, int) or not 1 <= n <= 511 or tree.n_features != n_features:
        raise ValueError('Invalid tree size or feature count')
    for name in ('features', 'thresholds', 'left_children', 'right_children'):
        a = getattr(tree, name)
        if not isinstance(a, np.ndarray) or a.dtype != np.int32 or a.shape != (n,):
            raise ValueError(f'Invalid tree {name}')
    vector(tree.values, n, 'tree.values')
    for name, dtype in (('missing_go_left', np.bool_), ('is_categorical_split', np.bool_), ('cat_bitsets', np.uint64)):
        a = getattr(tree, name)
        if a is not None and (not isinstance(a, np.ndarray) or a.shape != (n,) or a.dtype != dtype):
            raise ValueError(f'Invalid tree {name}')
    seen = set()
    def visit(i, depth):
        if i < 0 or i >= n or i in seen or depth > 8:
            raise ValueError('Invalid/cyclic tree routing')
        seen.add(i)
        left, right = tree.left_children[i], tree.right_children[i]
        if left == -1:
            if right != -1:
                raise ValueError('Invalid leaf children')
            return
        if not 0 <= tree.features[i] < n_features or not 0 <= tree.thresholds[i] <= 254:
            raise ValueError('Invalid split feature or threshold')
        visit(int(left), depth+1)
        visit(int(right), depth+1)
    visit(0, 0)
    # Reuse the serializer's complete scalar tree state, including missing/cat.
    from .._persistence import _dict_to_tree, _tree_to_dict
    state = _tree_to_dict(tree)
    return _dict_to_tree({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()})


class ExtensionSession:
    def __init__(self, bridge, builder, schedule, config):
        self.bridge, self.builder, self.schedule, self.config = bridge, builder, schedule, config

    def coefficients(self, round_idx):
        out = self.schedule.coefficients(round_idx, self.bridge.channel_names, self.config.learning_rate)
        exact_keys(out, self.bridge.channel_names, 'coefficients')
        result = {}
        for k, v in out.items():
            if not np.isscalar(v) or not np.isfinite(v) or v < 0 or v > np.finfo(np.float32).max:
                raise ValueError('Schedule coefficients must be finite nonnegative scalars')
            result[k] = float(v)
        return result

    def build(self, binned, grad, hess, channel):
        context = ExecutionContext('cpu', np, self.bridge.rng, self.bridge.round_idx, channel)
        borrowed = replace(binned, data=readonly(binned.data))
        built = self.builder.build(borrowed, readonly(grad), readonly(hess), config=replace(self.config), context=context)
        if not isinstance(built, BuiltTree):
            raise TypeError('TreeBuilder.build must return BuiltTree')
        tree = snapshot_tree(built.tree, binned.n_features)
        update = vector(tree(binned), binned.n_samples, 'tree prediction')
        if built.train_prediction is not None:
            cached = vector(built.train_prediction, binned.n_samples, 'cached prediction')
            if not np.array_equal(cached, update):
                raise ValueError('Cached training prediction differs from the tree')
            update = cached
        return tree, update
