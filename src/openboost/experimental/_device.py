"""Strict CUDA extension boundaries; all sample arrays stay on the current device."""

from dataclasses import replace
from types import MappingProxyType

import numpy as np

from ._builders import BuiltTree, ExtensionSession, snapshot_tree
from ._contracts import ExecutionContext, ObjectiveBridge, exact_keys


def device_vector(value, n, label):
    import cupy as cp

    if not isinstance(value, cp.ndarray):
        raise TypeError(f"{label} must be a CuPy array")
    if (
        value.device.id != cp.cuda.runtime.getDevice()
        or value.dtype != cp.float32
        or value.shape != (n,)
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"{label} must be current-device contiguous float32 ({n},)")
    if not bool(cp.all(cp.isfinite(value))):
        raise ValueError(f"{label} contains non-finite values")
    return value


class DeviceObjectiveBridge(ObjectiveBridge):
    device_capable = True

    @property
    def context(self):
        import cupy as cp

        return ExecutionContext("cuda", cp, self.rng, self.round_idx)

    def step(self, raw, y, sample_weight=None, extra=None):
        import cupy as cp

        self.round_idx += 1
        for k in self.channel_names:
            device_vector(raw[k], len(y), f"raw[{k}]")
        # CuPy does not provide NumPy's read-only views. Isolate untrusted plugin
        # inputs and detect writes; these device copies are explicit overhead.
        originals = list(raw.values()) + [y] + ([] if sample_weight is None else [sample_weight])
        copies = [a.copy() for a in originals]
        borrowed = dict(zip(raw, copies[: len(raw)], strict=True))
        result = self.objective.step(
            MappingProxyType(borrowed),
            copies[len(raw)],
            None if sample_weight is None else copies[-1],
            extra,
            context=self.context,
        )
        if any(not bool(cp.array_equal(a, b)) for a, b in zip(originals, copies, strict=True)):
            raise ValueError("Objective mutated borrowed device inputs")
        exact_keys(result, self.channel_names, "step")
        seen = copies + originals
        validated = {}
        for k in self.channel_names:
            pair = result[k]
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Each channel must return a (grad, hess) tuple")
            for label, a in zip(("grad", "hess"), pair, strict=True):
                device_vector(a, len(y), label)
                if label == "hess" and bool(cp.any(a < 0)):
                    raise ValueError("Effective Hessian must be nonnegative")
                if any(cp.may_share_memory(a, old) for old in seen):
                    raise ValueError("Objective buffers must not alias inputs/statistics")
                seen.append(a)
            validated[k] = pair
        return validated


class DeviceExtensionSession(ExtensionSession):
    """Explicit builder dispatch and independent device traversal of stored trees."""

    def initialize(self, binned, base, y, weights):
        import cupy as cp

        self.binned = replace(binned, data=cp.asarray(binned.data), device="cuda")
        return (
            {k: cp.full(len(y), v, cp.float32) for k, v in base.items()},
            cp.asarray(y),
            None if weights is None else cp.asarray(weights),
        )

    def validate_update(self, raw, n):
        device_vector(raw, n, "updated raw")

    def build(self, binned, grad, hess, channel):
        import cupy as cp

        context = ExecutionContext("cuda", cp, self.bridge.rng, self.bridge.round_idx, channel)
        original = (self.binned.data, grad, hess)
        copies = tuple(a.copy() for a in original)
        borrowed = replace(self.binned, data=copies[0])
        built = self.builder.build(
            borrowed, copies[1], copies[2], config=replace(self.config), context=context
        )
        if any(not bool(cp.array_equal(a, b)) for a, b in zip(original, copies, strict=True)):
            raise ValueError("Builder mutated borrowed device inputs")
        if not isinstance(built, BuiltTree):
            raise TypeError("TreeBuilder.build must return BuiltTree")
        tree = snapshot_tree(built.tree, binned.n_features)
        # Standard numeric traversal validates even a third-party cached update.
        ids = cp.zeros(binned.n_samples, cp.int32)
        rows = cp.arange(binned.n_samples)
        feature, threshold, left, right = [
            cp.asarray(a)
            for a in (tree.features, tree.thresholds, tree.left_children, tree.right_children)
        ]
        if any(
            a is not None and np.any(a) for a in (tree.is_categorical_split, tree.missing_go_left)
        ):
            raise ValueError("Strict CUDA supports numeric nonmissing trees only")
        for _ in range(9):
            active = left[ids] != -1
            nodes = ids[active]
            go_left = self.binned.data[feature[nodes], rows[active]] <= threshold[nodes]
            ids[active] = cp.where(go_left, left[nodes], right[nodes])
        update = cp.asarray(tree.values)[ids]
        if built.train_prediction is not None:
            cached = device_vector(built.train_prediction, binned.n_samples, "cached prediction")
            if not bool(cp.array_equal(cached, update)):
                raise ValueError("Cached training prediction differs from the tree")
        return tree, update
