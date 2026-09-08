"""Experimental D2 CUDA consumer of public operations; hardware check pending."""

import numpy as np

from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_tree import depthwise


class DeviceCohortLearner:
    """Upload independent information once, then constrain every supplied learner.

    Construct before calling a device recipe. Each child needs one unit of each
    cohort, regardless of objective weight. The caller closes this object after
    training; its buffers have a separate lifetime from any particular run.
    """

    def __init__(self, ops, problem, information, *, binning, max_depth=2, reg_lambda=1.0):
        values = np.asarray(information, dtype=float)
        if (
            values.ndim != 2
            or values.shape[0] != len(problem.target)
            or values.shape[1] == 0
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values > np.finfo(np.float32).max)
        ):
            raise ValueError("aligned finite nonnegative float32 cohort information required")
        if not isinstance(ops, DeviceOperations):
            raise ValueError("DeviceOperations required")
        if not isinstance(binning, Binning) or binning.feature_names != problem.data.feature_names:
            raise ValueError("matching fitted binning required")
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("nonnegative integer max_depth required")
        if (
            isinstance(reg_lambda, (bool, str))
            or not np.isscalar(reg_lambda)
            or not np.isfinite(reg_lambda)
            or not 0 <= reg_lambda <= np.finfo(np.float32).max
        ):
            raise ValueError("finite nonnegative float32 regularization required")
        self.ops, self.problem_identity = ops, problem.identity
        self.binning, self.max_depth, self.reg_lambda = binning, max_depth, reg_lambda
        self.names = tuple(f"cohort:{i}" for i in range(values.shape[1]))
        self.columns, self.closed = [], False
        try:
            for column in values.T:
                self.columns.append(ops.execution.upload(column.astype(np.float32)))
        except Exception:
            self.close()
            raise

    def legal(self, ops, candidates):
        mask = ops.feasible(candidates)
        for name in self.names:
            mask = ops.mask_and(mask, ops.child_minimum(candidates, name, 1))
        return mask

    def __call__(self, ops, data, fields):
        if self.closed:
            raise ValueError("cohort learner is closed")
        if ops is not self.ops or data.problem_identity != self.problem_identity:
            raise ValueError("cohort information belongs to different operations or problem")
        if fields.data is not data:
            raise ValueError("fields belong to different prepared data")
        current = fields
        try:
            for name, column in zip(self.names, self.columns, strict=True):
                augmented = ops.add_independent(current, name, column, nonnegative=True)
                if current is not fields:
                    ops.release(current)
                current = augmented
            return depthwise(
                ops,
                data,
                current,
                binning=self.binning,
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                legality=self.legal,
            )
        finally:
            if current is not fields:
                ops.release(current)

    def close(self):
        """Release this learner's information; no run or context is closed."""
        if not self.closed:
            for column in self.columns:
                self.ops.execution.release(column)
            self.columns.clear()
            self.closed = True
