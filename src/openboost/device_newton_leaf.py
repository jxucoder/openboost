"""Explicit correctly rounded scalar Newton leaves over resident original rows."""

import numpy as np

from .device import DeviceFields, DeviceRows, _atomic, _parameter, _workspace
from .device_newton_order import NewtonParameters, columns


@_atomic
def leaf(ops, fields, rows, *, reg_lambda=1.0):
    """Own one finite float32 -sum(G)/(sum(H)+lambda), rounded once to even.

    Original stored fields and exact routed rows are borrowed. Integer sums may
    exceed float32 range; no floating histogram is constructed. Exact zero is
    positive zero; rounding a nonzero negative ratio to zero preserves its sign.
    Nonnegative curvature, positive exact denominator and finite rounded output
    are required. Only numerical status returns to the host.
    """
    regularization = _parameter(reg_lambda)
    fields = ops._get(fields, DeviceFields)
    rows = ops._get(rows, DeviceRows)
    if rows.data is not fields.data:
        raise ValueError('exact leaf fields and rows must share their prepared data binding')
    g, h, _ = columns(fields.names, fields.roles, NewtonParameters())
    context = ops.execution
    ops._validate(context._array(fields.values)[:, h:h+1], nonnegative=True)
    with _workspace(ops) as retained:
        output = context._empty((1,), np.float32)
        status = context._empty((1,), np.int32)
        ops._launch('exact_newton_leaf', 1, context._array(fields.values), context._array(rows.positions),
                    g, h, regularization, context._array(output), context._array(status))
        ops._flags(status, 'exact Newton leaf has invalid field, denominator, integer capacity or nonfinite rounded output')
        retained.add(output)
        return output
