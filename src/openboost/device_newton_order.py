"""Explicit exact dyadic scalar Newton ordering over resident original-row fields.

The caller supplies a prepared candidate batch. Its rounded sums are not used by
this operation; original fields/codes/rows determine integer sums, feasibility
and rational order. Existing candidate construction's finite support checks
still apply. This module does not change tree or recipe defaults.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ._newton_integer_math import COMPARE_LIMBS, FIELD_LIMBS
from .device import DeviceCandidates, DeviceMask, DeviceSplit, _atomic, _parameter, _workspace
from .execution import DeviceBuffer


@dataclass(frozen=True)
class NewtonParameters:
    reg_lambda: float = 1.0
    min_child_h: float = 0.0
    split_penalty: float = 0.0
    min_information: object = None

    def __post_init__(self):
        for name in ('reg_lambda', 'min_child_h', 'split_penalty'):
            object.__setattr__(self, name, float(_parameter(getattr(self, name))))
        info = self.min_information
        if info is None:
            info = {}
        if not isinstance(info, Mapping) or any(not isinstance(n, str) or not n for n in info):
            raise ValueError('named independent minimum mapping required')
        object.__setattr__(self, 'min_information', tuple(sorted((n, float(_parameter(v))) for n, v in info.items())))


def columns(names, roles, parameters):
    """Validate named once-weighted G/H and separately weighted information."""
    if (len(names) != len(roles) or len(set(names)) != len(names) or 'unweighted' in roles
            or any(n not in names for n in ('gradient', 'curvature'))):
        raise ValueError('named once-weighted scalar fields required')
    g, h = names.index('gradient'), names.index('curvature')
    if roles[g] != 'training' or roles[h] != 'training':
        raise ValueError('once-weighted gradient and curvature required')
    indices = []
    for name, _ in parameters.min_information:
        if name not in names or roles[names.index(name)] != 'independent':
            raise ValueError('named independent information minimum required')
        indices.append(names.index(name))
    return g, h, tuple(indices)


@dataclass(frozen=True, eq=False)
class DeviceNewtonOrder:
    candidates: DeviceCandidates
    parameters: NewtonParameters
    sums: DeviceBuffer
    numerator: DeviceBuffer
    denominator: DeviceBuffer
    eligible: DeviceBuffer

    @property
    def data(self):
        return self.candidates.data


@_atomic
def rank(ops, candidates, *, reg_lambda=1.0, min_child_h=0.0, split_penalty=0.0, min_information=None):
    """Own exact integer statistics/order; all work executes on the context GPU.

    Sums retain [candidate, left-positive-G/negative-G/H, right-positive-G/negative-G/H,
    digit] in unsigned base-2^16, in units of 2^-149. Numerator/denominator arrays
    encode the sum of child scores as P/(2*2^149*Q). Eligibility includes exact
    curvature/information minima, physical child counts and strictly positive
    full Newton gain. Only status flags return to the host during construction.
    """
    parameters = NewtonParameters(reg_lambda, min_child_h, split_penalty, min_information)
    batch = ops._batch(candidates)
    fields = batch.histogram.fields
    g, h, information = columns(fields.names, fields.roles, parameters)
    context = ops.execution
    for q in (h, *information):
        ops._validate(context._array(fields.values)[:, q:q+1], nonnegative=True)
    with _workspace(ops) as retained:
        sums = context._empty((batch.size, 6, FIELD_LIMBS), np.uint32)
        numerator = context._empty((batch.size, COMPARE_LIMBS), np.uint32)
        denominator = context._empty((batch.size, COMPARE_LIMBS), np.uint32)
        eligible = context._empty((batch.size,), bool)
        flags = context._empty((batch.size,), np.int32)
        info = context.upload(np.asarray(information, np.int32)) if information else context._empty((0,), np.int32)
        minima = (context.upload(np.asarray([v for _, v in parameters.min_information], np.float32))
                  if information else context._empty((0,), np.float32))
        ops._launch('exact_newton_rank', batch.size, context._array(fields.values),
                    context._array(batch.data.codes), context._array(batch.data.missing),
                    context._array(batch.histogram.rows.positions), context._array(batch.active),
                    max(batch.data.bin_counts), g, h, np.float32(parameters.reg_lambda),
                    np.float32(parameters.min_child_h), np.float32(parameters.split_penalty),
                    context._array(info), context._array(minima), context._array(sums),
                    context._array(numerator), context._array(denominator), context._array(eligible), context._array(flags))
        ops._flags(flags, 'exact Newton integer capacity, field domain or denominator failure')
        handles = (sums, numerator, denominator, eligible)
        result = ops._record(DeviceNewtonOrder(batch, parameters, *handles), handles, handles)
        retained.add(result)
        return result


@_atomic
def choose(ops, ordering, mask=None):
    """Best exact strictly positive gain; lexicographic ties, optional extra mask.

    The optional mask can add caller constraints; it cannot alter the declared
    exact Newton minima. The returned split borrows its original candidate batch.
    Parallel pair reductions carry original indices and exact rational ties;
    only the final index returns to the host. All intermediate storage is scratch.
    """
    ordering = ops._get(ordering, DeviceNewtonOrder)
    batch = ops._batch(ordering.candidates)
    if mask is not None:
        ops._get(mask, DeviceMask)
        if mask.candidates is not batch:
            raise ValueError('extra mask belongs to another exact ordering batch')
    context = ops.execution
    with _workspace(ops) as retained:
        size = max(1, (batch.size+1)//2)
        output = context._empty((size,), np.int32)
        numerator, denominator = context._array(ordering.numerator), context._array(ordering.denominator)
        ops._launch('exact_newton_choose', size, numerator, denominator, context._array(ordering.eligible),
                    context._array(ordering.eligible if mask is None else mask.values), context._array(output))
        while size > 1:
            size = (size+1)//2
            previous, output = output, context._empty((size,), np.int32)
            ops._launch('exact_newton_reduce', size, numerator, denominator,
                        context._array(previous), context._array(output))
        index = int(ops._compact(output)[0])
        if index == -1:
            return None
        if not 0 <= index < batch.size:
            raise ValueError('exact Newton comparison capacity failure')
        result = ops._record(DeviceSplit(batch, index), ())
        retained.add(result)
        return result
