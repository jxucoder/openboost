"""Explicit compatible tree growth through grouped reductions and stable routing."""

from dataclasses import dataclass, field

import numpy as np

from . import device_group_tree as routing
from . import device_groups as reductions
from . import device_tree as trees
from .binning import Binning
from .device import DeviceData, DeviceFields, _atomic, _parameter, _schema, _workspace


@dataclass(frozen=True)
class TreeJob:
    """Borrowed inputs and independent policy for one depthwise tree.

    Callbacks follow device_tree.depthwise: compose resident operations, borrow
    existing inputs, and do not retain newly allocated scratch outside the call.
    Supplied callbacks own their policy parameters. Leaf fields may differ from
    split fields, but each layout must be compatible across the submitted group.
    """

    run_id: str
    data: DeviceData
    fields: DeviceFields
    binning: Binning
    max_depth: int = 2
    reg_lambda: float = 1.0
    min_child_h: float = 0.0
    split_penalty: float = 0.0
    scoring: object = None
    legality: object = None
    leaf: object = None
    leaf_fields: DeviceFields | None = None
    output_width: int = 1


@dataclass(frozen=True)
class TreeOutcome:
    """Caller-order complete, inactive or failed result; complete trees are owned.

    ValueError/ArithmeticError from one node fail only its run. Allocation,
    driver and unexpected failures raise for the whole call without outputs.
    """

    run_id: str
    status: str
    tree: trees.DeviceTree | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class GrowthPlan:
    """Validated host metadata, not a promise about physical pool capacity."""

    run_ids: tuple[str, ...]
    active: tuple[bool, ...]
    depths: tuple[int, ...]
    output_widths: tuple[int, ...]
    split_names: tuple[str, ...]
    leaf_names: tuple[str, ...]


def _leaves(job):
    return job.fields if job.leaf_fields is None else job.leaf_fields


def growth_plan(jobs, *, active=None):
    """Validate 1–32 jobs sharing features and both field layouts, even if inactive."""
    ids, active = routing._jobs(jobs, active, TreeJob)
    first = None
    for job in jobs:
        identity = routing._features(job.data)
        trees._check_binning(job.data, job.binning)
        trees._check_width(job.output_width)
        if job.output_width > np.iinfo(np.int32).max:
            raise ValueError("int32 output width required")
        if type(job.max_depth) is not int or job.max_depth < 0:
            raise ValueError("nonnegative integer max_depth required")
        for value in (job.reg_lambda, job.min_child_h, job.split_penalty):
            _parameter(value)
        for callback in (job.scoring, job.legality, job.leaf):
            if callback is not None and not callable(callback):
                raise ValueError("callable device operation required")
        if (
            (job.scoring is not None and (job.reg_lambda != 1 or job.split_penalty != 0))
            or (job.leaf is not None and job.reg_lambda != 1)
            or (job.legality is not None and job.min_child_h != 0)
        ):
            raise ValueError("supplied operations own their policy parameters")
        layouts = []
        for fields in (job.fields, _leaves(job)):
            if not isinstance(fields, DeviceFields) or fields.data is not job.data:
                raise ValueError("fields must share their exact data binding")
            layout = _schema(fields.names, fields.roles, len(fields.names))
            if "unweighted" in layout[1]:
                raise ValueError("apply objective training weights before aggregation")
            reductions._buffer(fields.values, (job.data.n_rows, len(fields.names)), 'f4')
            layouts.append(layout)
        compatibility = (identity, tuple(layouts))
        if first is not None and first != compatibility:
            raise ValueError("shared feature handles and compatible split/leaf layouts required")
        first = compatibility
    return GrowthPlan(ids, active, tuple(j.max_depth for j in jobs),
                      tuple(j.output_width for j in jobs), jobs[0].fields.names,
                      _leaves(jobs[0]).names)


@dataclass
class _Growth:
    job: TreeJob
    queue: list
    index: int = 0
    nodes: list = field(default_factory=list)
    values: list = field(default_factory=list)
    outcome: TreeOutcome | None = None

    @property
    def current(self):
        return self.queue[self.index]


def _release(ops, state):
    for rows, _ in state.queue[state.index:]:
        ops.release(rows)
    for value in state.values:
        ops.execution.release(value)
    state.queue.clear()
    state.values.clear()


def _fail(ops, state, error):
    _release(ops, state)
    state.outcome = TreeOutcome(state.job.run_id, 'failed', error_type=type(error).__name__,
                                error_message=str(error))
    reductions._count(ops.execution, 'grouped_growth_failures', 1)


def _histograms(ops, states, *, leaves):
    return reductions.histograms(ops, tuple(
        reductions.HistogramJob(s.job.run_id, s.job.data,
                                _leaves(s.job) if leaves else s.job.fields, s.current[0])
        for s in states))


def _choose(ops, job, histogram):
    # Keep the selected candidate chain live through the grouped partition phase.
    with _workspace(ops) as retained:
        batch = ops.candidates(histogram)
        scores = (job.scoring(ops, batch) if job.scoring is not None else
                  ops.newton_scores(batch, reg_lambda=job.reg_lambda, split_penalty=job.split_penalty))
        mask = (job.legality(ops, batch) if job.legality is not None else
                ops.feasible(batch, min_child_h=job.min_child_h))
        if job.legality is not None:
            mask = ops.mask_and(mask, ops.nonempty(batch))
        split = ops.choose(batch, scores, mask)
        if split is not None:
            retained.update((batch, split))
        return split


def _phase(ops, states):
    """One ready node per run; no scratch escapes except copied leaves and children."""
    context = ops.execution
    reductions._count(context, 'grouped_growth_phases', 1)
    reductions._count(context, 'grouped_growth_node_slots', len(states))
    with _workspace(ops) as retained:
        histograms = _histograms(ops, states, leaves=True)
        split_histograms = {}
        separate = []
        for state, outcome in zip(states, histograms, strict=True):
            if outcome.status == 'failed':
                _fail(ops, state, ValueError(outcome.error_message))
                continue
            job, histogram = state.job, outcome.histogram
            try:
                with _workspace(ops) as copied:
                    value = (job.leaf(ops, histogram) if job.leaf is not None else
                             ops.leaf(histogram, reg_lambda=job.reg_lambda))
                    ops._validate(ops._float(value, (job.output_width,)).reshape(1, job.output_width))
                    value = context.copy(value)
                    copied.add(value)
                state.values.append(value)
            except (ValueError, ArithmeticError) as error:
                _fail(ops, state, error)
                continue
            if state.current[1] < job.max_depth:
                if _leaves(job) is job.fields:
                    split_histograms[job.run_id] = histogram
                else:
                    separate.append(state)
        if separate:
            outcomes = _histograms(ops, separate, leaves=False)
            for state, outcome in zip(separate, outcomes, strict=True):
                if outcome.status == 'failed':
                    _fail(ops, state, ValueError(outcome.error_message))
                else:
                    split_histograms[state.job.run_id] = outcome.histogram
        requests, selected = [], {}
        for state in states:
            if state.outcome is not None or state.job.run_id not in split_histograms:
                continue
            try:
                split = _choose(ops, state.job, split_histograms[state.job.run_id])
                if split is not None:
                    requests.append(routing.PartitionJob(state.job.run_id, state.current[0], split))
                    selected[state.job.run_id] = split
            except (ValueError, ArithmeticError) as error:
                _fail(ops, state, error)
        routed = {o.run_id: o.children for o in routing.partitions(ops, tuple(requests))} if requests else {}
        for state in states:
            if state.outcome is not None:
                continue
            rows, level = state.current
            node = (-1, -1, False, -1, -1)
            if state.job.run_id in routed:
                children = routed[state.job.run_id]
                node = (*selected[state.job.run_id].key, len(state.queue), len(state.queue) + 1)
                state.queue.extend((child, level + 1) for child in children)
                retained.update(children)
            state.nodes.append(node)
            retained.add(state.values[-1])
            ops.release(rows)
            state.index += 1


@_atomic
def depthwise(ops, jobs, *, active=None):
    """Grow compatible independent trees using real grouped reductions/routing.

    Work is synchronous, with one ready node from each remaining run per phase.
    Existing split/leaf policies keep their per-run semantics and original-row
    ordering. Completed trees are independent of all input/scratch lifetimes.
    No fit-level state, RNG, acceptance or stopping is inferred or shared here.
    """
    plan = growth_plan(jobs, active=active)
    for job in jobs:
        ops._get(job.data, DeviceData)
        for fields in (job.fields, _leaves(job)):
            ops._get(fields, DeviceFields)
            ops._float(fields.values, (job.data.n_rows, len(fields.names)))
    if not any(plan.active):
        return tuple(TreeOutcome(j.run_id, 'inactive') for j in jobs)
    reductions._count(ops.execution, 'grouped_growth_calls', 1)
    with _workspace(ops) as retained:
        states = [_Growth(job, [(ops.rows(job.data), 0)]) if active else
                  _Growth(job, [], outcome=TreeOutcome(job.run_id, 'inactive'))
                  for job, active in zip(jobs, plan.active, strict=True)]
        while pending := [s for s in states if s.outcome is None]:
            _phase(ops, pending)
            for state in pending:
                if state.outcome is not None or state.index < len(state.queue):
                    continue
                try:
                    tree = trees.assemble(ops, binning=state.job.binning, topology=tuple(state.nodes),
                                          values=tuple(state.values), output_width=state.job.output_width)
                except (ValueError, ArithmeticError) as error:
                    _fail(ops, state, error)
                    continue
                _release(ops, state)
                retained.add(tree)
                state.outcome = TreeOutcome(state.job.run_id, 'complete', tree)
                reductions._count(ops.execution, 'grouped_growth_successes', 1)
        return tuple(s.outcome for s in states)
