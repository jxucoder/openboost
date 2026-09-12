"""Synchronous compatible scalar squared scheduling through real grouped work."""

from dataclasses import dataclass, fields

from . import device_group_growth as growth
from . import device_group_tree as routing
from . import device_recipes
from .device import DeviceOperations, _workspace
from .device_active import SquaredConfiguration, SquaredPhase
from .device_groups import _count
from .device_inputs import DeviceFeatures, check_pair
from .device_runs import RunOutcome, RunSpec


@dataclass(frozen=True)
class SchedulePlan:
    """Host-only grouping and policy snapshot; live registration is checked later."""

    run_ids: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    configurations: tuple[SquaredConfiguration, ...]


def schedule_plan(specs, *, group_size=32):
    """Validate 1–32 independent squared runs borrowing the same feature pair.

    Exact shared records are explicit; equivalent newly prepared encodings are
    not silently combined. Targets, weights, offsets, policies and identities
    remain per run. Unsupported recipes, options and groups reject before owned
    device allocation. Host planning does not establish live handles or capacity.
    """
    specs = tuple(specs)
    if type(group_size) is not int or not 1 <= group_size <= 32:
        raise ValueError('integer group_size in 1..32 required')
    if not 1 <= len(specs) <= 32 or any(not isinstance(s, RunSpec) for s in specs):
        raise ValueError('1..32 explicit RunSpec records required')
    ids = tuple(s.run_id for s in specs)
    if len(set(ids)) != len(ids):
        raise ValueError('unique run IDs required before scheduling')
    allowed = {f.name for f in fields(SquaredConfiguration)}
    configurations = []
    for spec in specs:
        if spec.recipe is not device_recipes.squared or set(spec.options) - allowed:
            raise ValueError('default scalar squared recipe and supported phase options required')
        check_pair(spec.prepared, spec.train, spec.validation, binning=spec.binning)
        if any(a is not b for a, b in zip(spec.prepared, specs[0].prepared, strict=True)):
            raise ValueError('all runs require the same explicit shared feature records')
        configurations.append(SquaredConfiguration(**spec.options))
    return SchedulePlan(ids, tuple(ids[i:i+group_size] for i in range(0, len(ids), group_size)),
                        tuple(configurations))


def run_many(ops, specs, *, group_size=32):
    """Return caller-order detached outcomes from compatible grouped squared fits.

    Each active run owns its accepted state, trial history, stop policy and RNG.
    One ready tree per run enters grouped growth, followed by grouped training and
    validation predictions. Predictions enter the phase without another traversal.
    Terminal/failed runs release their owned state promptly. Borrowed feature
    records and other caller storage survive the call; physical pool caps apply.

    Initialization, request, advance and export exceptions fail their individual
    run. Declared tree/prediction slot failures do likewise. An unexpected shared
    operation exception fails every still-active member of that submitted group;
    other groups and already completed outcomes remain available. Interrupts
    propagate after cleanup. This is cooperative ownership, not process isolation
    against arbitrary mutations of the execution context.

    Only the exact default scalar squared recipe is scheduled here. Custom tree
    requests/external schedules use SquaredPhase directly. No CPU fallback, other
    objective family, general distributed execution or speed claim is implied.
    """
    specs = tuple(specs)
    plan = schedule_plan(specs, group_size=group_size)
    if not isinstance(ops, DeviceOperations):
        raise ValueError('DeviceOperations required; CPU fallback is unavailable')
    ops.execution._check()
    for features in specs[0].prepared:
        ops._get(features, DeviceFeatures)
    phases, outcomes = {}, {}
    context = ops.execution
    _count(context, 'grouped_run_calls', 1)

    def complete(name):
        result = phases[name].result()
        phases.pop(name).close()
        outcomes[name] = RunOutcome(name, result)
        _count(context, 'grouped_run_completions', 1)

    def fail(name, error_type, message):
        phase = phases.pop(name, None)
        if phase is not None:
            phase.close()
        outcomes[name] = RunOutcome(name, None, error_type, message)
        _count(context, 'grouped_run_failures', 1)

    # All work is synchronous and owned by this call. No scope is suspended
    # across an external run's allocations; detached results need no retention.
    with _workspace(ops):
        try:
            for spec, configuration in zip(specs, plan.configurations, strict=True):
                try:
                    phase = SquaredPhase(ops, spec.train, spec.validation, run_id=spec.run_id,
                                         seed=spec.seed, configuration=configuration,
                                         binning=spec.binning, prepared=spec.prepared)
                    phases[spec.run_id] = phase
                    if not phase.active:
                        complete(spec.run_id)
                except Exception as error:
                    fail(spec.run_id, type(error).__name__, str(error))
            while phases:
                for group in plan.groups:
                    selected = [name for name in group if name in phases]
                    if not selected:
                        continue
                    owned, requests = [], []
                    try:
                        for name in selected:
                            try:
                                job = phases[name].request_tree()
                                requests.append(job)
                                owned.append(job.fields)
                            except Exception as error:
                                fail(name, type(error).__name__, str(error))
                        if not requests:
                            continue
                        _count(context, 'grouped_run_groups', 1)
                        _count(context, 'grouped_run_round_slots', len(requests))
                        grown = growth.depthwise(ops, tuple(requests))
                        trees = {}
                        for outcome in grown:
                            if outcome.status == 'complete':
                                trees[outcome.run_id] = outcome.tree
                                owned.append(outcome.tree)
                            else:
                                fail(outcome.run_id, outcome.error_type, outcome.error_message)
                        if not trees:
                            continue
                        predicted = []
                        live = list(trees)
                        for attribute in ('data', 'validation_data'):
                            batch = tuple(routing.PredictionJob(name, trees[name], getattr(phases[name].run, attribute)) for name in live)
                            if not batch:
                                break
                            successful = {}
                            for outcome in routing.predictions(ops, batch):
                                if outcome.status == 'complete':
                                    successful[outcome.run_id] = outcome.prediction
                                    owned.append(outcome.prediction)
                                else:
                                    fail(outcome.run_id, outcome.error_type, outcome.error_message)
                            predicted.append(successful)
                            live = list(successful)
                        for name in live:
                            try:
                                phase = phases[name]
                                phase.advance(predicted[0][name], predicted[1][name])
                                if not phase.active:
                                    complete(name)
                            except Exception as error:
                                fail(name, type(error).__name__, str(error))
                    except Exception as error:
                        for name in selected:
                            if name in phases:
                                fail(name, type(error).__name__, str(error))
                    finally:
                        for record in reversed(owned):
                            ops.release(record)
            return tuple(outcomes[name] for name in plan.run_ids)
        finally:
            for phase in phases.values():
                phase.close()
