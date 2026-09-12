# Experimental grouped device histograms

`openboost.device_groups.histograms` computes several independent histograms in
two grouped reduction launches plus one grouped finite-value check. The
[checkpoint](checkpoint.md) distinguishes historical component validation from
the candidate checks still required. This operation alone does not provide a
batched trainer or establish a speed improvement.

Each `HistogramJob(run_id, data, fields, rows)` names one reduction. Supply an
explicit tuple of 1–32 jobs with distinct stable IDs. Every job must borrow the
same resident feature codes/missingness and fitted cuts through
`device_inputs.prepare` and `bind`. Weights, fields and original row selections
belong to their individual bindings. Matching feature values in separately
allocated buffers do not satisfy the sharing contract.

Field names, order and weight roles must match across the group. Apply training
weights before aggregation; independent information columns retain their values.
Group width M is independent of field width Q. Different targets, offsets or
weights need not imply a different group if their resulting fields share the
declared layout and features.

The following helper accepts already prepared public records and requires real
CUDA. Each successful output works with the existing candidate, feasibility,
split, partition and scalar/vector leaf operations:

```python
from openboost.device_groups import HistogramJob, histogram_plan, histograms

def aggregate_group(ops, bindings, fields, rows, run_ids, active):
    jobs = tuple(
        HistogramJob(run_id, data, field, selected)
        for run_id, data, field, selected in zip(
            run_ids, bindings, fields, rows, strict=True
        )
    )
    plan = histogram_plan(jobs, active=active)
    outcomes = histograms(ops, jobs, active=active)
    return plan, outcomes
```

`active` is an aligned tuple of booleans. Inactive jobs still undergo registration
and compatibility checks, but do not pack or reduce their fields/rows. An all
inactive group allocates nothing and launches no kernel. Outcomes preserve caller
order and contain `run_id`, `status`, `histogram`, `error_type` and `error_message`:

- `complete`: an independently owned ordinary `DeviceHistogram`. Release it with
  `ops.release(outcome.histogram)` when finished. Input records remain caller-owned.
- `inactive`: no histogram or error.
- `failed`: finite input fields produced a nonfinite accumulated sum or row total.
  Only that slot fails; other complete histograms remain available.

Allocation or driver errors raise for the whole operation and discard new
allocations. Inputs remain valid for a later call. This cooperative ownership
contract does not isolate hostile callbacks or recover a failed CUDA context.

`histogram_plan` checks host metadata and reports padded shape, row counts and a
conservative additional logical byte bound. Actual execution checks live records
and applies the context memory cap. Physical pool alignment/cache effects can
still reject a group below the logical bound.

This initial layout copies fields/rows into packed buffers and copies successful
outputs into separate allocations. Counters distinguish `grouped_pack_bytes`,
`grouped_unpack_bytes`, metadata uploads and validation exports. Exactly two
`grouped_histogram_kernel_launches` and one `grouped_validation_kernel_launches`
occur for a completed nonempty active group, regardless of M. Only M int32 finite
flags return to the host inside the operation. These are work/accounting
observations, not a full-fit cost verdict. Counts describe attempted work and are
not rolled back when an allocation later fails.

## Grouped routing and prediction: local construction

`openboost.device_group_tree` adds `PartitionJob(run_id, rows, split)` and
`PredictionJob(run_id, tree, data)` with corresponding `partition_plan` /
`partitions` and `prediction_plan` / `predictions` operations. These operations
have their own historical routing/prediction checks, distinct from histogram
checks; see the [evidence boundary](checkpoint.md#historical-observations-and-available-evidence).

Both operations require an explicit tuple of 1–32 distinct IDs, common actual
feature handles and preparation, and an optional aligned tuple of boolean
`active` flags. Inactive slots still require live compatible records; an all
inactive group allocates nothing. Field layouts need not match for routing.
Each split must belong to its exact original rows and live candidate/histogram
chain. Each prediction tree must have the matching fitted cuts; trees may have
different node counts, but their output width L must match. L is separate from M.

`partitions` uses one grouped stable-routing kernel and exports only two int32
child sizes per job. Complete outcomes contain two independently owned ordinary
`DeviceRows`; inactive outcomes contain no children. Release each child through
`ops.release`. Original row order is preserved within both children.

`predictions` uses one grouped traversal plus one grouped finite check and exports
only M int32 flags. Complete outcomes contain a registered `DevicePrediction`
with `tree`, `data` and its owned `[N,L]` `values` buffer. It binds the computed
snapshot to the exact inputs; constructing an identical dataclass does not
register it. Release the record with `ops.release`, which releases only its values.
The source tree/data remain caller-owned. The data must remain live for registry
operations; releasing the tree does not release the snapshot. A later transaction
consumer requiring that tree must validate its live registration separately.
Inactive outcomes have no prediction; nonfinite predictions fail only their slot.

Plans report conservative additional logical bytes, including packing and detached
outputs. Physical pool rounding may still reject a group. Allocation/driver errors
raise, discard all new records/buffers and preserve inputs. Counters prefixed
`grouped_partition_` and `grouped_prediction_` distinguish packing, output copies,
metadata uploads, compact exports and kernel launches; attempted work remains
counted after a failure. Feature, row, topology, leaf and prediction arrays do not
return to the host inside these operations.

Historical routing/prediction checks include failure status and fresh exported
tree inference. The nonfinite status control injects an invalid output; finite
leaf copies themselves perform no arithmetic that could overflow. Full candidate
validation and real-data quality/cost acceptance remain open.

### Consuming predictions in a proposal

`openboost.device_runtime.PredictedTerm(term, train, validation)` binds a
`DeviceTerm` to its registered training and validation `DevicePrediction` records.
`run.propose_predicted(state, (item, ...), coefficient=...)` uses their resident
values directly, applying mappings in tuple order without another tree traversal.
The run checks exact live source trees, exact per-run training/validation data,
registered prediction ownership, mapping widths and coefficient before allocating.
Shared feature handles alone do not make another run's binding interchangeable.

Trees have no public value mutation operation. An explicit `device_tree.copy`
can freeze the learner output before grouping. The proposal makes its own tree
and raw copies, so its lifetime is independent of those borrowed inputs:

```python
from openboost.device_runtime import DeviceTerm, PredictedTerm

item = PredictedTerm(DeviceTerm(snapshot, [[1]]), train_prediction, validation_prediction)
proposal = run.propose_predicted(state, (item,), coefficient=0.5)
updated = run.resolve(state, proposal, accept=True)
run.release(proposal)
```

Callers own and release source trees and prediction records. Keep their source
data live until releasing predictions. The same predictions can be reused for
another coefficient; failures preserve inputs and parent state and publish no
proposal identity. Ordinary `resolve` retains its acceptance and best-model rules;
patience remains a separate caller-owned `StopState`. Each operation workspace
ends before another run acts. Do not suspend a whole-fit workspace across runs.

The historical prediction-to-proposal controls cover ordered terms, ownership
faults, allocation failures, finite mapping overflow, rejection and same-ID retries.
Their raw archives are outside this PR; broader consumer shapes, full train-many
acceptance and real-data quality/cost evaluation remain required.

## Grouped tree growth

`openboost.device_group_growth.depthwise(ops, jobs, active=...)` accepts an explicit
tuple of 1–32 `TreeJob` records. Each job binds its own data, split fields, optional
leaf fields, fitted binning, depth, output width and optional scoring/legality/leaf
callbacks. Jobs share live feature handles and compatible split/leaf field layouts.
Widths and depth budgets remain per job. Inactive jobs still validate metadata and
live inputs; an all-inactive group allocates nothing.

Each phase submits one ready node per remaining run to the public grouped histogram
and partition operations. Candidate decisions and leaf policies remain independent.
`ValueError` and `ArithmeticError` fail one job and release its queued rows/values;
allocation, driver and unexpected exceptions abort the whole operation atomically.
Callbacks borrow inputs and may not mutate or release them or retain new scratch.
This is a synchronous tree operation; it does not suspend a shared workspace across
independent recipe calls. It does not yet implement a complete train-many recipe.

`device_tree.assemble` packs validated topology and one borrowed resident leaf-value
buffer per node into an independently owned tree. The caller retains its inputs;
the completed tree can outlive them. Release each successful outcome's tree through
`ops.release`. The execution context's physical pool cap remains authoritative.

## Active squared recipe phases

`device_active.SquaredPhase` owns one run's accepted state, best model, trial
history and stopping policy. Supply an immutable `SquaredConfiguration` and the
usual train/validation problems, identity, seed and optional fitted feature pair.

- `request_tree()` returns a `TreeJob` with freshly owned fields; release those
  fields through `ops.release` after tree growth. Repeated requests consume no round.
- `advance(train_prediction, validation_prediction)` borrows an exact registered
  scalar prediction pair, performs configured fixed/backtracking trials and observes
  stopping once. The caller retains its tree and prediction ownership.
- `result()` requires terminal stopping and exports detached final/best models.
- `close()` releases the phase's run. A context manager also closes it; caller-owned
  field requests, trees, predictions and shared features retain their own lifetimes.

No workspace remains open across these calls. This component lets callers compose
grouped operations explicitly and follows the scalar squared recipe's reported
comparison. The compatible scheduling wrapper below composes these phases.

## Compatible squared scheduling

`device_group_runs.run_many(ops, specs, group_size=32)` accepts 1–32 `RunSpec`
records using the exact `device_recipes.squared` recipe and supported scalar
`SquaredConfiguration` options. Each spec explicitly borrows the same registered
training/validation feature records. Targets, weights, offsets, identity, seed,
accepted state and stopping remain independent. `schedule_plan` validates host
compatibility and returns immutable caller-order groups before device allocation.

The wrapper owns active phases, groups tree growth and prediction, advances each
run and returns detached `RunOutcome` records in caller order. Initialization,
request, advance, export and declared slot failures affect one run. An unexpected
shared-operation exception affects its submitted active group; other groups and
completed results survive. Interrupts propagate after cleanup. The physical
context pool cap applies; caller-owned feature records and allocations survive.

Custom recipe callables and arbitrary learner options are rejected explicitly.
Use `SquaredPhase` for custom tree requests and external scheduling. The
[historical two-round Housing audit](checkpoint.md#historical-observations-and-available-evidence)
covers grouped, reversed and regrouped schedules, isolated failure and same-ID
retry. Full-budget ensembles, full R9/E4 acceptance and exact candidate validation
remain open; no speed claim follows.
