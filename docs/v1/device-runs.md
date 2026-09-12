# Experimental sequential device runs

`openboost.device_runs` schedules independent scalar device work sequentially.
Its historical component checks include state isolation, retained failures and
fresh final/best inference. See the [checkpoint](checkpoint.md) for the available
evidence and pending candidate validation. The example below requires real CUDA.

```python
import numpy as np
from openboost import NumericData, Problem
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_recipes import squared
from openboost.device_runs import RunSpec, run_many
from openboost.execution import ExecutionContext

train_data = NumericData([[0], [1], [2], [np.nan]], [0, 1, 2, 3], ("x",))
valid_data = NumericData([[0.5], [1.5]], [10, 11], ("x",))
train = Problem(train_data, [[0], [1], [3], [2]], train_data.row_ids)
validation = Problem(valid_data, [[0.5], [2]], valid_data.row_ids)
cuts = Binning.fit(train_data, bins=4)
specs = tuple(
    RunSpec(
        f"fit-{i}", 17, train, validation, squared,
        options={"rounds": rounds, "max_depth": 1, "learning_rate": 0.1},
        binning=cuts,
    )
    for i, rounds in enumerate((2, 4, 8))
)
with ExecutionContext(max_bytes=8 * 1024**2) as context:
    outcomes = run_many(DeviceOperations(context), specs)

# Results contain CPU artifacts, so prediction happens after the context closes.
for outcome in outcomes:
    if outcome.result is None:
        print(outcome.run_id, outcome.error_type, outcome.error_message)
    else:
        print(outcome.run_id, outcome.result.stop.reason)
        print(outcome.result.best_model.predict(valid_data))
```

`RunSpec` owns a copy of immutable scalar options. A recipe callable receives
`(ops, train, validation, run_id=..., seed=..., **options)` and the explicit
`binning`, when supplied. Put custom learner configuration in that callable.
This first reference accepts scalar squared-compatible target schemas and native
`DeviceFitResult` results. Specialized targets are rejected before dispatch;
their inference metadata needs an explicit future scheduling contract.

The scheduler checks all stable IDs for uniqueness before running any recipe.
Each successful outcome contains final/best `Model` artifacts, detached
`DeviceState` diagnostics, native step records and a terminal `StopState`.
Ordered channel observations may share an outer-round index. Diagnostic state
cannot be used as an accepted record in a new device run.

Ordinary exceptions become that run's error outcome. Partial work and malformed
results are released before the next recipe. A wrong ID, seed, problem, explicit
binning, requested round budget, foreign state or borrowed run cannot become a
successful outcome. `KeyboardInterrupt` and `SystemExit` propagate after cleanup.
Callbacks must respect caller-owned allocations; this is cooperative ownership,
not process isolation. The context memory cap still applies to live device work.

The default path reuses supplied host cuts and prepares device inputs per fit.
The optional resident feature path has historical sharing, cap, cleanup and
rebinding checks. Its original failed cap fixture and separate correction remain
in development history; the new raw archives are omitted from this PR.
Prepare codes/missingness once and pass them explicitly:

```python
from dataclasses import replace
from openboost import device_inputs

with ExecutionContext(max_bytes=8 * 1024**2) as context:
    ops = DeviceOperations(context)
    prepared = tuple(
        device_inputs.prepare(ops, cuts.transform(data))
        for data in (train_data, valid_data)
    )
    try:
        outcomes = run_many(ops, tuple(replace(s, prepared=prepared) for s in specs))
    finally:
        for features in prepared:
            ops.release(features)
```

Feature identity includes values, source row IDs, order, schema and fitted cuts.
Changed targets, offsets and weights can reuse those features; each fit binds its
own weights and objective inputs. Raw predictions, models, RNG and stopping stay
independent. The recipe must accept the explicit pair; ignoring it is an error.
Keep feature records alive while runs borrow them. Explicitly releasing the
caller-owned features invalidates borrowers, whose cleanup remains available.

`execution="sequential"` is the only supported mode; batching and parallel
requests fail explicitly. Feature reuse is preparation work, not GPU batching;
there is no measured speed improvement for it yet.

The required M=1/8/32 device tests compare direct, requested, reversed and
regrouped outcomes by stable ID. They cover stopping, no-ops, rejection, a later
failure, same-ID retry, keyed RNG and fresh final/best inference. All pass in the
declared Sprint 122 scalar scope. Resident sharing also has its separate bounded
device evidence. A separate [compatible grouped scheduler](device-groups.md#compatible-squared-scheduling)
composes grouped operations through `openboost.device_group_runs`; the sequential
module's mode remains unchanged. The [checkpoint](checkpoint.md) describes the
bounded real diagnostic and the unfinished full-budget and broader recipe scope.
