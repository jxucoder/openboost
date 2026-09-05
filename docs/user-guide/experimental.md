# Experimental CPU extensions

`openboost.experimental` is a small, evolving research API. Its `Booster`
uses the existing unified trainer and currently executes on CPU. CUDA requests
fail before fitting; `fallback="warn"` explicitly selects the entire CPU path.
The P2 legacy CUDA baseline does not establish experimental CUDA support.

An objective declares `channel_names` as a tuple of unique strings and
`supported_devices` as a frozenset. It implements `init_raw`, `step`,
`loss_value` and `constrain`. `step` receives the entire round-start raw state,
CPU target and optional weights, plus a frozen `ExecutionContext` with device,
NumPy module, per-fit generator, round index and channel (None for objectives).

Each channel returns an independent `(grad, hess)` tuple of contiguous float32
vectors. Gradient points toward increasing loss; Hessian is nonnegative
effective curvature, potentially a Fisher/preconditioner rather than an exact
second derivative. The objective applies sample weight to both arrays exactly
once; the trainer does not apply it again. `loss_value` should report the
weighted mean. All-zero, negative and non-finite weights are rejected.

Inputs are read-only views. Output buffers must not alias inputs, each other,
or another channel's statistics; they remain valid until the round finishes.
These checks prevent accidental buffer reuse without copying full training
arrays every round. They do not sandbox hostile Python code.

`DistributionObjectiveAdapter("normal", natural=True)` exposes existing
NaturalBoost objective math, including its unweighted initialization estimate.
`TrainerConfig` is shared with the internal trainer; it includes `random_state`
and `min_gain`. The initial CPU facade supports 0–8 tree depth, 2–254 bins,
numeric input including NaN, or a CPU `openboost.BinnedArray`. No new
performance or calibrated-quality guarantee follows from these interfaces.

Builder/schedule and inference persistence contracts are being completed in
P3; see the repository execution checklist for their verified status.
