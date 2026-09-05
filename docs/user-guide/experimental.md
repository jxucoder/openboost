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

## Tree builders and updates

Pass `tree_builder` and `step_schedule` explicitly to `Booster`. A builder
implements `build(binned, grad, hess, *, config, context)` and declares CPU
capability. Context now identifies the round and channel; the builder never
receives raw scores. It returns `BuiltTree(TreeStructure, train_prediction=None)`.
Only standard scalar package-owned trees are supported. Tree topology, values
and optional missing/categorical metadata are validated; compact state is
copied before storage so builder scratch buffers can be reused. If provided,
the cached prediction must exactly match tree prediction on training data.

Explicit builders always take precedence over native eligibility. The current
`CPUHistogramBuilder` delegates splitting to the existing CPU core. Split gain
uses the unhalved score `G_L²/(H_L+lambda) + G_R²/(H_R+lambda) - G²/(H+lambda)`
(with L1 soft thresholding when requested), not half that value. The default
leaf is `-G/(H+lambda)`. Zero denominator and zero G yields a zero root leaf;
nonzero G fails. This adapter rejects the unverified combination
`reg_lambda=0, min_child_weight=0` before fitting; choose positive minimum child
weight for unregularized trees. It does not silently change those parameters.

A schedule implements `coefficients(round_idx, channel_names, base_learning_rate)`.
It returns every channel's full finite, nonnegative coefficient. The trainer
applies each coefficient exactly once and stores it in `coefficients_` for
prediction/evaluation. `ConstantSchedule` returns the configured learning rate.
Use schedules rather than learning-rate-mutating callbacks. Only predetermined
per-round coefficients are supported, not line search or reweighting old trees.

Inference persistence and coefficient-aware early stopping follow in P3.3.
