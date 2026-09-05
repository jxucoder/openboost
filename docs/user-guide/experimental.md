# Experimental extensions

`openboost.experimental` is a small, evolving research API. Its `Booster`
uses the existing unified trainer. CPU supports the wider extension surface;
strict `device="cuda"` uses declared CUDA objectives and builders with resident
CuPy raw scores, targets, weights and updates. The built-in Normal/Poisson
adapter declares CUDA support; independent example packages provide NumPy/CuPy implementations.
Unsupported capability with `fallback="warn"` selects the entire CPU path before
training; runtime plugin/kernel failures raise and restore prior model state.

An objective declares `channel_names` as a tuple of unique strings and
`supported_devices` as a frozenset. It implements `init_raw`, `step`,
`loss_value` and `constrain`. `step` receives the entire round-start raw state,
target and optional weights on the declared device, plus a frozen
`ExecutionContext` with device, NumPy/CuPy array module, per-fit generator, round index and channel (None for objectives).

Each channel returns an independent `(grad, hess)` tuple of contiguous float32
vectors. Gradient points toward increasing loss; Hessian is nonnegative
effective curvature, potentially a Fisher/preconditioner rather than an exact
second derivative. The objective applies sample weight to both arrays exactly
once; the trainer does not apply it again. `loss_value` should report the
weighted mean. All-zero, negative and non-finite weights are rejected.

CPU inputs are read-only views; CUDA inputs are isolated device copies checked
for mutation. Output buffers must not alias inputs, each other,
or another channel's statistics; they remain valid until the round finishes.
These checks prevent accidental buffer reuse. CPU uses views; strict CUDA pays
for defensive device copies and validation. They do not sandbox hostile Python code.

`DistributionObjectiveAdapter("normal", natural=True)` exposes existing
NaturalBoost objective math, including its unweighted initialization estimate.
`TrainerConfig` is shared with the internal trainer; it includes `random_state`
and `min_gain`. The initial CPU facade supports 0–8 tree depth, 2–254 bins,
numeric input including NaN, or a CPU `openboost.BinnedArray`. No new
performance or calibrated-quality guarantee follows from these interfaces.

For a one-page implementation checklist and capability matrix, see
[Write an experimental extension](../cookbook/experimental-extensions.md).

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

Save with `model.save("model.ob")` and load with `Booster.load("model.ob")`.
The saved state contains package-owned trees, binning metadata, base scores and
per-tree coefficients. Objective, builder and schedule objects are excluded.
Loading supports CPU `predict_raw` without installing the training plugins;
loaded models are inference-only. Create a fresh Booster for further training.
As with other OpenBoost persistence, load only trusted files (joblib).

Early stopping restores trees and coefficients together. Checkpoints can be
loaded through the same inference API. A state without coefficients uses the
saved constant learning rate for each tree; this compatibility rule cannot
recover a missing nonconstant schedule. Unsupported experimental format versions
and old categorical states are rejected. Existing model persistence policies
are unchanged.

## Fixed-slot histogram primitive

`build_histograms(binned, grad, hess, sample_node_ids, active,
memory_budget_bytes=256 * 1024**2)` returns `HistogramBatch` with float32
`grad`/`hess` arrays `(slots, features, 256)`, int32 `counts` `(slots,)`, and an
owned bool `active` mask. Inputs are contiguous NumPy arrays or CuPy arrays on
the current device. G/H already include objective weights: no second weighting
occurs. Counts include zero-weight rows. ID -1 excludes a row; inactive slots
ignore assigned rows. Empty slots are zero; bin 255 remains the missing bin.

The limit is 511 slots. The budget covers returned arrays, excluding inputs
and transient device validation masks; allocation is rejected before creating
histograms if it would exceed the budget. CUDA uses the current CuPy stream,
retains aggregation results on device, and synchronizes scalar validation checks.
Floating-point CUDA accumulation order is not deterministic. No speed claim is
made. The primitives are composed by `LevelWiseBuilder` below and used by strict CUDA
extension sessions in the shared trainer.

## Numeric split and routing primitives

`find_splits(histograms, reg_lambda=1., min_child_weight=1., min_gain=0.)`
returns `SplitBatch`: int32 feature/threshold/left_child/right_child arrays,
float64 gain, and bool valid mask for each fixed slot. Prefix sums and scores
use float64 from the float32 histograms. Gain is the unhalved L2 score; it must
be positive and at least min_gain. Both children require positive H and at
least min_child_weight, including when min_child_weight=0. Exact ties select
the first feature, then the first threshold. Inactive, empty, terminal and
unsplittable slots have -1 indices, zero gain and valid=False.

`partition(binned, sample_node_ids, splits)` returns a new int32 array of routed
IDs. It preserves -1 and nodes without a valid split. Children use fixed indices
`2*i+1` and `2*i+2`; routing never modifies the input IDs. Construct the next
active mask from valid children and call `build_histograms` on these routed IDs
to build actual child statistics.

These operations accept the same contiguous NumPy/current-device CuPy boundary
as histograms. GPU arrays stay on device; scalar input checks synchronize.
Only numeric L2 splitting is supported. Histogram missing-bin G/H must be zero,
and routing rejects bin 255 even for zero-weight rows. Callers must provide
numeric bins; categorical metadata is outside this primitive API. `LevelWiseBuilder`
rejects missing/categorical inputs before growth. These primitives alone do not establish an end-to-end speedup.

## Leaf reduction and custom rules

`reduce_leaves(grad, hess, sample_node_ids, active)` returns `LeafStatistics`:
float32 grad/hess sums, int32 physical row counts, and an owned bool active mask,
all `(slots,)` on the input device. Inputs use the same contiguous array and
fixed-slot contract as histograms. Statistics are already weighted; zero-weight
rows still count. ID -1 and inactive assignments are excluded.

`leaf_values(..., leaf_rule=None, config=None, context=None)` performs this
reduction and calls `rule.values(G, H, config=config, context=context)`. A rule
declares `supported_devices` and returns contiguous finite float32 `(slots,)`
on that device. Pass the fit's shared `ExecutionContext` when composing a
builder. Standalone calls create a context from the config seed. The rule gets
private compact G/H copies and a config copy; changing G/H is rejected. Returned
values are detached from plugin scratch. No sample-sized defensive copy occurs.

`NewtonLeafRule` implements L2 `-G/(H+reg_lambda)`. Zero denominator with zero G
returns zero; nonzero G fails. Empty, inactive and zero-G/zero-H slots must be
zero for every rule. L1 is rejected by the default rule. A bounded rule can use
`context.xp.clip(NewtonLeafRule().values(G, H, config=config, context=context),
-c, c)` with a validated positive bound. Clipping changes leaf outputs and the
next round's gradients, while keeping the split criterion unchanged.

CUDA reduction and rule arithmetic stay on device; scalar validation checks
synchronize. Named download-wrapper checks are not a complete profiler trace.
The assembled level-wise builder is described below.

## Level-wise builder

Select `LevelWiseBuilder(leaf_rule=..., memory_budget_bytes=...)` explicitly to
compose the batch primitives. It supports numeric, nonmissing features, L2,
full row/feature sampling and depth 0–8. Both children must have positive
curvature to split. Unsupported metadata/parameters and insufficient histogram
budgets fail before growth. The existing CPU default builder remains available.

```python
import numpy as np
from openboost.experimental import (
    Booster, DistributionObjectiveAdapter, LevelWiseBuilder, TrainerConfig,
)

rng = np.random.default_rng(7)
X = rng.normal(size=(64, 3)).astype(np.float32)
y = (0.5 * X[:, 0] + 0.3 * rng.normal(size=64)).astype(np.float32)
model = Booster(
    objective=DistributionObjectiveAdapter("normal", natural=True),
    tree_builder=LevelWiseBuilder(),
    config=TrainerConfig(n_trees=2, max_depth=2, random_state=7),
).fit(X, y)
raw = model.predict_raw(X)
```

Direct `LevelWiseBuilder.build` also accepts a BinnedArray with contiguous CuPy
data and matching CUDA ExecutionContext. Metadata stays on host. This entry
currently requires the default CUDA stream. It returns a standard host
TreeStructure and an owned CuPy training prediction in BuiltTree. Only five
compact tree arrays are downloaded after growth; sample statistics, routed IDs
and histograms stay on device. Fixed slots and previous histogram release keep
one histogram batch live between levels; the budget excludes input arrays,
prediction caches, compact tree arrays and transient validation masks.

The device cache survives release of input views. Host tree arrays support
CPU prediction and existing persistence. Direct GPU builder composition is
validated separately from strict GPU `Booster.fit` integration.
The default numeric CPU builder is not replaced: this opt-in builder has a
narrower feature boundary and no established end-to-end performance advantage.

## Independent extension packages

The repository's `examples/extensions/` contains two separately buildable CPU/CUDA
packages. `normal_fisher` implements weighted Gaussian NLL gradients, expected
Fisher curvature and a nonconstant per-channel schedule. `bounded_leaves`
implements a bounded Newton leaf rule through the public API. They compose in
`demo.py` without core changes or private imports.

From a development checkout, verify the real installation boundary with:

```sh
uv run --no-sync python examples/extensions/verify_wheels.py /tmp/openboost-extension-evidence
```

The verifier builds three wheels and installs them in a fresh environment outside
the repository, runs independent mathematical and training checks, executes the
public example, then uninstalls both plugins. A new interpreter checks exact CPU
predictions for six saved models without the training packages. The example's
CPU dependency pins avoid the failed Intel macOS source build encountered with
Numba 0.63.1 / llvmlite 0.46.0; see its README for the tested versions and setup.

This verifies a CPU extension installation boundary. These repository-authored
examples do not establish external adoption. GPU installation is checked separately
with the foundation `extensions` suite described below.

## Strict CUDA fit boundary

Set `device="cuda"` with a CUDA-capable objective. When no builder is specified,
CUDA selects `LevelWiseBuilder`; an explicit builder always takes precedence.
CPU retains `CPUHistogramBuilder`. Host numeric features/targets/weights enter
fit; binning and initialization run on CPU, then binned values, targets and
weights upload once. Raw state and per-round arithmetic use CuPy. Standard host
tree arrays finalize each tree; `predict_raw` remains CPU inference and saved
models load without training plugins.

Strict CUDA rejects missing/categorical data, L1, sampling, eval sets, callbacks,
early stopping, unsupported leaf rules and non-default streams before training.
The first version supports numeric nonmissing L2/full-sampling trees. Invalid
plugin results, modified borrowed inputs or a cached prediction inconsistent with
the returned tree fail rather than retrying on CPU. CuPy lacks NumPy read-only
views, so the session copies borrowed inputs on-device and checks for mutation.
It independently traverses compact trees on-device to validate prediction caches.
These copies, compact uploads and scalar synchronization have a performance cost.

`fit_report_` separates CPU binning/initialization, actual objective/tree/update
execution, fallback reason and transfer scope. Named transfer-wrapper tests do
not prove whole-process transfer absence. Profiler availability and real-device
results are recorded in the foundation evidence; no speedup is claimed here.


The independent examples at version 0.2.0 support explicit CUDA context arithmetic.
Use `python demo.py --device cuda` after installing both extension wheels and CUDA
dependencies. CPU initialization and host input requirements remain unchanged.
The foundation `extensions` suite installs all three wheels in an isolated T4
container, checks GPU objective mathematics and composed training, then uninstalls
both training plugins before CPU inference in a new interpreter. See
`examples/extensions/README.md` for build and Modal commands. This establishes
an installation/conformance boundary, not third-party adoption or performance.
