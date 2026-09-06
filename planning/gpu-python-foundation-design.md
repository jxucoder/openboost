# GPU Python boosting foundation: design draft

> Historical design and evidence record. Follow the [new F0–F5 plan](agent-boosting-foundation-plan.md)
> for subsequent work: the foundation is the product and incompatible redesign is permitted.
> Later user instructions supersede this document's requirements to reuse the old trainer,
> preserve fixed interfaces, and prioritize a distributional product. Original tests and failures remain evidence.

Status: P0–P6 technical and installation verification complete; P7 quality passed,
GPU performance budget failed, and isolated profiling and design review completed.
Exact peak device memory and CUDA traces remain unverified. External adoption is
unverified. Date: 2026-09-05.

This document defines interfaces, boundaries, verification, and execution order for
implementation by a medium model. Consult the execution checklist for actual
behavior and evidence from completed phases. The P2 baseline does not verify the
new extension API's GPU path. See the [execution checklist](gpu-python-foundation-execution.md).

## 1. Product hypothesis to test

> A Python researcher can change boosting objectives, tree construction, or update
> rules in an independent package, reusing OpenBoost's CPU reference, GPU data path,
> prediction, and verification tools without maintaining a fork.

This research infrastructure direction is a hypothesis. Python implementation
percentage, GPU labels, and API counts are not success metrics. The existing
[product mission and evidence rules](../AGENTS.md) remain applicable; NaturalBoost
is the first real consumer. This iteration does not rename or promote the whole
repository as a general replacement.

Bound the initial investment to a suggested 6–8 week exploration window with
technical gates. This limits scope; it is not a delivery promise. Establish one
trustworthy, modifiable path before expanding tree structures or algorithms.

| Goal | Evidence available in this iteration | Conclusions it cannot establish |
|---|---|---|
| Impact | Three extension types change training results, checked against independent mathematical/routing references | Algorithmic novelty or publication impact |
| Adoption | Two independent wheels depend only on public experimental interfaces and run in clean environments | Self-authored examples do not prove external adoption |
| Value | Record implementation effort, end-to-end time, quality, device memory, and migration obstacles | Willingness to pay without interviews |

The product gate for further investment is at least two external developers
trying extensions, with one completing their own method without core changes.
Record assistance and failures. Contact and invitations require separate user
authorization; prepare runnable materials without sending messages. Technical
acceptance does not substitute for this product gate.

## 2. Starting point: integrate existing work first

Design branch: `codex/gpu-python-foundation-design`, created from local `main` at
`82cf1e25b21a69093e85a270af7eb93c9ae7aa19`.

After fetching on 2026-09-05, the remote baseline was fixed at
`6ebe3a8ced0e621b17e3cf63e31721af58471053`. The common ancestor was
`504fdd0bfc60e5d8e7518250e087fb7e4766d1b4`; there were 12 local-only and 9
remote-only commits at branch creation, excluding later design commits.

The remote already has `_trainer.py`, `_objectives.py`, FormulaBoost, and
WeibullAFT; reuse them. Preserve local categorical persistence/cardinality fixes,
batch fail-fast behavior, ScoringBench, performance CI, and AGENTS/learnings.
Do not reset away either side or write another training loop.

Read-only `git merge-tree` found textual conflicts in `CLAUDE.md`,
`docs/getting-started/gpu-setup.md`, and `docs/getting-started/installation.md`.
This is not a complete semantic conflict list. Even a successful automatic merge
requires persistence, CUDA eligibility, and documentation evidence review.
Perform the merge in P0.

| Inspected call path | Design consequence |
|---|---|
| Remote `fit_boosting` invokes the objective by channel but directly selects native trees / `fit_tree` | Add builder and schedule arguments here; explicit extensions must participate in dispatch |
| Remote `TrainerConfig` has no seed; growth uses global `numpy.random` calls | Pass seed/RNG through the selected path, not merely a new configuration field |
| Extensible primitives download histograms / sample node IDs; `compute_leaf_values_gpu` delegates to CPU | Add device batch representations and leaf reduction; renaming interfaces does not establish GPU residency |
| Native builder can update raw in place; old tree facade uses host arrays for persistence | Trainer owns raw updates; allow compact structure downloads after tree completion |
| Remote selects device kernels by distribution class name and broadly catches exceptions before CPU fallback | Declare capabilities, strict mode, and visible fallback; prohibit name-based builtin selection |
| Remote decides `unit_hessian` separately from `sample_weight` | Native `const_hess=1` may override nonuniform weights; reproduce before fixing |
| Eval/callbacks currently download host raw or predictions | Report their capability and cost separately; do not claim residency for every training configuration |

These are source observations. The suspected weighted-Hessian issue has not yet
been reproduced on GPU. The local extension baseline is **35 passed, 1 skipped**,
which does not establish correctness of those GPU paths. Remote design speed
numbers are not verified evidence for this design.

## 3. MVP support boundary

Add interfaces under `openboost.experimental`, initially promising contracts only
within one explicit version. Preserve existing model defaults; connect NaturalBoost
and cover it with regression tests. Do not migrate every model immediately.

| Item | MVP decision |
|---|---|
| GPU | One NVIDIA GPU, CUDA 12, Numba/CuPy RawKernel kernels; extensions receive CuPy arrays |
| CPU | NumPy oracle and runnable experimental engine; no GPU bitwise equality requirement |
| Data | Dense numeric features; sample-major input, feature-major bins; one-dimensional y, multichannel raw |
| Trees | Level-wise, scalar leaves, one tree per channel per round; maximum depth 8 |
| Objectives | Multichannel objectives with explicit CPU/CUDA implementations; two-parameter Normal is first acceptance case |
| Weights | Finite nonnegative sample weights with positive total; applied once, including zero weights |
| Sampling | Initial experimental GPU requires subsample=colsample=1; reject other values explicitly |
| Regularization | Initial experimental GPU uses L2, reg_alpha=0; enforce min_child_weight and min_gain |
| Missing/categories | Initial experimental builder rejects them; continue testing old model support and preserve category cardinality fixes |
| Metadata | Initial experimental fit rejects exposure/censoring; old models retain their existing capabilities |
| Callbacks/eval | CPU may use the existing flow; strict GPU residency initially covers training without callbacks/eval; report hybrid evaluation paths explicitly |
| Prediction/storage | CPU raw prediction required; verify GPU prediction and GPU save followed by CPU load |
| Excluded | Ray, multi-GPU, out-of-core, GOSS, train-many, vector leaves, autodiff framework interoperability, arbitrary dynamic training graphs |

“Pure Python” means users and maintainers modify algorithms with Python/CuPy/Numba.
It does not promise no compilation, no CUDA runtime, or automatic GPU compilation
of arbitrary Python functions.

## 4. Interfaces and training semantics

### 4.1 One loop, three explicit extension points

```text
Existing facades such as NaturalBoost       experimental.Booster
                         \                 /
                            fit_boosting
                                 |
                   Objective.step(raw at round start)
                                 |
                      TreeBuilder.build per channel
                                 |
                  trainer applies StepSchedule coefficients
                                 |
                   tree storage / eval / persistence / report
```

The following is a target contract, not an executable current API. Freeze names
in P3. Pass objects directly without a plugin discovery system. Thin adapters
connect existing Objectives without rewriting distribution mathematics.

```python
class Objective:
    channel_names: tuple[str, ...]
    supported_devices: frozenset[str]  # e.g. {"cpu", "cuda"}

    def init_raw(self, y, sample_weight=None, extra=None): ...
    # CPU initialization once; returns {channel: finite scalar}.

    def step(self, raw, y, sample_weight=None, extra=None, *, context): ...
    # Returns {channel: (grad, hess)} on the same device as raw.

    def loss_value(self, raw, y, sample_weight=None, extra=None, *, context): ...
    def constrain(self, raw, extra=None): ...

class TreeBuilder:
    supported_devices: frozenset[str]

    def build(self, binned, grad, hess, *, config, context): ...
    # Returns BuiltTree(tree, train_prediction=None).

class StepSchedule:
    def coefficients(self, round_idx, channel_names, base_learning_rate): ...
    # Returns {channel: finite nonnegative scalar}; full coefficient, not multiplier.
```

`experimental.Booster(objective=..., tree_builder=..., step_schedule=..., config=...,
device="cpu"|"cuda", fallback="error"|"warn")` is a thin facade over the existing
trainer. It offers
`fit(X,y,sample_weight=None,eval_sets=None,callbacks=None,early_stopping_rounds=None)`,
`predict_raw(X)`, `save(path)`, and `load(path)`. Initially CPU supports evaluation
arguments; strict GPU rejects them before the first update. Do not duplicate the
distribution prediction API. Users can call
`objective.constrain(booster.predict_raw(X))`.

Add `random_state` and the missing `min_gain` to `TrainerConfig`, keeping one source
for each hyperparameter. `ExecutionContext` supplies `device`, `xp` (NumPy/CuPy),
fit-scoped `rng`, `round_idx`, and `channel`. New interface objects do not infer
execution location from a global backend. Internal legacy adapters still use
`backend_context`, restoring it after fit; mixed-backend concurrent fits in one
process are prohibited. Objective context.channel is None; builder context.channel
is the current channel. Plugins do not mutate context. Builtin adapters consume
the added context argument before invoking existing objective mathematics.

### 4.2 Explicit mathematical and array contracts

1. Each raw/gradient/Hessian channel is contiguous float32 `(n_samples,)`.
   Keys must be complete with no extras and fixed channel order. Reject y that
   is not one-dimensional rather than hiding errors with ravel.
2. `grad` points toward increasing loss. Default leaf value is
   `-sum(grad)/(sum(hess)+reg_lambda)`. `hess` is nonnegative effective curvature
   for tree optimization, possibly Fisher/preconditioning rather than the exact
   Hessian. Regularization is finite and nonnegative. A zero denominator with
   G=0 yields a zero leaf; a zero denominator with nonzero G is an error.
3. The objective applies sample weights to both grad and hess **once**; trainer
   and builder do not reapply them. Default loss is a weighted mean. Reject all-zero,
   negative, NaN, or infinite weights.
4. Initially disable constant-Hessian hints for custom objectives. Builtin hints
   require a suitable objective, no sample weights, and no sampling/transformation
   that changes Hessians, with comparison tests; `natural=True` alone is insufficient.
5. `step` reads all raw channels at round start once and returns all channel
   statistics. Build trees in fixed order without recomputing other channel
   gradients after each update. Inputs are read-only; returned buffers live at
   least until the round ends.
6. `F[r+1,k] = F[r,k] + eta[r,k] * tree[r,k](X)`. Builders cannot modify raw.
   `BuiltTree` contains a tree and optional training predictions, with no implicit
   “raw already updated” state. The trainer updates exactly once and stores the
   actual `eta[r,k]`.
7. Initial `StepSchedule` only supplies predetermined per-round/channel coefficients.
   It cannot read/modify raw or the old tree collection. Line search, momentum, and
   reweighting all old trees are outside this deliberately narrower-than-UpdateRule contract.
8. CUDA extensions receive CuPy arrays; internal Numba uses zero-copy CUDA Array
   Interface views. MVP uses the default stream; caller-provided streams are
   unsupported. Test owner references and lifetime; do not mistake host ndarrays
   for device arrays. Initialization/final output may copy; no implicit `.get()`
   in the hot path.
9. CPU seeds reproduce results without changing global RNG state. Old sampling
   paths receive the same scoped Generator; explicit indices may be reused in
   CPU/CUDA comparisons. No GPU bitwise determinism claim.

### 4.3 Dispatch, errors, and reports

Explicit builders take precedence even when inputs qualify for a native path.
Default builders may select verified native kernels; adapters pass `pred_gpu=None`
and the trainer applies updates. Some fusion may initially be lost: measure its
cost without breaking update semantics for a speed number. Do not yet rewrite
all of the separate old `_models/_boosting.py` loop.

CUDA capability checks use explicit declarations and builtin implementation
identity, not `type(...).__name__`. Experimental GPU defaults to `fallback="error"`:
unsupported capabilities fail before the first training update. `fallback="warn"`
may select and report a complete CPU path before fit only for known capability
gaps. Do not catch arbitrary round-time exceptions and silently continue after
copying raw. Numerical errors, invalid shapes, and kernel failures retain their
causes and fail. Existing facades may retain legitimate hybrid execution if
reports describe it clearly.

`fit_report_` records at least requested/actual device, actual objective/tree/update/eval
locations, builder path, fallback reason, seed, trees per channel, and timing
synchronization boundaries. Transfer counts cover only OpenBoost wrappers;
external CuPy code may copy independently, so counts are not process-wide proof.
Check residency with a small profiler trace as well; shapes or `cuda.is_available()`
are not substitutes.

## 5. Tree primitives: Python modification with a GPU data path

Keep the host `dict[int, NodeHistogram]` API compatible and add an experimental
batch API without pretending return-type compatibility. Reuse histogram/split/partition
kernels first, establishing boundaries before performance optimization.

| Batch representation/operation | Target shape and responsibility |
|---|---|
| HistogramBatch | grad/hess `(node_slots, features, 256)` float32; int32 sample counts and bool active mask on the same device |
| SplitBatch | Per-node_slot feature/threshold/child IDs/gain/valid-mask arrays on the same device |
| `build_histograms` | Aggregate actual sample node IDs; define zero weights, empty nodes, and inactive slots |
| `find_splits` | Compare candidates in batches, exclude invalid children, break equal-gain ties by feature then threshold |
| `partition` | Update sample node IDs using splits; return a new array or explicitly declare in-place buffers |
| `leaf_values` | Device sum/reduction and leaf rule; no grad/hess/sample-ID download |

To bound dynamic allocation, initial level-wise growth uses fixed complete-binary-tree
slots with a device active mask: root=0, left=2i+1, right=2i+2; leaf children=-1.
Fixed per-level loops need not download sample arrays to count active nodes.
Maximum depth is 8; default histogram temporary budget is 256 MiB. Reject budget
excess rather than implementing out-of-core. Bin 255 remains reserved for missing
values even on a path that rejects missing inputs. Independently test constant
features, all-zero effective curvature, no valid split, and negative/nonfinite gain.

`LevelWiseBuilder(leaf_rule=...)` exposes the first minimal tree extension: a leaf
rule receives batched G/H, config, and context and returns same-device leaves.
Users needing deeper changes can compose public batch primitives. Arbitrary
Python callbacks are not automatically injected into compiled kernels.

Return the existing scalar `TreeStructure`. Allow O(tree_nodes) structure/leaf
downloads after each tree, retaining device caches for training predictions.
This reuses CPU prediction and serialization without rewriting all tree objects.
Allowed transfers are initialization, compact structures per tree, necessary
scalar state/error checks, explicit evaluation, and final output. The hot path
must not download O(samples) raw/grad/hess/node IDs or complete histograms.

Split-gain scaling and min_gain comparisons follow verified CPU behavior, with
independent formula tests in P3. If CPU/native formulas disagree, fix the
correctness issue instead of temporarily relaxing parity thresholds.

## 6. Three extension experiments, two independent packages

Place packages in `examples/extensions/normal_fisher/` and
`examples/extensions/bounded_leaves/`, each with pyproject, README, and independent
tests. Build and install the OpenBoost and extension wheels in fresh environments,
then run outside the repository. Prohibit editable/PYTHONPATH injection, private
imports, core copies, or dispatch monkeypatching. Sources may share a repository,
but tests must verify the installation boundary; this is not external adoption.

**A. External two-parameter Gaussian/Fisher objective.** Raw channels are mu and
log_sigma; `s2=exp(2*log_sigma)`, `g_mu=(mu-y)/s2`, `h_mu=1/s2`,
`g_log_sigma=1-(mu-y)^2/s2`, and `h_log_sigma=2`, then apply weights. Initialize
location and variance using weighted means, with variance floor 1e-6. Minimal
tests use a finite range and reject nonfinite state. Check gradients against
independent finite-difference NLL and Fisher against an analytic reference.
Implement CPU/CUDA. This tests independent implementation and extensibility,
not Gaussian Fisher novelty.

**B. External bounded-Newton leaf.** During tree growth use
`clip(-G/(H+lambda), -c, c)`, with zero leaves for no effective samples and finite
c>0. Compare against an unclipped reference and observe changes in actual leaves
and next-round gradients, not just callback invocation. GPU clipping/reduction
runs on device and saved results persist. The method retains the default split
criterion; it does not reoptimize every split for a clipped objective.

**C. Nonconstant per-channel schedule.** Implement in package A:
`eta[r,k] = base_lr * channel_scale[k] / (1 + r/tau)`, tau>0, mu scale=1 and
log_sigma scale=0.5. Hand-check all coefficients in a two-round example, then
check training raw, new predictions, early-stop restoration, and save/load.
This verifies the update interface, not line search or a new training algorithm.

After A/B, arrange an external author experiment. If an actual external method
still needs private imports or a fork, record the missing interface before
adding speculative hooks.

## 7. Persistence and compatibility

Save the binner, base scores, channels, tree arrays, actual per-tree coefficients,
and version. Coefficients must share semantics across prediction, eval,
early-stop truncation/restoration, and persistence, not just fit. For old files
without coefficient state, synthesize it from old learning_rate and test.

Experimental Booster raw inference does not depend on custom objective/builder/schedule
code. Do not serialize lambdas or arbitrary training objects for deployment.
Loaded raw predictions work; further training requires resupplying the objective.
Initial warm-start/resume is explicitly unsupported. Existing builtin model
prediction transforms retain their behavior.

If the shared serializer changes, increment its actual current version and keep
old-version rejection rules. Regress any touched numeric, missing, categorical,
symmetric/linear/vector, or other specialized state; do not delete unrecognized
fields for the experimental path.

## 8. Correctness, performance, and product gates

| Gate | Required evidence |
|---|---|
| G0 Integration | Both histories preserved; local fixes, remote new models, and CPU suite pass or existing failures are explicit |
| G1 CPU contract | Independent math oracle, seed, single weighting/update, explicit dispatch, error paths, saved prediction equality |
| G2 CUDA baseline | Real GPU weighted/unweighted Normal/Poisson end-to-end checks; no known failures carried into the new API |
| G3 Resident extensions | A/B independent wheels run on CPU/CUDA and C changes updates; verify gradients, splits, leaves, predictions, and task metrics |
| G4 Engineering value | Cold/warm end-to-end time, device memory, transfers, and matched quality, with committed raw results |
| G5 External adoption | External authors' extensions and obstacles; no attempts or all requiring forks does not pass |

Prefer exactly representable micro-oracles without approximate ties. Starting
hist/grad/leaf tolerances: rtol=1e-5, atol=1e-6; split topology must match on these
data. Test exact optimal ties separately. For larger floating reductions use
explained prediction/quality bounds, not mandatory equality of every tree.

Freeze seeds=0,1,2 in advance. Start numeric real regression with a frozen
sklearn California Housing version/hash. Record download failures; do not replace
real evidence with synthetic data. Fit preprocessing only on training partitions.
Compare held-out Normal NLL, CRPS, and interval coverage; use a known Poisson
generating process for count checks. Separate synthetic scaling and real-data
quality from each other and from official ScoringBench.

Freeze P2 before observing implementation results. Prespecified screening bounds
for the same algorithm/configuration: absolute mean NLL difference no more than
`0.01*max(1,abs(baseline_NLL))`, CRPS relative degradation no more than 1%, and
90% interval coverage difference no more than 1 percentage point. Report every
seed and investigate failures. Three seeds do not establish statistical significance.
These wider real-data bounds cannot waive micro-oracle correctness.

A default GPU median end-to-end fit regression above 20% versus the integrated
old path triggers profiling and design review, without changing quality or
hiding warmup. This is a design budget, not measured performance or a speed
promise for custom algorithms. Prioritize reproducible records of public
interfaces, additional method code, and GPU cost needed to implement a method.

## 9. Modal verification design

P1 adjustment: use a separate `benchmarks/foundation/modal_app.py`, preserving
the old runner. The old app registers source mounts and other jobs with loose
dependencies, which would compromise wheel isolation. It copies a single test
file, uses loose dependencies, and some entry points only print failures; it
cannot directly supply this iteration's evidence. Modal supports image-time
dependency installation and explicit local-file inclusion, with official test
examples. Use the current SDK's uv installation path, not ad-hoc pip installs.
Sources: [image guide](https://modal.com/docs/guide/images),
[CI example](https://modal.com/docs/examples/ci-on-modal).

- Python 3.12, CUDA 12, fixed image digest and Linux dependency lock including
  numba-cuda, CuPy, pytest/xdist. Validate the lock in Linux; a macOS installed-package
  list is not a Linux environment.
- Build a wheel from a clean implementation commit and calculate SHA256. Upload
  only the wheel, required tests/conftest/config, extension wheels, and manifest.
  Verify site-packages import provenance and wheel hashes. Do not upload the
  whole workspace, .git, credentials, or user data.
- Default one T4, concurrency 1. Remote timeouts: `foundation_smoke`=300 seconds,
  `foundation_correctness`=1800, `foundation_benchmark`=1800; leave 60 seconds per
  pytest subprocess for collection. Run smoke then correctness before benchmarks;
  do not automatically expand GPU models or matrix size.
- App retry=0. Record cumulative remote seconds, failures, and existing run IDs.
  Initial plan: at most approximately 1 GPU-hour. This is not a hard spending cap;
  record image builds/startup and platform restarts separately. Modal timeouts
  limit individual executions and infrastructure retries may occur.
  [Timeouts](https://modal.com/docs/guide/timeouts),
  [failure handling](https://modal.com/docs/guide/functions).
- Local entry points such as `::foundation_smoke` control status. Any pytest
  failure, missing required GPU test, skipped selected required test, environment
  validation failure, or unretrievable result makes the local command fail.
  Legitimate CPU skips in mixed files are not failures; list required node IDs.
- Every result includes run ID, source SHA/dirty, wheel hash, dataset/version/hash/split,
  exact commands, dependencies, CPU/RAM/threads, GPU/driver/runtime, actual CUDA
  path, fallback, synchronization/warmup, JSON, JUnit, and logs. Save locally in
  `benchmarks/results/foundation/<run_id>/`; explicitly adjust ignore rules to
  commit reviewed frozen results.
- Correctness jobs do not advertise speed. Benchmarks measure complete fit/predict,
  first compilation separately from warm medians, quality/peak device memory,
  and differing CPU/GPU resources. Use at least 3 warm runs and synchronize GPU
  timing boundaries.

At planning time, Modal credentials, image builds, GPU quota, and actual prices
were unverified. Start with a minimal smoke; preserve failures and fix their
specific causes rather than repeatedly retrying long jobs.

## 10. Conditions for changing direction

- If both independent extensions still need core changes, the interface hypothesis
  failed: narrow or repair the contract first.
- If extensible GPU execution consistently lacks economic benefit on target
  workloads, preserve CPU research utility and pause GPU platform expansion.
- If external authors only need custom distributions, return investment to the
  distributional product instead of expanding the core for generality.
- If authors repeatedly contribute different algorithms and want independent
  packages to depend on OpenBoost, reconsider separate split/leaf gradients,
  vector leaves, and deeper update interfaces, each motivated by an actual algorithm.

See the [impact/adoption/value study](../learnings/2026-09-05-impact-adoption-value-strategy.md)
for competition and broader strategy. This plan seeks a narrow foundation that
can be adopted or falsified; it does not claim an existing ecosystem.

## 2026-09-05 goal review after P4.1

The larger objective remains useful, trustworthy distributional/risk modeling
and a shorter path from a research idea to a usable implementation. The GPU
foundation is one bounded hypothesis supporting that objective. Passing kernels,
more APIs and more Python code are not adoption or value evidence.

G0/G1/G2 and one histogram primitive have evidence. G3 (independent GPU
extensions), G4 (matched-quality end-to-end cost) and G5 (external author use)
remain open. Continue the smallest path through numeric split/routing, one
bounded leaf rule and two-channel Normal training to the two independent wheels.
Do not add tree families, custom split criteria or extra device support on the
way. The next product checkpoint is a reproducible method implemented through
public APIs, with implementation effort, installation obstacles and runtime
cost recorded; then an external author task, not another list of kernels.

Keep the existing stop conditions: if authors only need custom distributions,
return investment to the distributional product; if GPU gives no end-to-end
benefit, keep it optional. Prepare author materials without sending invitations
or publishing. External attempts and retention still require actual users.

Implementation clarifications: P4.1 uses a small CuPy RawKernel for its separate
layout/count contract; Python percentage and Numba-only kernels are not goals.
Its non-default stream test covers that primitive only, not the future whole
trainer. P4.2 numeric splitting requires positive curvature in both children,
including when min_child_weight=0; this avoids inventing information about empty
versus zero-weight bins from G/H alone. Zero-curvature nodes stay leaves; missing
and categorical builder support remains out of scope. These limits must remain
visible in the public contract and independently tested.

Sequencing adjustment from this review: after the minimal P4.4 builder works,
run P6's independent CPU wheel examples before completing P5's strict GPU
integration. Record public/private imports, installation failures, method code
and steps to correct output. Fix demonstrated API obstacles first. GPU parity
and independent GPU wheels remain mandatory afterward; no external adoption
is claimed from examples we write ourselves.

## P7 observed value boundary

The [committed resident matrix](../benchmarks/results/foundation/20260905T183820Z-3c245f2d/README.md)
passes quality but shows 13.899x default CUDA fit time versus the paired legacy
CUDA path. The 20% regression budget failed. P6/G3 proves independent installed
extensions; it does not justify replacing the legacy path. G4 retains negative
performance evidence and explicit profiler/peak-memory limits. G5 is still open.
The product mission remains calibration-first distributional risk, with this
API available for bounded research rather than a claimed general GPU speed layer.

The [isolated follow-up](../benchmarks/results/foundation/20260905T184856Z-dcd49569/README.md)
places most diagnostic wall time in tree construction and its extension boundary,
not objective math. The original uninstrumented timing verdict remains unchanged.
