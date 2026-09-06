# GPU Python foundation: medium execution checklist

> Historical P0–P7 execution and evidence record, no longer the active queue.
> Continue from [F0 in the new plan](agent-boosting-foundation-plan.md). The user
> clarified that the foundation is the product and permits incompatible redesign;
> this checklist's old interface and trainer-reuse requirements no longer constrain work.

Status: P0–P6 complete. P7.1 real-data quality passed; performance budget failed
(initially 13.899x, then 12.888x after fixed-slot optimization). P7.2 developer
materials complete. P7.3 diagnostics without concurrent sampling and design review
complete. Exact peak device memory and CUDA traces remain unverified. G5 external
adoption incomplete. See the [design contract](gpu-python-foundation-design.md).
P1 results: 12 local result-protocol tests passed; real single-T4 smoke 2 passed /
0 skipped, with wheel provenance and device invocation verified.
[P1 learning and raw results](../learnings/2026-09-05-foundation-p1-modal.md).
P0 results: CPU regression 749 passed / 34 skipped, focused loader regression
21 passed, lint/docs/packaging passed. See the
[P0 learning](../learnings/2026-09-05-foundation-p0-integration.md).
New modules, tests, and Modal entry points described below were plan targets,
not proof that they already existed when the plan was written.

## How to use this checklist

Work in dependency order on `codex/gpu-python-foundation-design`, one small task
at a time: read the call path → write the smallest failing test → implement →
verify → update learning → inspect staged diff → commit. Resolve routine choices
without requesting repeat approval for authorized work. The user authorized Modal
for this plan. Do not push, release, change main, or contact external authors.

When evidence falsifies a design assumption, provide a minimal reproduction and
update the relevant design section; do not conceal the issue by expanding scope.
Distinguish implementation, CPU verification, real-GPU verification, and unverified
external adoption in reports. If execution resources are limited, finish one
verified commit and identify the next task without declaring the full route complete.

## Planning snapshot

| Item | State at planning time |
|---|---|
| Original branch | `main`, clean |
| Current branch | `codex/gpu-python-foundation-design` |
| Original HEAD | `82cf1e25b21a69093e85a270af7eb93c9ae7aa19` |
| Fetched remote | `6ebe3a8ced0e621b17e3cf63e31721af58471053` |
| Local/remote-only commits | 12 / 9, not yet merged; excludes later design commits |
| CPU extension baseline | `tests/test_extensibility.py`: 35 passed, 1 GPU skipped |
| Modal | User authorized; no job, credential verification, or billing initiated in the planning round |

Recheck `git status --short --branch` before execution; the user may have edited
files. Merge the fixed SHA above for reproducibility. Review later remote commits
separately rather than unconditionally pulling the latest state.

## P0: Integrate the remote unified trainer and local fixes

**P0.1 — Freeze the baseline and resolve the merge.**

- Read local AGENTS, relevant learnings, and remote `planning/unified-engine-design.md`.
- Verify the branch contains the original local HEAD and design files, with no
  unexplained working-tree changes.
- Preserve both histories with merge, not rebase/reset:

```bash
git merge --no-commit --no-ff 6ebe3a8ced0e621b17e3cf63e31721af58471053
```

- Known conflicts: keep `CLAUDE.md` pointing to canonical AGENTS. Reconcile GPU
  installation docs with remote new models and local limitations; do not restore
  performance claims lacking raw evidence.
- Review `_persistence.py`, `_array.py`, `_core/_tree.py`, `_models/_boosting.py`,
  CI, ScoringBench, and examples semantically; preserve categorical/save/load/batch guards.
- Run P0.2 before committing. Combine the two steps into one verified merge;
  do not commit unresolved conflicts.

**P0.2 — Integration regression.**

Start with local fixes and remote new models:

```bash
OPENBOOST_BACKEND=cpu uv run pytest tests/test_categorical.py tests/test_persistence.py tests/test_batch.py tests/test_extensibility.py tests/test_formula.py tests/test_survival.py -n 0 -q
```

Then run the CPU suite, production lint, documentation build, and package build:

```bash
OPENBOOST_BACKEND=cpu uv run pytest tests/ -m "not gpu and not benchmark" --tb=short
uv run ruff check src/openboost/
uv run mkdocs build
uv build
```

First ensure dependencies with `uv sync --locked --extra dev`. Record concrete
platform restrictions on optional dependencies without claiming full verification.
Separate pre-existing failures from merge regressions. Fix relevant correctness
failures first; retain minimal reproductions and scope for unrelated legacy
issues. Do not advertise the foundation as usable before G0.

Acceptance: both parents are ancestors of new HEAD; AGENTS and all local fixes
remain; new models import and tests run. Suggested commit:
`merge: integrate unified trainer and local correctness fixes`.

## P1: Establish a traceable Modal test path that propagates failures

**P1.1 — Result protocol and test bundle.** Main files: separate
`benchmarks/foundation/` app to avoid old source mounts, new `tests/foundation/`,
required locks/test configuration, and `benchmarks/results/.gitignore`.

- First write local tests requiring nonzero exit for remote failure, timeout,
  missing reports, and skipped mandatory GPU tests.
- Manifest generation never reads or prints credentials; freeze source SHA,
  dirty state, wheel/data hashes, and argv.
- Package required tests/conftest/config and lock Linux/Python 3.12/CUDA 12
  dependencies including CuPy and xdist. Install with uv; do not point benchmark
  environments at workspace src.
- Explicitly allow only reviewed evidence under `benchmarks/results/foundation/`
  into Git. Keep temporary output ignored; do not broadly force-add all logs.

**P1.2 — Minimal real-GPU smoke.** New `::foundation_smoke`, one T4, 300 seconds,
retry=0.

- Verify the actual CUDA device, values and owner lifetime of Numba/CuPy zero-copy
  views, and installed-wheel provenance.
- Run a tiny builtin Normal fit/predict, returning environment, JUnit, and status;
  draw no speed conclusion.
- Test both successful and failing entry-point results locally; a “GPU available”
  boolean is insufficient.

Planned command, available after P1 implementation:

```bash
uv run modal run benchmarks/foundation/modal_app.py::foundation_smoke
```

Acceptance: locally retrieved real-GPU results match the source wheel and failures
propagate to CLI status. Record GPU time, commands, and limits. Fix specific
failures without adding GPU models or retrying long jobs.

## P2: Fix correctness first and freeze the old-path baseline

**P2.1 — Weighted constant-Hessian regression.** Main files: `_trainer.py`,
`_objectives.py`, and corresponding tests; source paths are relative to
`src/openboost/` and must follow the actual implementation.

- First failing test: fixed binned X and raw/grad, with zero and nonuniform
  positive sample_weight; compare weighted histograms, Newton leaves, and
  one-round predictions between CPU and native CUDA.
- Follow with end-to-end weighted Normal/Poisson fits; a mock proving argument
  forwarding is insufficient.
- Fix const_hess eligibility first: nonconstant Hessians must read the array.
  Initially hints may be disabled even for uniform weights. Inspect initialization
  and gradient weight semantics without changing every model's algorithm.
- If the static suspicion does not reproduce, explain why and add protection;
  do not turn a suspected issue into a claimed fixed bug.

**P2.2 — Capabilities, fallback, and seed.**

- Failing tests: custom objects sharing builtin distribution names must not select
  builtin CUDA mathematics; broad except must not swallow kernel RuntimeError;
  unsupported-capability fallback must be visible.
- Add scoped RNG to relevant fits; test same-seed repeatability, different-seed
  effects on actual sampling, and unchanged global NumPy RNG state. Reject
  unsupported MVP GPU sampling and connect old CPU sampling to the seed.
- Unsupported parameters fail before the first update without leaving fitted
  attributes that resemble a completed model.

P2 acceptance results: real T4 14 passed / 0 skipped, 12 baseline configurations,
24 fits. All three seeds passed quality, fallback, callback/eval, and bidirectional
save/load checks. [Baseline and raw results](../benchmarks/results/foundation/20260905T084129Z-2574e387/README.md).

**P2.3 — Freeze the integrated usable baseline.**

- Run Normal/Poisson gradient → split → leaf → raw → metric through
  `::foundation_correctness`. Separately verify CPU fallback, callback/eval
  download boundaries, and CPU/GPU saved predictions.
- Freeze real-data hash, splits, and seeds. Separate training residency without
  eval from end-to-end tasks with eval.
- Store default trainer/native correctness and cold/warm timing as the P7 baseline.
  Fix an incorrect or quality-failing baseline first; bad baselines cannot justify
  new-code acceptance.

Acceptance: relevant G0/G2 correctness passes, clean baseline source SHA, committed
results. Initial collection may use a bounded benchmark entry point; avoid
repeating the full matrix unnecessarily.

## P3: Freeze the minimal public contract on CPU

Completed: CPU facade, strict objective, explicit builder, per-channel schedule,
plugin-independent raw-inference persistence, and coefficient-aware early stopping.
182 relevant tests passed; 3 old long-running GBDT tests were interrupted and
are not counted as passes. A clean `f414b8c` wheel reproduced predictions exactly
in an isolated Python 3.12 environment.
[Contract, installation limits, and reproduction](../learnings/2026-09-05-foundation-p3-cpu-contract.md).
P3 supports CPU only; the native extension adapter is deferred to P5. The default
CPU builder temporarily rejects `reg_lambda=0/min_child_weight=0`. G1 passed
within this explicit boundary.

**P3.1 — Experimental facade and objective contract.** Main files: new
`src/openboost/experimental/__init__.py`, thin interface/type modules, `_trainer.py`.

- Failing tests: an independently implemented two-channel objective runs through
  the facade; invalid shape/device/keys, nonfinite values, invalid weights,
  missing capabilities, and unsupported parameters fail; existing adapters pass.
- Do not duplicate the training loop. Booster config, context, BuiltTree, and
  reports carry only the designed state.
- Freeze math conventions, float32/device/ownership, and a CPU oracle without near ties.
- Do not re-export all of `_core`; expose only types/functions actually used here.

**P3.2 — Builder and schedule dispatch.**

- Failing tests: explicitly supplied builders run even on native-eligible input.
  All two-channel/two-round coefficients apply once; training raw agrees with
  new predictions.
- Builders never receive writable raw references. Native adapters use
  `pred_gpu=None`; the trainer applies updates.
- Contract tests catch users reusing a buffer and overwriting a previous channel;
  do not conceal this with full host copies every round.
- Default schedules regress to old learning_rate. Custom coefficients must be
  finite, nonnegative, and complete across channels.

**P3.3 — Coefficient persistence and callbacks.** Main files: `_persistence.py`,
`_callbacks.py`, trainer/facade.

- Failing tests: nonconstant per-channel schedules preserve fit raw / predict /
  save-load equality; early-stop truncation and restore handle trees and coefficients.
- Saved experimental models support raw inference in a clean CPU environment
  without extension packages; do not pickle training plugins.
- Old files without coefficients load using old learning_rate. The experimental
  facade rejects old categorical state; retain the shared legacy loader's warnings.
- Round-trip every touched shared tree state, especially missing/categories/specialized leaves.

Acceptance: experimental CPU contract implemented, no relevant default-model
regression, G1 passed. Commit each P3 step separately; do not defer prediction/
storage semantics until GPU completion.

## P4: Build the device path one primitive at a time

Suggested new file: `src/openboost/_core/_batch_primitives.py`, plus existing
CPU/CUDA backends. Filenames may follow code organization; do not arbitrarily
expand responsibilities or interfaces.

**P4.1 — Batch histogram / CPU oracle. Complete.**

74 relevant CPU tests passed; clean `cf61611` wheel on real T4: 3 passed / 0 skipped.
Maximum G/H errors against an independent sample oracle: 9.835e-7 / 1.252e-6;
counts exact. [Evidence and limits](../benchmarks/results/foundation/20260905T150940Z-a5c80f7f/README.md).
This commit completed only histograms; split/routing verification follows in P4.2.

- Use direct sample sums as an independent oracle, not the production histogram
  function to generate expected values.
- Test weighted/zero-weight, empty nodes, inactive slots, constant features,
  reserved missing bin, and memory budget.
- Keep GPU aggregation on device, bypassing legacy host dict wrappers. Counts
  remain separate from H.

**P4.2 — Split / routing. Complete.**

`b75b95a`: 90 relevant CPU tests passed; real T4 4 passed / 0 skipped. Independent
oracles verified topology, exact ties, gain bounds, actual sample routing, and
next-level statistics. [Raw evidence and limits](../benchmarks/results/foundation/20260905T151819Z-e5eb30b7/README.md).
Numeric L2 with positive-curvature children; Booster GPU integration still awaited
P4.3/P4.4/P5.

- Exhaustively enumerate valid splits on tiny matrices; check gain,
  min_gain/min_child_weight, tie order, and no-valid-split behavior.
- Device-partitioned node IDs match the CPU oracle; next-level histograms use
  actual routed rows.
- Never approximate children by scaling parent histograms. Use fixed complete
  slots by default, explicitly masking empty slots.

**P4.3 — Leaf reduction / rule. Complete.**

`4da7f4b`: 107 relevant CPU tests passed; real T4 5 passed / 0 skipped. Per-sample
reduction references, weighted Newton/clipped leaves, next-round gradient changes,
and rule ownership passed. [Raw evidence and limits](../benchmarks/results/foundation/20260905T152639Z-62d96727/README.md).
CPU uses the real trainer with a root builder. GPU is a two-round primitive
composition at this point, not a complete Booster.

- Real GPU sum/reduction; independently check nonuniform weights and zero-effective nodes.
- Public leaf rules return same-device arrays. Bounded leaves change both leaf
  values and next-round gradients.
- Test no grad/hess/node-ID downloads or full host histograms.

**P4.4 — LevelWiseBuilder assembly. Complete.**

`e02403b`: 120 relevant CPU tests passed; real T4 6 passed / 0 skipped.
[Raw evidence](../benchmarks/results/foundation/20260905T163823Z-7b16b556/README.md)
covers independent whole-tree references, device cache lifetime, two rounds/two
channels at 16/4097 rows, and CPU-loaded predictions. This builder is explicitly
selectable; preserve the default CPU builder's existing parameter boundary.
GPU checks directly compose builders; full GPU Booster.fit still awaits P5.

- Compose the primitives into a selectable experimental builder; download only
  compact tree arrays after completion.
- Keep device prediction caches; default-stream/view lifetime tests include
  prediction after fit and temporary-buffer release.
- Verify one tree, then two rounds/two channels on CPU/CUDA before scaling n.
  Do not add vector leaves or leaf-wise growth in this step.

Each P4 GPU task needs real-GPU parity before being marked passed. Reuse one
image and small selectors rather than running the full Modal suite for each edit.
Without GPU availability, finish CPU/test preparation but do not skip the device
gate and claim completion.

### Move the usability check earlier

Goal-review sequencing adjustment: after P4.4's minimal builder works, run the
**independent CPU-wheel** parts of P6.1/P6.2/P6.3 before strict P5 GPU integration.
GPU acceptance remains unchanged. Record nonpublic dependencies, method code
size, installation obstacles, steps to correct output, and plugin removal after
saving. Fix demonstrated core/interface obstacles before adding GPU features.
Self-authored CPU packages are not external adoption: G5 remains incomplete.
Then complete P5 and P6 GPU verification.

## P5: Integrate strict GPU execution and reports

**Core execution verified; profiler-trace gap retained.** `8c34b26` implementation,
`4299858` extension tests: real T4 7 passed / 0 skipped. Normal/Poisson ordinary/
natural gradients, default/external builders, weighted two-channel schedules,
CPU loading, and error rollback passed.
[Raw evidence](../benchmarks/results/foundation/20260905T180000Z-7d73ba83/README.md).
Full CPU regression: 904 passed / 34 skipped; final focused tests: 89 passed.
nsys was unavailable, so there is no profiler trace. Report only named transfer
wrapper checks and device-copy cost. GPU eval/callbacks/early stopping are
explicitly rejected. Next: P6 independent GPU wheels.

**P5.1 — Actual dispatch and residency.**

- One trainer runs default/external builders; check all strict-GPU capabilities
  before training.
- Keep objective/raw/y/weights and updates on device. Report tree finalization
  and explicit eval/output copies.
- Verify no native bypass of explicit extensions, duplicate raw updates, or
  duplicate sample weighting.
- Host-transfer wrapper tests cover internal library paths; add one small
  profiler run for external CuPy/internal kernels. If unavailable, state the
  gap rather than claiming proof of no transfers.

**P5.2 — Regression and negative tests.**

- Compare default numeric NaturalBoost CPU/CUDA; same-name custom objectives,
  broken kernels, and wrong-device outputs fail appropriately.
- Reject unsupported categorical/missing/exposure/eval/sampling/regularization
  or use the declared complete CPU fallback. Never silently honor only some
  parameters. Precisely report each stage of retained legitimate legacy hybrids.
- GPU fit → save → CPU load prediction, nonconstant schedules, early-stop state,
  and existing persistence regression.

Acceptance: G3 core path passes and fit_report_ agrees with actual trace; fix
necessary compilation/lifetime issues. If default execution slows significantly,
preserve correctness and enter P7 profiling, without hiding fused updates as
builder side effects.

## P6: Two real installation boundaries and reproducible examples

**Independent CPU/GPU installation verified.** GPU `43fcda3`: real T4 3 passed /
0 skipped; 8 training combinations and standalone GPU examples passed. A new
process exactly reproduced 9 models after uninstalling both plugins.
[GPU evidence and initial startup failure](../benchmarks/results/foundation/20260905T181351Z-1aee9568/README.md).
Both packages at 0.2.0 declare CPU/CUDA, with no core changes. Maximum GPU raw
prediction error 3.58e-7. Updated independent CPU install: 7 passed; relevant
regression: 90 passed. Initial CPU verification follows.

**Initial CPU verification.** `3f8addd`: two independent wheels use only public
APIs in a fresh venv outside the repo. Five independent math/composition tests
passed; a 64-row weighted public example ran. Six models reproduced exactly
after uninstalling both plugins and restarting Python. Relevant regression:
76 passed. [Full evidence and installation obstacles](../benchmarks/results/foundation/p6-cpu-3f8addd/README.md).
Method source: 89 + 22 lines, no core changes. Development Numba/llvmlite source
installation failed on Intel macOS; examples pin installable CPU dependencies.
This initial run claimed CPU only; external adoption remains unverified.
P5/P6 core execution and installation requirements are verified. P7 evaluates
engineering value against preregistered thresholds.

**P6.1 — normal_fisher package.** Implement independent objective/schedule from
A/C, with finite-difference gradients, analytic Fisher references, and two-round
updates. Use public types/NumPy/CuPy; a registered builtin alias is not an external method.

**P6.2 — bounded_leaves package.** Implement leaf rule B and prove clipping acts
during training on device, changes later raw/gradients, and survives storage.

**P6.3 — Wheel conformance.**

- Install OpenBoost and both independent wheels in fresh venvs; test outside the repo.
- Check actual module paths; extension imports are restricted to documented public modules.
- Run A+B+C together to verify coexistence, and separately for diagnosis.
- Remove extensions and load saved models for CPU raw inference; record build/lock/source hashes.
- README covers installation, execution, CPU oracle, Modal GPU checks, and actual
  limits. Execute examples in tests.

Acceptance: both packages work without private imports, forks, or manual core
changes; all three extension points change intended behavior. This establishes
technical conditions for adoption, not external adoption itself.

## P7: Measure value, align documentation, and deliver

**P7.1 — Matched-quality performance records.**

- Compare P2 baseline and new defaults on the same Modal GPU, data/split/config.
- Separately record cold startup/compilation, warm median, end-to-end fit/predict,
  peak device memory, transfers, and per-seed NLL/CRPS/coverage.
- Apply preregistered thresholds without changing seeds, removing datasets, or
  relaxing limits after failure.
- Profile standard-path regression above 20%; after fixes rerun only affected
  comparisons, without searching other GPUs for better numbers.
- External extensions compute different algorithms: report quality and cost
  separately without requiring identical runtime to defaults.

**P7.2 — Developer materials and documentation.**

- Provide one page on implementing objectives/leaves/schedules, capability matrix,
  actual failure examples, and reproduction commands.
- Prepare an external author task without its answer and a record sheet: time
  to first correct output, help requests, private imports/core changes, GPU results,
  and willingness to depend on OpenBoost in an independent package.
- Do not change the mission to predeclare platform success; explicitly leave G5
  incomplete without external attempts.

**P7.3 — Final verification and conclusions.**

- Run CPU regression, required CUDA correctness, changed-file lint, docs build,
  and wheel-install conformance.
- Freeze the relationship between raw artifacts and final source SHA. Later
  changes to relevant code cannot inherit earlier results automatically.
- Update learnings with passed gates, failures, unsupported parameters, and next
  external verification action.
- Leave an explained workspace state and identify implementation commits; do not
  automatically push or merge main.

## Evidence file convention

Suggested frozen structure; add precise ignore exceptions for new directories:

```text
benchmarks/results/foundation/<run_id>/
  manifest.json        # source/wheel/data/env/argv/device provenance
  results.json         # per-test or per-seed raw results, includes failures
  junit.xml           # correctness jobs
  transfer_summary.json  # only when actually measured; declares coverage
  README.md           # reproduction, source commit, interpretation/limits
```

Keep necessary failure context without sensitive information; never commit full
credentials/environment dumps. Separate collection time from measured time.
Self-reported GPU labels do not establish device execution. Commit implementation
first, run from a clean SHA, then commit artifacts separately to avoid circular
source-hash references.

## Startup instructions for the medium model

> Continue `codex/gpu-python-foundation-design`. Read AGENTS.md,
> `planning/gpu-python-foundation-design.md`, and
> `planning/gpu-python-foundation-execution.md`. Execute from P0, reusing the remote
> unified trainer and preserving local correctness fixes. For each small task,
> write an independently justified failing test, implement, verify, update learning,
> inspect the diff, and commit. Modal is authorized for planned single-GPU checks:
> traceable smoke, then correctness, then benchmarks. Do not expand to multi-GPU
> or general training graphs, skip persistence, or count CPU fallback as GPU success.
> Do not push or merge main. Record falsifying evidence and revise the minimal
> design; distinguish technical acceptance from external adoption.

## P7 engineering value review (2026-09-05)

The [fixed matrix](../benchmarks/results/foundation/20260905T183820Z-3c245f2d/README.md)
preserves 12 cells, 48 timed fits, all seeds, and failure criteria. New default
GPU warm fit: 2.078694 s versus legacy GPU .149556 s, or 13.899x. Quality passed
for every fit/seed. Independent A+B+C proper scores were worse; coverage closer
to 90% alone cannot establish superiority. Original profiling ran concurrently
with memory sampling and contaminated host attribution. Collect isolated
diagnostics without replacing the original timing matrix.

Conclusion boundary: retain the experimental extension API without replacing
the old GPU path or claiming speed/cost advantages. Further optimization must
preserve borrowed inputs, cached-prediction validation, and final quality.
Identify synchronization/validation costs before optimizing. Current evidence
does not support expanding multi-GPU, train-many, or a “general GPU Python
foundation” positioning. The historical product direction remains
calibration-first distributional risk modeling.

The [author task and record sheet](../examples/extensions/AUTHOR_TASK.md) are ready,
but no external attempts or independent dependency willingness are evidenced.
G5 explicitly has not passed. The next adoption check is an actual trial, not
more self-authored examples. No outreach, push, publication, or external leaderboard update occurred.

The [follow-up without concurrent sampling](../benchmarks/results/foundation/20260905T184856Z-dcd49569/README.md)
had 3 passed / 0 skipped and unchanged quality. Of the default path's diagnostic
2.260 s fit, tree/session boundaries used 2.123 s (builder internal 1.588 s),
and objective boundaries .0885 s. Investigate tree construction, repeated
validation, and synchronization before changing objective mathematics as the
main performance remedy. Nested synchronized measurements add overhead and
cannot replace uninstrumented results or predict an optimization's benefit.

Final verification reuses P5's 904 CPU passes, P6 independent installation, and
real-CUDA conformance for the same unchanged core/plugin wheels. The added
harness passed 27 focused tests, production/changed-file lint, docs build, and
offline checks of both new artifacts. G4 performance budget failed. Exact peak
memory and full transfer traces remain gaps, not passes.

### Fixed-slot optimization (2026-09-05)

`81acf0b` preserves all checks while replacing dynamic boolean compression in
split application with fixed-shape selection and parent mappings for the frontier.
[Measurements](../benchmarks/results/foundation/20260905T193308Z-5ebd75ab/README.md):
warm fit 2.079 -> 1.868 s; paired legacy-normalized ratio 13.899x -> 12.888x.
This is a partial improvement, not the 1.2 budget. Quality, real-CUDA conformance,
and independent-wheel loading passed. Retain the change; next investigate
tree/session validation and synchronization costs more precisely.

### Bigger-goal review: immediate queue changed

An earlier [impact/adoption/value review](impact-adoption-value-next.md) moved
the queue to ScoringBench. The user's subsequent clarification supersedes that
ordering: the foundation is the product, use cases drive its abstractions, and
backward compatibility is not required. Continue with the
[new F0–F5 plan](agent-boosting-foundation-plan.md); retain the old measurements.
