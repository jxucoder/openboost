# Sprint 088: Resident scalar rounds and owned transactions

Status: local construction after the user's continuation at the 087 retrospective.
Baseline `e1a9c20`. Implements 086's 078-C. No GPU allowance is available: all
three prior invocations are consumed. No new agents, CPU search or phase exit.

## Decision and public boundary

Verified split operations are useful only if an algorithm can use them across
real rounds without breaking state ownership. Keep CPU preparation and small
topology/control metadata explicit. Targets, offsets, gradients, predictions and
raw updates remain resident. Model export is a separate explicit CPU boundary.

Construction is divided into three independently reviewable local slices:

1. Freeze a float64 original-row tree/round oracle, compare existing public CPU
   behavior, and define device ownership before kernels. The first failing check
   is the absence of a resident scalar runtime.
2. Add resident squared objective operations and public depthwise tree composition,
   inference and export. Grow through existing candidates/scores/masks/routes/leaves.
   Keep a supplied device scoring, legality or leaf operation effective; reject
   incompatible inputs. Regularization and depth remain explicit configuration.
3. Add a public scalar run with initialize, fields, propose, resolve, raw snapshot,
   model export and release. Compose a two-round recipe and real-device verifiers.
   Freeze a complete installed-wheel run package before requesting more hardware.

Use modules for objective/tree/runtime responsibilities, with public operations
and registered opaque records. Do not feed device buffers into NumPy-owned CPU
AcceptedState or Model constructors during training. Existing CPU interfaces keep
their explicit CPU execution contract; this slice does not extend run_many.

## Ownership and policy contract

- A run binds an execution context, stable run ID/seed, training/validation
  problem identities and one fitted Binning. Validation reuses those training cuts.
  Numeric scalar squared targets are the initial supported objective; incompatible
  targets/classes/structure and invalid float32 inputs fail explicitly.
- Accepted and proposal raw arrays are private allocations, never public workspace
  handles. A public raw snapshot is an independent device copy. Proposal evaluation
  updates accepted raw with only the new tree contribution, without ensemble replay.
- Proposals snapshot the supplied learner. Acceptance makes independent raw buffers;
  releasing a proposal or caller-owned tree cannot invalidate committed state.
  Immutable internal tree terms may be shared with reference accounting. Releasing
  a prior state must leave newer state and its best model intact.
- Rejection returns the identical accepted record. Foreign, forged, stale/released
  proposals and non-boolean decisions fail. A failed operation discards new work
  without changing accepted arrays, model, version, scores or keyed RNG behavior.
- The algorithm supplies acceptance; strict validation improvement selects best
  independently. Existing StopState observes once per outer round, including a
  rejection. Trials are not stopping observations. No opaque internal policy may
  silently replace a caller's decision or extension operation.
- Keyed RNG uses run ID/seed/logical round/component/purpose, so a same-step retry
  reproduces draws and distinct runs remain independent. The deterministic recipe
  does not manufacture a sampling claim from those checks.
- Default recipe history retains scalar diagnostics only. Release superseded raw
  states and proposals; final/current/best tree storage is O(T), raw storage O(N).
  Explicitly retained user snapshots cost additional storage. Context close is an
  explicit lifetime boundary, not hostile-callback/process isolation.
- CPU artifact export reuses validated Tree/TreeTerm/Model records, with float32
  leaves represented exactly as float64. Saved models must predict in a fresh CPU
  process without CUDA packages or training plugins. Offsets remain supplied at
  prediction, not absorbed into the learned raw model.

## Frozen cases and acceptance

| Case | Required observations |
| --- | --- |
| C1 weighted/missing | Eight training rows from 086 with unequal/zero weights, nonzero offsets and a distinct validation set; depth 0/1/2, two rounds, learning rate 0.5, lambda=1 |
| C2 D2 | Six-row targets negate 087 gradients, so base=0; alternating independent cohorts; minimum 0/1/2, renamed/reordered fields; constrained first split differs |
| C3 validation conflict | Training C2 with validation target sign reversed; finite accepted steps worsen validation so best remains the base model |
| C4 transactions | Rejection and same-step retry, wrong parent/run, forged/released records, proposal/tree/raw-snapshot release and context lifetime; prior raw/model/best/stop/RNG remain intact |
| C5 policy/retention | Fixed acceptance, strict training-improvement backtracking, patience and zero rounds; release superseded state, preserve retained snapshots, bounded repeated rounds |
| C6 failure/export | Invalid/nonfinite/overflowing scalar inputs and coefficients, allocation failure, mismatched preparation, full numeric/missing model round trip and fresh CPU inference |

The independent oracle loops original rows and exhaustively enumerates conditions;
it imports no production training or device operation. Compare public CPU behavior
before implementation, then GPU gradients, weighted fields, each selected split,
leaves, both raw updates, validation predictions and losses. Exact integer topology
and routes; float32 rtol=1e-4/atol=1e-5; metric difference at most
1e-3 * max(1, abs(CPU metric)). No added near-tie band and no hardware simulator.

Retain all previous 88 CUDA regressions in the next concrete package. New fixtures
remain small and deterministic; no new objective catalogue, performance benchmark
or full application evaluation is hidden inside this slice. A proposed future run
is one T4 invocation, at most 900 seconds / 600-second tests, 16-MiB private pools,
zero retries. These proposed bounds are not authorization to dispatch or upload
new files. Record exact files/cases/dependencies before the next approval request.

Reflect after every three implementation commits, ownership counterexample and
hardware result. Construction/local collection is not CUDA training acceptance.
All R/C/A/E gates, 065/068 obligations, independent author accounting, P7 and E4
remain required beyond this bounded scalar scope.

## Local fixture result

All 37 CPU oracle cases pass before device implementation: two rounds across
three fixtures, three depths and four cohort-minimum settings, plus a hand-checked
D2 split change and validation conflict. These verify the frozen expected behavior;
they do not execute resident CUDA training.

## Local objective/tree construction

`3561738` commits the design and independent reference first. Resident scalar
base, gradients, named fields and loss now compose with public depthwise tree
growth, prediction, independent copy and CPU artifact export. DeviceTree owns
packed values and topology; training data/fields/callback workspace may be released
without invalidating the tree. Root positions are generated on device. Fitted
binning identity is checked across distinct training/validation preparation.

Supplied scoring, legality and leaf operations remain effective. A separate
nonempty-child mask preserves tree structure even when supplied scores prefer an
empty-child candidate. Temporary callback allocations are scoped and discarded;
preexisting caller buffers remain owned by their caller. Forty-two real-device
cases collect locally; no CUDA execution or new upload has occurred. Accepted/
proposal integration and the scalar recipe remain the next construction slice.
