# Sprint 062: CPU exit audit and GPU entry decision

Reviewed revision: a2a03e4. Status: review complete; no phase exit declared.

## Purpose and method

Reconcile implementation with F0–F3/B11–B13, instead of treating adapter count as
completion. Read the current worker, installed scheduler verifier, search harness,
construction design and E0–E7 criteria. Retain earlier raw evidence and sealed
held-out boundaries. This is a source/evidence review, not a new test or benchmark.
Latest completed CPU regression remains Sprint 061: 923 tests.

## What is built

| Layer | Implemented evidence | Remaining acceptance boundary |
|---|---|---|
| C1 data/targets | Numeric/mixed inputs, training preparation, class/weight/offset/structure and event-right roles | Aggregate current R/C/A evidence into the declared gate ledger; source identity is not access isolation |
| C2/C3 ops/trees/leaves | Named statistics, candidate/feasibility/scoring/routing, three growers, scalar/vector/residual leaves; installed D2/D3 probes | Current coverage reconciliation and formal author evidence; no GPU ops |
| C4 runtime | Immutable transactions, deterministic keyed RNG, validation stopping, sequential M=1/8/32 and external result contract | Complete explicit installed D5 mutation probes; no device workspace/stream/transfer implementation |
| C5 persistence | Raw models and specialized output/scale/composition inference; current workers replay in fresh processes | Complete current workflow/installed delivery evidence; no training-resume or GPU parity claim |
| R1–R8 recipes | Twelve public CPU recipes plus frequency-severity composition; recent applications needed no core edits | Scope limits remain (fixed AFT scale/event-right, quadratic query pairs, no calibrated compound distribution) |
| R9/C7 workflows | Shared preparation, sequential scheduling, strict selection/release infrastructure | Current A13 evidence is synthetic A6 search; real searches and joint composition selection unfinished |
| C6 author evaluation | Installed D1–D4 exploratory packages and partial D5 evidence | Formal E5 cohort, independent attempts, cost accounting and held-out results absent |
| CUDA | Construction design only; RunContext rejects non-CPU | B12 device operations/recipe parity and B13 batching/cost are unimplemented |

These are implemented capabilities, not a claim that CPU E0/E1/E2 pass in full.
The historical Sprint 035 gap table is superseded by this review where newer
sprints supply evidence; its old missing-objective/stopping statements are not
current blockers.

## Application position

| Applications | Current evidence | Still open |
|---|---|---|
| A1/A2/A3/A5/A6/A7/A8/A10/A11 | Five frozen current validation fits and fresh inference per application | Full baseline-matched quality/search, applicable domain metrics and source obligations |
| A9 | Direct annualized aggregate and matched paid-event composition pass five folds | Joint aggregate selection, complete searches, quality and calibration claims |
| A12 | Five Formula folds with explicit age separation and fresh inference | Control comparisons, interpolation/extrapolation and full quality/search |
| A4 | Current pairwise/lambda adapter; weighted synthetic direct/fresh checks | MSLR data/agreement, preprocessing freeze, official-query-fold binding and real fits |
| A13 | Current synthetic A6 16-trial selection and sealed release; M1/8/32 semantics | Real-data complete search/release workflows and full-set device cost |

All A1–A13 remain required. No source was substituted, and no application was
removed. The Veteran source-license gap is not closed by its passing fits.
Short four-round validation jobs are not fair quality searches. No full E3 or E4
claim can be derived from these artifacts.

## Finite remaining work, in order

1. **Complete explicit installed D5 development probes.** Extend the existing
   examples/v1_extensions/scheduler_checks.py workflow, not a new trainer:
   - Same seed/different stable run IDs must produce distinct keyed streams;
     each stream must remain exact under permutation and retry.
   - Changed feature content with unchanged row IDs must invalidate old prepared
     data; altered row identity must also be rejected. Fresh preparation must
     recover and match independent direct execution.
   - Retain mixed K=1/2, M=1/8/32, different stopping and failure artifacts already
     present. No private imports/core edits; build and exercise installed wheels.
   These explicit cases are absent from the current installed verifier; this is
   not a claim that the underlying runtime behavior is absent or incorrect.
2. **Complete source/workflow prerequisites.** Resolve MSLR access/agreement and
   freeze official folds before real A4 execution. Preserve source terms. Finish
   a real A13 selection/release run from existing supported frozen data, then
   route remaining output schemas, including composition's joint selection,
   through the declared workflow. Retain failed trials; no budget retuning.
3. **Reconcile CPU E0/E1/E2 and F0.3 readiness.** Produce current item-to-artifact
   records for R1–R9/C1–C7/A1–A13, independent verifiers and fair comparator paths.
   Mark missing records as unverified. Do not relabel historical or synthetic
   results as real/current. Resolve judges, source and execution-budget gaps.
4. **Freeze and execute F2/E5.** First verify accounting with the required smoke.
   Freeze version, model/reasoning/tools/docs/budgets and independent judge access.
   Run all five development types and two held-outs with three attempts per arm;
   30 minutes/20k generated tokens per attempt, failures retained. Apply the
   existing correctness/cost thresholds unchanged. Do not inspect sealed tasks
   for implementation planning. Existing exploratory scripts are not a cohort.
5. **Enter B12/F3 after its prerequisites, or adopt an explicit amendment.**
   No more built-in objectives or unmeasured CPU optimization are scheduled.
   Correctness failures and demonstrated consumer blockers still take priority.

Full baseline quality E3 and external adoption E7 remain v1 obligations, but
must not be invented as prerequisites for every initial GPU experiment. The
actual current ordering is F0/F1/F2 before F3, including the existing approved
CPU overlap; this review does not silently widen that authorization.

## GPU decision and first slice

**Decision under the current plan: F3 is not started or accepted.** The first
next implementation is the installed D5 slice above. Starting CUDA before the
formal CPU/author prerequisites would be a sequencing change; this review does
not claim such a change has already been approved. If the user chooses that
route, record a bounded B12 feasibility overlap, with all CPU/E5 obligations
retained and no interface-freeze or performance claim.

The first B12 deliverable is one resident squared-error vertical path, followed
immediately by a two-parameter Normal path. Use the design's CuPy-owned arrays
and streams with numba-cuda kernels initially; change the kernel technology only
from measured evidence. Host metadata remains separate from device targets,
codes, raw predictions, statistics and row routing. Initial CPU preparation plus
one upload is permitted with its complete cost recorded.

Acceptance for that first slice:

- Real CUDA hardware; no skipped job counted as success and no CPU fallback.
- At least two rounds with declared precision/tie policy: compare gradients,
  curvature, histogram totals, candidate decisions, routing, leaves, raw updates,
  final predictions and task metrics against CPU/independent fixtures.
- Preserve transaction/stop/RNG semantics; expose synchronization and failures.
- Record preparation/upload, compilation cold/warm policy, transfer bytes,
  synchronization, workspace peak and complete fit/predict cost.
- Persist output for CPU fresh inference. Single-recipe parity does not establish
  multiparameter, nondefault extension, train-many or E4 acceptance.

Only then expand required device compositions, a nondefault public component and
actual backtracking. B13 groups compatible M=1/8/32 runs after sequential device
parity; no fusion, multi-GPU or Ray claim precedes its evidence.

## Reflection and verification

Sprints 053–061 mostly added adapters and evidence rather than core primitives.
That supports the boundary for these internal consumers, but is not evidence of
lower external authoring cost. The next work changes from application plumbing
to explicit scheduling probes and phase acceptance. Count growth must not become
a substitute for the foundation's central hypothesis.

At review start, the local branch was clean and twelve commits ahead of origin/main. No push,
network retrieval, new model fits, held-out inspection or threshold changes were
part of this review. Documentation and whitespace checks passed. See
[learning](../learnings/2026-09-06-v1-cpu-exit-review.md).
