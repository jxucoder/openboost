# Sprint 085: Return to the programmable-foundation hypothesis

Status: user-approved sequencing amendment, 2026-09-07; storage/aggregation are verified,
training and independent measurement remain pending. This card redirects 069
and bounded 078 work;
it does not create a new v1 scope or certify a phase exit.

## Goal and retrospective

OpenBoost helps researchers and agents build correct, useful algorithm changes
through readable public components, then execute those compositions practically.
CPU is the semantic reference and development path. CUDA supplies accelerated
execution under the same contracts. Beating mature libraries' CPU implementations
is not a separate product goal.

We have twelve CPU recipes, installed extensions, state/persistence checks and
1125 passing CPU tests (one Linux-only skip). None demonstrates independent
benefit to an author. The paired A6 change preserves exact results and reduces
observed fit time from 992 to 757 seconds. Comparator probes finish in seconds.
These separate observations expose a practical runtime concern, not a precise
matched-quality ratio. The evidence is useful; extending CPU qualification as
our main activity would continue postponing the hypothesis we need to test.

## Approved change to sequencing

Pause the next OpenBoost shared configuration-05 CPU probe, wider 400-job A6
execution and speculative CPU optimizations. Keep all freezes, failures and raw
artifacts. Sprint 070 continues only for correctness, isolation and evidence
requirements needed by these two bounded workstreams; its full ledger remains open.

Allow exploratory F2.1 authoring and bounded B12/F3.1 feasibility to proceed
alongside each other before formal F2/E5 completion or full E3 CPU searches.
Audit the relevant 065/068 ownership/transaction contracts on the actual device
path before remote execution. Do not reinterpret this exception as an F2/F3 exit.
Full required R1/R4/R5/R6/R8 CUDA coverage, R9, P7 reproduction, E4, every A1–A13
and all R/C/E obligations remain required in their existing cards. Scalar CUDA
cannot substitute for Normal/P7, and these development tasks are not held out.

## Slice 1: Make authoring evidence possible (069)

Use existing public D1 expectile as a control and D2 cohort-constrained split
selection as the deep change. Preserve the task definitions and independent
mathematics in planning/foundation-tasks.md. D2 must select the best feasible gain
using separate cohort information sums; total child weight is not a substitute.
No new objective catalogue or task invention is needed.

Deliver an executable, source-pinned packet: installed wheel, initial public docs,
task cards, verifier identities, appropriately audited incumbent path, prompts,
model/settings/tools, budgets and an append-only accounting record. Built-ins,
hooks and incumbent source edits are legitimate alternatives. Audit an appropriate
D1 incumbent and D2 source/hook path before freezing the comparison; do not select
a weaker opponent just because its wrapper is convenient.

Use the existing 1800-second / 20000 generated-token per-attempt ceilings. Before
an independent attempt, demonstrate actual evaluator access denial and enforced
budget exhaustion. Record setup, active/blocked time, generated tokens, assistance,
failures and core/private edits. Unknown token usage is an explicit accounting gap,
not zero or a pass. Synthetic accounting events verify plumbing only.

Acceptance: one complete accounting/isolation smoke and a reviewable frozen packet.
Then an authorized independent author must deliver a correct D2 extension through
public components, tested against the existing oracle and fresh persisted replay;
record failures and hints without recasting retries as clean attempts. D1 remains
a control. Formal E5 still needs its original cohort, repeats and held-out cases.
No new agent is launched by this planning amendment. Do not inspect H1/H2 contents.

## Slice 2: Build the scalar device path (bounded 078)

First test: strict CUDA execution currently fails as unsupported. Implement
weighted squared-error numeric/missing training with explicit device/workspace
ownership, then two rounds of resident gradients, named statistics, candidate
selection, routing, leaves and raw updates. Keep readable Python orchestration.
CPU preparation plus one declared upload is allowed; bulk round work cannot
silently fall back to CPU. Small host decisions must expose synchronization costs.

Reuse the construction design's CuPy array ownership and initial numba-cuda kernel
choice; pin compatible dependencies before execution. Do not build a separate
trainer that bypasses public components, atomic state, stopping or persistence.
An ownership/API counterexample triggers reflection before adding more kernels.

Bound the first device experiment to one T4, float32, at most 8192 rows, 32 features,
32 bins, depth 2 and two rounds, with missing values, weights, ties and zero-mass
cases. Freeze concrete fixtures and E1 tolerances before implementation/results.
At most two declared device runs, each capped at 900 seconds (30 GPU-minutes total),
no automatic retries. Record allocation limits, actual peak memory, driver/CUDA,
JIT policy, transfers and synchronization. Failure consumes this budget. This
synthetic feasibility budget does not change the formal T4 E4 protocol.

Acceptance: independent CPU/reference agreement for gradients, statistics, split
optimality/ties, routes, leaves, raw updates, predictions and task metrics. Counts
and routes are exact; numeric tolerances follow E1. Fresh CPU inference reads the
saved model without GPU/training plugins. A skipped/unavailable GPU is not_run.
No GPU performance claim follows two-round correctness.

## Slice 3: Exercise the programmable device boundary

Carry the same D2 extra-statistics/feasibility composition onto the scalar device
path. It must execute real cohort reductions and constrained candidate selection
on the device, pass the D2 best-feasible/no-feasible counterexamples and preserve
persisted inference. A Python callback that downloads bulk candidate arrays and
performs the algorithm on CPU does not pass this device extension check.

Development task reuse is deliberate; freeze author attempts before changes
informed by their results, and distinguish revised cohorts. Use a bounded scaling
check only after parity: compare the same composition's CPU/CUDA end-to-end cost,
including setup/JIT/transfers and failures. Preregister that workload and budget
at the next reflection, rather than deriving a speed gate from these fixtures.

## Reflection and stop conditions

Reflect after each slice and after at most three implementation commits. Report:
what an independent author could build, where public boundaries helped or forced
core edits, what actually remained on device, correctness failures and practical
cost. Do not substitute test counts, harness count or kernel speed for these answers.

If author accounting is unavailable, retain the gap and continue independent GPU
preparation. If CUDA needs semantic changes, fix the shared contract and CPU oracle
first. If performance is poor, profile that demonstrated composition before
optimizing. Resume wider CPU searches only when they answer a specific correctness,
quality or cost question for the product, with a recorded decision and unchanged
formal budgets. The paused deeper CPU probe is not a failed or passing trial.

Original next action was to assemble the 069 D1/D2 accounting packet and audit
the device ownership seam. The author view and storage implementation/evidence
now exist; full accounting and training remain open. Follow the
[086 execution plan](086-next-execution-plan.md) for the next aggregation slice
and retrospective. This planning record itself dispatches no author or device run.


## Device budget update

Run 1: clean `77aa105`, twelve T4 storage tests pass; see
[078 evidence](../benchmarks/v1/evidence/cuda-storage-078/README.md). After that
allocation one run remained; it verified storage only, not two-round training.

Run 2: clean `ad2f4e6`, 33 T4 storage/aggregation tests pass; see
[aggregation evidence](../benchmarks/v1/evidence/cuda-aggregation-078/README.md).
No retry. **Both runs are consumed; zero remain.** 078-A passes and the planned
retrospective is reached. Candidate selection, D2 feasibility, GPU training and
independent author benefit remain open. Further hardware verification requires a
new concrete workload/budget; 086 retains the subsequent construction sequence.
