# Sprint 070: Coverage ledger and trustworthy selection judging

Status: in progress; evaluator-freeze integrity slice implemented, broader gates open. Mapping: N4 / B02–B11 / C6 / F0.3 and CPU F1 entry/exit auditing.
Entry: inventory/source work can start now; practical execution preflight follows 068.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first failing checks

Required evidence cannot disappear from a report, and candidates cannot access
sealed test labels or escape frozen resource caps. First feed the existing judges
missing/duplicate/stale required cells and an omitted application. Separately run
fault workers that attempt to read test labels, exceed RAM/time, produce NaNs,
claim a wrong backend, or exit unsuccessfully. Record which failures are already
handled and which demonstrate a real gap.

## Work

- Join every R1–R9/C1–C7/A1–A13 and E0–E7 obligation to revision, verifier, raw
  artifact, environment, status and next action. Derive the expected set from the
  frozen protocol; producer omission cannot shrink it. Manual rows are navigation.
- Extend existing integrity, search and quality tools with independent gate checks;
  do not create another universal runner. Distinguish integrity_pass from actual
  quality, authoring or device acceptance. Missing evidence must not pass a gate.
- Enforce test-label separation at the process/container/access boundary and RAM,
  time and thread caps. A selection receipt or hash alone is not access isolation.
  Confirm validation-only selection precedes any authorized test release.
- Start A4 MSLR terms/source closure and A10 Veteran provenance closure. Freeze an
  equivalent documented substitute before evaluation if necessary; keep unresolved
  rows open and progress unrelated sources independently.
- Audit actual F0/F1 prerequisites against public CPU call paths and installed
  workflows. List each remaining blocker to 077; do not infer exit from recipe count.

## Acceptance and reflection

Omission, duplication, stale identities, invalid outputs, worker/resource failures,
and attempted label access cannot produce a pass. A small valid search selects
using validation, releases only the selected model and reproduces its independent
score. This synthetic harness test does not pass real A13 or E3.

Publish the complete ledger and enforced-environment manifest, plus the frozen
job-count/resource preflight for [071](071-real-multioutput-selection.md). Keep
source issues and formal phase blockers explicit. If a missing judging component
needs a substantial implementation, split a focused follow-up before expensive jobs.
Reflect on whether the report now proves conditions or merely inventories files.

## Results

Not run. Existing integrity and short quality reports do not establish these gates.

### Plan and first counterexample

First inspect existing coverage/integrity tools, reproduce a producer-shrunk matrix,
add an evaluator-owned execution freeze to the existing judge, verify malformed/
rehashed cases and CLI pin checks, then commit a reproducible smoke. Separately
inventory remaining accounting/isolation and complete-coverage work before large jobs.

The old judge correctly rejects missing records relative to its manifest, but a
producer can delete a second required A1 fold from both manifest and records and
recompute all cache keys. All A1–A13 are still present and declared integrity passes.
The new optional evaluator freeze rejects that attack, changes to provenance/
protocol/backend/requiredness and duplicate/omitted cases even after rehashing.
The CLI requires a pinned reference-file hash outside the producer output root.
58 focused tests pass, including the deliberately failing initial counterexample.

This is an execution-manifest binding, not the complete R/C/A/E coverage ledger.
The reference and invocation must be evaluator-owned; filesystem/process isolation
and actual experiment completeness remain open. No E-gate result is emitted.

Full regression: **1010 passed**; lint/docs pass. A nine-case reproducible smoke
includes the valid matrix, rehashed fold omission/code change, missing/duplicate
records, wrong backend, worker-error/timeout status and nonfinite metrics. These
are injected judge inputs, not evidence of actual OS resource/access enforcement.

### Real worker retention follow-through

The current A1–A12 worker still inherited full traces after 068. Three failing
checks observed full payloads through actual squared, Normal and A5 quantile
calls. It now explicitly passes summary retention and records that policy in
training metadata, without adding a search parameter. All current-worker direct
recipe and fresh-prediction parity checks run before commit. Other worker families,
including composed frequency/severity, remain a separate preflight audit item.
This fixes a known diagnostic-memory policy gap; it is not a full-search launch.

### Evidence checkpoint

The [clean synthetic smoke](../benchmarks/v1/evidence/frozen-judge-070/README.md)
accepts one valid bundle and rejects eight faults. Source hashes match `63700db`.
Worker summary policy at `1a7bfd5` passes 86 adapter tests and 1013 total CPU tests,
including direct full-recipe and fresh inference checks. No core production API
changed in these slices.

Reflection: two concrete gaps are closed, but their evidence must not be promoted
to full isolation or coverage acceptance. A producer cannot shrink an externally
frozen execution matrix; the evaluator must still construct that matrix correctly
and protect it. Summary retention is explicitly selected in the current worker;
the full-search environment is still unqualified. See the
[readiness inventory](070-readiness-inventory.md) for the next bounded work.
Sprints 069/070 remain open; no independent author or expensive search was launched.

### Preregistered real permission/resource probe

Extend the existing process runner with explicit Linux address-limit and unprivileged
worker modes. First source checks reject invalid limits and unsupported hosts before
launch. Then commit the harness and run one Modal CPU container (2 CPUs, 8192 MiB,
120 seconds, no application retries) with only three allowlisted probe/runner files.
Use synthetic evaluator-only labels/verifier fixtures under root-owned mode-0700
storage, never actual sealed tasks or test labels.

The worker runs as UID/GID 65534 with empty supplementary groups, no_new_privs,
an explicit 8-GiB address ceiling and minimal environment. Probe denied label reads,
verifier writes, parent-environment reads, UID restoration and hard-limit raising;
confirm evaluator files remain unchanged. Exercise a 9-GiB mapping rejection,
actual worker error and a 0.25-second forced timeout with retained logs. A successful
NumPy worker uses the 1800-second search timeout setting, verifies two BLAS threads
and touched-allocation peak accounting; this does not wait 1800 seconds or qualify
a 300/1000-round model workload.

This checks a concrete UID/file boundary and process resource mechanism, not a
complete hostile-code sandbox. Network/new sessions are not restricted and same-UID
outputs must not be shared across independent attempts. Full author dispatch and
full-search access/container integration remain open, even if these probes pass.

### Composition worker audit

Frequency/severity calls now explicitly retain summaries, with per-component
metadata. The failing-before payload check and existing direct full-recipe/fresh
inference checks pass (10 focused, 1019 total CPU tests; lint passes). The global
parametric control worker has no boosting trace history. This closes the remaining
identified worker policy gap without changing frozen search configurations.

### Real probe results and reflection

The [clean Modal probe](../benchmarks/v1/evidence/worker-access-070/README.md) at
`26a6797` passes all 13 checks across seven subprocesses (one success, six expected
failures). All 17 artifact hashes and three committed source hashes were checked
locally, together with actual permission, allocation and timeout logs. Composition
summary policy is committed at `8eace67`; 1019 CPU tests and lint pass.

Reflection after three implementation/evidence slices: worker policy and a real
UID permission mechanism now have direct evidence. They do not establish a full
author sandbox, complete protocol coverage or real selected quality. The next
bounded step is integrating the existing search worker with evaluator-owned data
and container boundaries, then exercising that path before a full frozen search.
Keep the protocol-derived expected matrix and 069 token/budget accounting explicit
parallel planning obligations. No full search, independent attempt or GPU work was
launched. Sprints 069/070 remain open.

### Protected selection integration slice

Plan: extend the existing current selection smoke with an opt-in Linux permission
mode, verify unsupported-host rejection before any writes, preserve the local
16-trial selection/replay path, then exercise actual Linux permissions and workers.
The first two host tests failed on the missing interface.

Implementation separates evaluator records/test features from read-only candidate
packets, applies the existing unprivileged runner with an 8-GiB address ceiling
and 1800-second deadline, and reclaims completed output directories before another
worker starts. The current recipe worker still requires one thread; this stays
within the two-thread ceiling and is not a claim of exact two-thread execution.

Local validation: 16 four-round trials succeed and select `openboost:15`; the
existing independent validation audit and fresh selected-model replay succeed.
CPU regression: **1021 passed, 1 skipped**; lint passes. The skip is the new real
Linux/root integration test, which attempts protected-file reads/writes and runs
all 16 actual workers. It must pass in a traversable installed Linux environment
before this mode is considered operationally verified. No full search was launched.

Next: run that integration test on bounded Modal CPU hardware from a committed
source snapshot, retaining execution evidence. Do not infer this new path passes
from the standalone probe at `26a6797`.

### Linux selection preregistration

Run the three current selection tests and retain one additional 16-trial protected
selection/replay bundle on Modal (two CPUs, 8192 MiB, 180 seconds, zero application
retries). Upload only public CPU sources, named support modules, the focused test
and packaging metadata. Create a separate Git snapshot inside the container so
the smoke's provenance remains literal; record the original revision and every
uploaded hash in the outer manifest. No original Git history or sealed data moves.
Probe denied reads of test features and completed models, and protocol writes,
with actual known-path PermissionError logs and unchanged hashes.

### Linux results and correctness reflection

At `a0bc473`, all three focused tests run on Linux and pass. An additional retained
16-trial protected selection and selected-model replay passes; actual feature read,
protocol write and completed-model read attempts fail under permissions. All five
harness checks pass. [Raw evidence](../benchmarks/v1/evidence/protected-selection-070/README.md)
includes 115 verified artifact hashes and all 38 original source hashes.

New counterexample: macOS re-audit of the unchanged Linux receipt rejects 12 last-bit
score differences (maximum absolute difference 3.552713678800501e-15), although the
winner and all non-score fields agree. Same-environment Linux release passed.
Record this as a receipt portability limitation; do not silently replace the receipt
or relax integrity. Next define and test the numerical receipt contract, including
changed winners and near ties, before further search expansion. This correctness
reflection supersedes the prior immediate full-search direction.

Validation: 1021 local CPU tests pass, one Linux-only test skips locally and passes
on Modal; lint/docs pass. Sprints 069/070 and full-search/author/device gates stay open.

### Numerical receipt replay contract

Plan: reproduce the unchanged Linux receipt failure, specify a score-only numerical
bound, retain exact identity/winner checks, then test boundary, malformed, material
change and near-tie rejection. Three new tests failed before implementation.

Recomputed finite scores may differ by at most eight times the smaller binary64
spacing of the two values. There is no absolute-error floor or broad relative
tolerance. Every non-score field, including winner, remains exact, and the saved
score ordering must independently select that same winner. Receipt bytes and all
artifact/protocol/record hashes remain exactly pinned. Changed winners across a
near tie reject release rather than silently changing selection.

The unchanged committed Linux receipt now releases on macOS. Thirty focused
selection tests pass, including the 8/9-spacing boundary, malformed/material
scores, near-tie reversal and existing artifact/receipt tampering checks. This
resolves the observed receipt case, not every possible cross-platform difference
(e.g. training-scale recomputation still has its own exact contract).

Final verification with two test workers: **1031 passed, 1 skipped** (Linux-only).
Production/support lint and documentation build pass. No new Modal run was needed
to replay the unchanged Linux bundle on the host that exposed the counterexample.

### Real A6 resource matrix preparation

Compile the OpenBoost preflight directly from the frozen numeric-tree family in
search-design.json. Use its 16 configurations unchanged for each of shared and
independent topology, adding the frozen 255-bin budget and patience 50. This makes
topology a separate method (32 fits per fold), not a hidden extra tuning dimension
inside a 16-trial allowance. All five subject folds are required.

The [generated plan](070-a6-resource-plan.json) contains 160 explicit jobs, source/
preprocessing freeze hashes, one-thread worker semantics under the two-CPU ceiling,
1800-second deadlines, 8192-MiB container requests, 8-GiB address limits and zero
retries. Maximum fit-worker time is 288000 seconds (80 hours); reserved two-CPU
time is 160 CPU-hours. Setup, inference and comparators are excluded from these
ceilings. This is the OpenBoost portion only, not the complete A6 comparison matrix.

First bounded probes are fold-zero configuration 00 for each topology, sequential
and with stop-on-failure. Before dispatch, bind real train/validation packets and
exclude all test material from candidate containers. No jobs are launched by this
compiler. Seven focused tests verify complete config preservation and reject
omissions, duplicates, shortened rounds, ignored fields and changed resource/fold
budgets. Full comparator coverage and the complete R/C/A/E ledger remain open.

Final validation: **1038 CPU tests passed, 1 Linux-only test skipped** with two
test workers. Production/support lint and documentation build pass.

### Real packet binding and probe dispatch

The existing exporter verifies all five Parkinsons subject folds against source
and preprocessing freezes. Only fold zero's train/validation worker packet is
allowlisted for upload (no test arrays/files). Dispatch the two preregistered
configuration-00 jobs sequentially, with no retries and stop after any failure.
Each child has 1800 seconds, an 8-GiB address limit, one worker thread and summary
retention; the container requests two CPUs and 8192 MiB with a 1900-second timeout.
Record actual fit status, ru_maxrss in Linux bytes, and exact fresh validation
replay. The 300-round budget includes preregistered patience 50, so legitimate
early stopping is not a budget shrink. No full search or test release is implied.

### Dispatch blocked before upload

Automatic approval review rejected the Modal command before execution. It requires
explicit authorization for this real Parkinsons train/validation payload despite
prior general Modal authorization. No remote probe or upload occurred. Do not
retry via another mechanism. The local packet is 1494662 bytes: training features
3487 by 38 and targets 3487 by 2; validation features 1151 by 38 and targets 1151
by 2, plus 1151 validation row IDs. No test files are allowlisted.

Local CPU validation: 1038 passed, one Linux-only skip; lint, compilation and docs
pass. The committed harness is ready for review; actual resource/replay outcomes
remain unknown until payload upload is explicitly approved.

### Approved dispatch results and reflection

After explicit payload approval, the two real probes run successfully from clean
`9387d9f`. [Raw evidence](../benchmarks/v1/evidence/a6-real-probes-070/README.md):
shared trees take 1077.14 seconds with 59 completed rounds and 108097536-byte peak
RSS; independent trees take 767.13 seconds with 65 rounds and 136032256-byte peak
RSS. Both stop by frozen patience 50 within their 300-round budgets and replay
validation predictions exactly in a fresh process. No test data is uploaded.
All 31 source hashes, 20 artifacts, packet/plan hashes and target scales verify.

These are two scoped resource passes, not full-search qualification. Practical
runtime is now directly measured and substantial despite early stopping. Before
expanding to the remaining configurations, inspect a bounded profile of this exact
input and configuration, then decide whether a targeted change is justified. No
bottleneck or comparative speed claim follows from wall time alone. Keep the
complete ledger, comparator matrix and author accounting open. This is a reflection
checkpoint; no additional jobs or optimization are included in this slice.
