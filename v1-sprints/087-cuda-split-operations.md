# Sprint 087: Public CUDA candidate and feasibility composition

Status: 078-B complete with 88 real T4 cases passing at `9ce790e`; retrospective reached.
Baseline `4251a36`. Implements 086's 078-B; it does not establish a formal phase exit.
Both 085 runs and the separately approved invocation below are consumed.
078-C resident training and accepted/proposal integration remain open.

## Retrospective decision

Real T4 statistics preserve identity, objective weights and independent information.
The remaining product risk is expressing an algorithm change through public device
operations. Continue with independently composable candidate scores and masks,
then routing and leaves. Keep CPU searches paused and author accounting gaps visible.
Do not turn this work into a private GPU trainer or a generic compiler project.

## Public construction contract

- `candidates(histogram)` returns a device batch with left/right named sums,
  integer child counts, parent totals and an active-candidate mask. Its padded
  slots are lexicographic feature/threshold/missing-direction positions. A slot
  is active only for a bin observed anywhere in the prepared numeric data, matching
  the CPU candidate universe even when a node uses a subset of rows.
- `newton_scores(batch, reg_lambda, split_penalty)` and `feasible(batch,
  min_child_h)` are separate operations. Scores for structurally illegal candidates
  are zero; legal gains must be finite. Standard scalar operations require named,
  once-weighted gradient/curvature and nonnegative row curvature.
- `child_minimum(batch, name, minimum)` checks a declared independent information
  column on both children. It validates that original information is nonnegative.
  `mask_and(a, b)` composes predicates on exactly the same batch. D2 uses these
  public operations for arbitrary cohort names/minima, without task-name dispatch.
- `scores(batch, buffer)` / `mask(batch, buffer)` bind explicitly supplied resident
  float32/bool vectors. Borrowed lifetime and exact batch identity are validated.
  These are explicit bindings, not a claim of external custom-kernel support.
- `choose(batch, scores, mask)` returns one registered split or None, with strictly
  positive gain and exact lexicographic ties. It always excludes inactive padding,
  even if a supplied mask enables it. Only a compact winner index returns to host.
- `partition(rows, split)` produces two original-position device row views in the
  input order, bound to the selected batch. Export only compact child sizes for
  allocation. `leaf(histogram, reg_lambda)` returns an owned scalar Newton value.
  Child histograms are recomputed from those actual routed rows.

All kernel arrays/scratch remain in the owned CuPy pool/stream. Explicit validation
flags, winner/size exports and sync counts remain visible. No host candidate loop,
bulk round-trip or writable authoritative state is exposed. Unsupported scalar
schemas, wrong contexts/batches, stale/released buffers and invalid parameters fail.
No training or accepted/proposal integration is included in this slice.

## Frozen development cases before implementation

The independent oracle enumerates conditions and original rows in float64; it
does not import histogram or device production operations. Compare its candidate
universe, child sums/counts, gains, masks, winner, routes and leaves with public CPU
operations locally before adding kernels. Cases use lambda=1 and minimum=1 unless
specified, with fixed integer bin codes and original row IDs distinct from positions.

| Case | Inputs and deciding property |
| --- | --- |
| S1 | Six distinct bins, g=[-6,1,1,1,1,2], h=ones, alternating A/B; unconstrained winner differs from D2's best feasible winner |
| S2 | Same gradients, one predictor with codes [0,0,0,1,1,1], A only in first three rows, B only in last three; no D2-feasible split |
| S3 | 086's eight-row F1, including missing values, zero weights, independent information and unsorted subset [6,0,3,7] |
| S4 | Duplicate columns [0,0,1,1], g=[-2,-2,2,2], h=ones; exact feature/missing-direction ties resolve lexicographically |
| S5 | S4 with zero gradients, then with zero curvature/positive lambda; no positive legal split; zero-curvature leaf with lambda=1 remains defined |
| S6 | One column [0,0,missing,missing], g=[-2,-2,2,2], h=ones; last observed bin separates missing rows |
| S7 | Codes [0,0,2,2] and a second all-missing feature; inactive bins/padding never become eligible, including under an explicitly supplied all-true mask |
| S8 | S1 with cohort names/order changed and minima 0/1/2; predicates compose and bind to the same batch without fixed cohort names |
| S9 | Empty/all-one-bin nodes, invalid denominators and invalid/overflowing parameters, nonfinite gains, negative information/curvature, wrong-batch buffers, stale rows/splits and allocation failure |

Integer counts/routes and active universe are exact. Float32 comparisons use
unchanged E1 rtol=1e-4/atol=1e-5. Exact tie fixtures have no additional near-tie band;
other winning gains are separated. Similar final scores cannot excuse a wrong
non-tied split. No two-round training, quality or speed claim follows these cases.

## Work and acceptance

1. Commit the independent fixtures/oracle and CPU agreement; first failing check
   is the missing public candidate operation.
2. Implement public records/operations and Python CUDA kernels, with real-device
   tests kept explicitly not_run. Preserve all earlier 078-A/storage cases.
3. Freeze a reviewable installed-wheel validation package with exact cases,
   dependencies, source identities and proposed resource bounds. Verify its local
   guards/judge and CPU regression, lint and documentation.
4. Only after the package is concrete request a new one-run T4 allowance: maximum
   900-second function, 600-second tests, 16-MiB private pools, zero automatic
   retries. Include previous 33 cases and the new fixed small split cases. This
   proposal does not itself authorize a run. A failure consumes an approved run.

Hardware acceptance requires every preregistered cell to pass on an installed
wheel, with independent intermediate checks and installed source/version integrity.
Until then report implemented/unverified. No simulator or local collection counts
as CUDA correctness. No new author agent, held-out access or external contact.

Reflect after each verified slice/three implementation commits, or an ownership
counterexample. Next, after verified 078-B, is 078-C resident two-round training
under atomic state and CPU-readable inference contracts.

## Local implementation checkpoint

The fixture/oracle slice is committed at `0ba39a3`. Public candidate records,
scalar scores, independent predicates, composition, selection, stable routing and
leaves are implemented. Fifty-five real-device tests cover the frozen cases,
renamed/reordered information, explicit resident bindings, parameter/denominator
failures, source lifetime, exact batch identity, allocation recovery and stream
restoration. No simulator, device compile or hardware execution has been used.

Full CPU regression: 1185 passed, one Linux-only skip. This includes 31 exhaustive
split reference cases and the earlier eight run-2 judge checks. Local collection
finds 55 new CUDA tests; collection is not device acceptance. Production/test lint
passes. Hardware implementation risk remains open until the new run is approved
and all preregistered cases pass.

Reflection: D2 now composes named information and ordinary scalar legality without
private task dispatch. The implementation keeps public score/mask bindings separate
from arbitrary custom-kernel integration, which remains unimplemented. Candidate
storage and ordered routing prioritize bounded correctness; performance and
resident training are not established. Complete the run package before requesting
hardware. Required author, application, quality and end-to-end cost gates remain.

## Concrete run-3 request

Implementation `0bcb52f` and the exhaustive CPU oracle `0ba39a3` are frozen in
[078-splits-run3.json](078-splits-run3.json). The user approved the new allowance
by replying "Continue" to the concrete 88-case, 900-second, zero-retry request.
The package includes exactly 88 preregistered cases: 55 new split cases and all
33 previous storage/aggregation cases, with the same 17 pinned dependencies.
It freezes hashes of 38 uploaded files; the protocol itself is the 39th file and
is hashed in the dispatch manifest. Installed production modules, the uploaded
snapshot and package versions must all match, alongside an exact passing JUnit
case matrix and zero exit code. Missing/skipped/duplicated cases fail acceptance.

Requested resources: one additional T4 invocation, two requested CPU cores and
8192 MiB requested host memory, function timeout 900 seconds, test timeout 600
seconds, no automatic retries and one container maximum. Private CuPy pools stay
within 16 MiB; this excludes CUDA context/driver/JIT allocations. The original
8192-row aggregation regression remains the largest fixture. New split fixtures
are at most eight rows. No training, performance search or real-data evaluation
is included. A failed dispatch consumes this allowance; no automatic repair run.

After explicit user approval, record authorization in the protocol and commit
that state before running from clean source:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python \
  -m benchmarks.v1.cuda_splits_preflight benchmarks/v1/evidence/cuda-splits-078
```

The launcher rejects pending/consumed authorization, changed source hashes, dirty
source, a different output location or an existing output directory. It records
the runtime/driver versions separately from the CUDA image tag. Preserve failures,
logs, JUnit, installed hashes and verdict under that fixed evidence directory.
Hardware success or failure is the next retrospective boundary before 078-C.

Local package verification: 17 manifest/judge checks pass (nine new and eight
existing); collection matches all 88 cells. Production and changed support lint,
documentation build and offline wheel/sdist build pass. All 23 production modules
in the built wheel match the frozen source bytes. The pending launcher fails
before importing Modal or creating output. No remote call, upload, device compile
or GPU execution occurred. Full CPU regression remains the 1185-pass/one-skip
implementation check above; the nine new harness checks are additional local tests.

Reflection: foundation construction progressed through a public algorithm-change
boundary without treating implementation as device evidence. Keep the next step
bounded to validating this boundary. Do not accumulate a resident trainer on top
of unverified kernels or restart CPU throughput work. Independent author accounting
and all required application/device gates remain open.

## Approved execution sequence

1. Verify unchanged source hashes and the exact 88-case matrix, record approval
   and commit it before any upload. Only the protocol authorization changes.
2. Dispatch the fixed installed-wheel package once on T4 with the frozen limits.
   Preserve any failure; do not retry or change the cases after seeing results.
3. Audit raw logs, JUnit, installed/snapshot hashes and environment metadata;
   record acceptance or failure, consume the allowance and reflect before 078-C.

Authorization does not include another invocation, pushing commits, independent
author attempts or an expanded experiment. No hardware outcome is known yet.

## Dispatch approval block

At clean source `043861a`, automatic approval review rejected process creation for
the Modal launcher. The stated reason was that the bounded T4 compute allowance
did not explicitly authorize uploading private source code, tests and metadata to
Modal. No launcher process, output directory or redirected log was created; no
remote invocation or upload occurred. The one approved invocation remains unused.
This is an approval block before dispatch, not a failed CUDA test or a retry.

The exact upload is the 39-file allowlist in the frozen package: 23 production
Python modules, package metadata/README/license, the protocol, two launchers and
the required test/oracle modules. It excludes credentials, other repository files
and sealed task cards. All frozen source hashes still match and all 88 cases remain
fixed. Request explicit permission to upload this package to Modal before retrying
process creation. Do not use an indirect upload or another execution route.

The user subsequently replied "Approved" to the explicit request to upload the
frozen 39-file source/test/metadata package to Modal for the already approved
single T4 run. This resolves the transfer block. Record approval and verify the
unchanged package before using the original launcher; no workaround is needed.

## Run-3 result and acceptance

[Raw evidence](../benchmarks/v1/evidence/cuda-splits-078/README.md) records one
installed-wheel T4 invocation at clean `9ce790e`, with all 88 expected cases
passing: 55 split checks plus all 33 prior regressions. Local re-judging,
all 39 source files against the Git commit, all 23 installed modules, the uploaded
snapshot, 17 pinned versions and three artifact hashes agree. No case was missing,
skipped, retried or changed after results. The additional allowance is consumed.

D2 at minimum=1 selects threshold 1 with gain 20/3 instead of unconstrained
threshold 0 with gain 12. Renamed/reordered information and minima 0/1/2 pass the
same public path. Candidate intermediates, exact ties, actual child rows and leaf
values agree with the original-row oracle. Failures and lifetimes are exercised
on the device; there is no CPU feasibility callback or bulk candidate download.

T4 driver 580.95.05, runtime 12090 and driver API 13000 are distinct from the
12.6.3 image tag. The worker took 9.75 seconds and pytest 8.26 seconds, including
lazy compilation; these are test durations, not boosting speed. All 98 occupancy
warnings remain in the raw log. The largest of 30 small split-context pool samples
is 19968 bytes; the private-pool cap remains 16 MiB and excludes driver/JIT memory.

At the logged six-row D2 checkpoint, 24 decision-export bytes cover two choices
and their child sizes; 252 validation bytes and diagnostic exports are separate.
Its 89 explicit synchronizations include tests' diagnostics. Compact transfers
establish resident operations but do not establish low latency or full fit cost.

## Closure reflection and next work

1. The public device boundary now preserves named statistics, independent
   information and a changed split decision through actual CUDA execution.
2. CPU work is explicit preparation and compact control metadata; gradients,
   iterative raw updates, tree construction/inference and transactions are not
   implemented by this slice. Current recipes still train on CPU.
3. D2 required no private task switch. Generic external custom-kernel registration
   remains absent; composability of these operations is the verified scope.
4. 065 run isolation and 068 retained-state ownership remain requirements for
   078-C. Operation cleanup and borrowed-handle validation are not substitutes for
   atomic accepted/proposal state, best-model selection or stopping semantics.
5. No independent author attempt/accounting result was added. Required R/C/A/E
   application, authoring, quality, P7/E4 and adoption gates remain open.

Stop at the planned user retrospective. Next local construction should freeze
078-C's owned accepted/proposal/raw/tree records and independent two-round scalar
and D2 expectations before implementation. Acceptance must include raw updates,
rejection and same-step retry, selected/best/stopping separation, bounded retained
storage, and plugin-free inference from a saved artifact in a fresh CPU process.
Only a new concrete workload and allowance can authorize its hardware checks.

Local closure: full CPU regression passes 1195 tests with one Linux-only skip;
18 manifest/judge checks pass, including stored raw-artifact integrity. After
completion, source/case freeze tests audit the recorded run instead of requiring
future README/code to remain identical. The original 38 prefrozen hashes and all
dispatch guards are unchanged. Updating README acceptance first exposed the old
test's incorrect use of the live tree for a consumed run; archived verification
now fixes that without weakening the before-dispatch check.
Production and changed-support lint, whitespace checks, documentation and offline
wheel/sdist builds pass. No production code changed after the successful T4 run.
