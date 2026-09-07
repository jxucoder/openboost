# Sprint 086: From verified storage to programmable CUDA boosting

Status: next execution plan, 2026-09-07. Planning baseline: clean `8617b38`.
This decomposes the approved [085 amendment](085-foundation-focus-amendment.md)
into bounded deliverables within [069](069-authoring-pilot.md) and
[078](078-cuda-scalar-path.md). It adds no device budget or phase exception.
All R1–R9/C1–C7/A1–A13 and E0–E7 requirements remain unchanged.

Execution update: 078-A passes 33 real T4 checks at `ad2f4e6`; see
[the result and retrospective](078-cuda-scalar-path.md). Both approved device
runs are consumed. Stop at this checkpoint; next local construction is 078-B.
The budget/next-action language below records the original planning baseline,
not another remaining allowance. GPU training and independent authoring remain open.

## Product outcome and present evidence

The next product milestone is an installed public composition that trains at least
two weighted numeric/missing boosting rounds on CUDA, then changes split feasibility
through D2's independent cohort statistics. Its saved model must predict in a fresh
CPU process. This connects the programmable foundation to actual accelerated work.

| Area | Verified at this planning baseline | Missing evidence |
| --- | --- | --- |
| CPU foundation | Twelve recipes, public operations, installed D1–D5 development checks, sequential M=1/8/32 semantics | Formal CPU exits, complete real application quality/selection |
| CUDA storage | Twelve installed T4 tests at `77aa105`; explicit ownership, copies, lifetimes and counters | Device statistics, splits, trees, transactions and training |
| Authoring | Clean D1/D2 author view and wheel at `cfca092` | Enforced accounting, isolation, fair frozen arms, independent attempts |
| Product value | Development examples and bounded evaluation artifacts | Measured author benefit, matched-quality GPU cost and independent reuse |

Evidence: [storage](../benchmarks/v1/evidence/cuda-storage-078/README.md),
[author packet](../benchmarks/v1/evidence/author-packet-069/README.md),
[coverage inventory](070-readiness-inventory.md). The latest recorded CPU suite is
1135 passed, one Linux-only skip; this is a regression record, not v1 completion.
No GPU training, performance or adoption claim follows from these results.

## Construction boundaries

Keep Python as the public composition and orchestration layer. CuPy owns device
allocations and streams; initial custom kernels use numba-cuda as specified in the
[construction design](../planning/foundation-construction-design.md). CPU remains
the semantic reference and usable development backend.

Build resident representations with the same mathematical contracts as CPU
components. Do not pass CUDA buffers into constructors that convert to NumPy.
Preserve prepared-data identity, original row positions, field names and weight
roles, deterministic selection, accepted/proposal ownership and persisted inference.

Host work may prepare data, validate metadata, orchestrate nodes and consume
declared compact decisions. Gradients, field aggregation, candidate scoring and
feasibility, routing, leaf arithmetic and repeated raw updates belong on device.
Record validation flags, winner/metric exports, synchronizations and their costs.
Reference exports in diagnostic tests must be distinguished from recipe execution.

The public device component boundary must be designed with its consumer:

- D2 adds arbitrary named independent columns, combines child-information minima
  with ordinary feasibility, and selects the best remaining ordinary Newton gain.
  No hardcoded cohort names, task-name branches or CPU candidate loops.
- Declare batch input/output schemas and supported device operations. An installed
  extension cannot depend on `ExecutionContext._array` or other private imports.
  Settle ownership before exposing kernel inputs: custom work must not obtain a
  mutable alias to authoritative accepted state. Owned work buffers and returned
  results need explicit lifetimes and allocation accounting.
- Strict CUDA preflight checks the actual components and rejects unsupported
  CPU-only callbacks before training/upload. A wrapper that silently ignores an
  extension is a correctness failure. A general compiler/autograd system is outside
  this iteration; implement only the public operations justified by the consumer.

## Execution slices

The following are deliverables inside existing sprints, not additional mandatory
phase gates. Commit each independently verified slice; reflect after at most
three implementation commits or any ownership/semantic counterexample.

### 078-A: Named fields and routed histograms — next bounded slice

First failing check: public device field/aggregation operations are absent. Freeze
the fixtures below and implement their independent original-row CPU checks before
writing kernels. Preserve the existing CPU path; do not change its expected results
to match a new device reduction.

Deliver resident numeric codes/missing flags, named fields and routed row views.
Objective weights apply once to objective fields; independent information remains
unweighted, including on zero-objective-weight rows. Aggregate actual selected
original rows with separate missing bins and integer counts. Represent empty
routed views explicitly; the current nonempty storage-upload restriction cannot
be bypassed with fabricated rows. Reject stale/foreign/released handles, misaligned
identities, duplicate/out-of-range rows and invalid field roles. Value checks on
resident inputs must expose any compact validation synchronization.

Frozen development fixtures for this slice:

| Fixture | Inputs | Required check |
| --- | --- | --- |
| F1: weighted named fields | Eight rows; feature-major codes `[[0,0,1,1,2,2,3,3],[3,2,1,0,3,2,1,0]]`; four regular bins per feature; missing rows 1/6 for feature 0 and row 4 for feature 1; g=`[-6,1,1,1,1,2,-2,2]`, h=ones; weights=`[0,1,2,1,0,3,1,1]`; cohort A on even row positions, B on odd positions | Once-weighted G/H; independent A/B sums; missing bins; total/count agreement |
| F2: routed subsets | F1 with all rows, `[6,0,3,7]`, empty, `[0,4]`, and `[1,6]` | Unsorted original-row identity, empty output, zero objective mass with nonzero information, all-missing feature-0 route |
| F3: bounded size | N=8192, F=32, B=32; code `(r+3*f)%32`; missing `(r+f)%17==0`; g=`(r%13-6)/8`, h=`(r%7+1)/8`, weight=`(r%5)/4`; even/odd cohort indicators | Full and every-third-row reductions; reproducible nonrandom inputs within the approved shape bound |
| F4: failures/lifetimes | Wrong context/identity, bad names/roles/shape/rows, nonfinite fields, negative D2 information, repeated weighting, released/closed storage, allocation exhaustion | Explicit failure with no corruption of previously valid inputs; all temporary allocation paths accounted |

Row positions `r` and feature indices `f` are zero-based. Use float32 fields and
integer codes/counts. Integers and routed positions compare
exactly; float32 values use E1 `rtol=1e-4`, `atol=1e-5`. The oracle sums original
input rows in float64, without importing device implementations. Also compare the
public CPU operation where its input contract applies. No tree or split optimality
claim is made by these histogram tests.

Acceptance: all declared fixture cells pass on an installed wheel on real T4;
the operation path performs no bulk host round trip; prior storage tests still
pass. Commit raw failures as well as passes. A simulator, skipped job, or successful
local import cannot establish CUDA correctness.

### 078-B: Candidate batches, routing and leaves

Entry: 078-A correctness and ownership review. Deliver batched numeric candidates,
both missing directions including missing-only splits, positive-gain selection,
routed child views and weighted Newton leaves through public operations. Preserve
the CPU candidate universe and condition keys; avoid transfers per candidate.

Freeze an exhaustive candidate oracle before implementation. Include exact ties,
unambiguously separated winners, no positive gain, empty children and zero
curvature. Exact ties use lexicographic `(feature, threshold, missing_left)` order;
there is no added near-tie equivalence in this initial fixture set. Do not loosen
E1 or accept an incorrect non-tied winner because final loss is similar.

Carry the known six-row D2 fixture into this slice: g=`[-6,1,1,1,1,2]`, h=ones,
lambda=1, alternating cohorts. Include the all-A-left/all-B-right no-feasible case
and zero objective weights with independent information. Independently enumerate
every candidate, compare legality and gain, then verify the selected condition,
child rows and leaves. Device reductions plus a CPU feasibility callback do not pass.

Acceptance: scalar and D2 candidate-level checks pass on real CUDA under the
unchanged E1 tolerances; all declared custom operations are observed executing.
The concrete remote workload and further budget must be recorded before dispatch.

### 078-C: Resident two-round training and transaction integration

Entry: verified primitives and an explicit accepted/proposal ownership design.
Deliver the weighted squared recipe using the same public composition/runtime
contracts, with resident targets/raw values, device derivatives, tree assembly,
leaf updates and at least two actual rounds. CPU metadata/control is allowed;
replaying the host ensemble each proposal is not resident execution.

Preserve base predictions, offsets, weights, learning rate, preparation identity,
parent identity and acceptance/best/stopping separation. Test rejected work and
same-step retries against accepted snapshots, including raw/model/best/stopping
state and scoped RNG. Workspace reuse must not mutate prior accepted results.
Define supported policies explicitly and reject unsupported ones, rather than
replacing public policy behavior with a fixed internal training loop.

Acceptance: independent agreement for derivatives, named statistics, every chosen
split, routes, leaves, first and second raw updates, predictions and task metrics.
Counts/routes are exact; float32 uses E1; task metric difference is at most
`1e-3 * max(1, abs(CPU metric))`. An installed model exports topology/leaves and
replays in a fresh CPU process without CUDA or training plugins. Record transfer,
sync, allocation and compilation scopes. Only then describe scalar GPU boosting
as implemented and verified within its declared limits.

### 085-D: Installed programmable device composition

Use the same public D2 definition with the completed scalar training path. An
external package must add information fields and compose feasibility through the
public device interface, change the selected split, run two rounds and export
plugin-free inference. Verify renamed/reordered cohort columns and a changed
information minimum so the result does not depend on a special task switch.

Acceptance: exhaustive D2 math and state checks pass, device work is observed,
and no core edits/private dependencies/bulk CPU callback are required. This is
development expressiveness evidence. Only a separately isolated author attempt
can establish author cost; the foundation designer implementing D2 cannot do so.
It supplies the D2 portion of [079](079-cuda-distribution-and-extension.md), whose
Normal/backtracking and P7 requirements remain open.

## Authoring work alongside construction: 069

Keep the existing `cfca092` author packet immutable. A later wheel or better docs
create a new named cohort. Do not give an author task-specific solution assistance
and then count the retry as an independent first attempt.

1. Audit the available runner's actual generated-token and wall-clock accounting.
   Produce an executable enforceable path or a concrete blocked record; do not
   spend repeated iterations inventing synthetic accounting interfaces. No author
   attempt starts with unobservable token use.
2. Make D1/D2 verifiers separately invocable with a frozen dependency closure.
   Test actual author-environment denial of evaluator files and write access, and
   enforced budget exhaustion. Verify the initial docs' links work in that view.
3. Audit and freeze the appropriate incumbent control/deep-change paths, allowing
   hooks or source edits. Pin task/arm prompts, model/settings/tools, wheels and
   observable 1800-second/20000-generated-token ceilings. A ready packet is the
   deliverable before seeking any missing independent-run authorization.
4. After readiness and authorization, execute isolated pilot attempts and retain
   failures, hints, setup/blocked/active time and first correct completion. Preserve
   the original formal E5 cohort sizes and thresholds; a D1/D2 pilot does not pass E5.

No new agent dispatch or contact is authorized by this plan. The existing sealed
H1/H2 evaluator remains isolated; do not inspect its task contents. Accounting
blockers do not prevent independent CUDA construction work.

## GPU budget and next retrospective

085 permits two device runs capped at 900 seconds each, with zero automatic retries.
Run 1 was consumed by storage. **Exactly one capped run remains.** Its planned
scope is 078-A plus existing storage regression checks, not a promise to validate
the entire training stack in the remaining allocation.

Before run 2, commit the fixture/oracle definitions, component implementation,
installed-wheel harness, exact expected cases and source hashes. Use one T4,
the existing 8192-row/32-feature/32-bin bounds, float32 and a 16-MiB private pool
cap. Keep 900 seconds for the function and 600 seconds for tests, zero retries and
one invocation. Freeze compatible CuPy/numba-cuda/transitive dependency versions,
image and CLI before dispatch; log the actual loaded runtime and driver separately
from the image tag. All temporary kernel allocations must respect the declared
allocator scope. A private pool peak is not whole-device memory usage.

The existing storage harness hardcodes run 1; do not reuse it to claim a fresh
allowance. A follow-up harness must name run 2 and bind the expected cases to its
frozen manifest. A failure/timeout consumes the run; missing cells cannot pass.
Do not add unregistered experiments inside the same function to evade this bound.

Stop for a user retrospective after run 2, or earlier if the public ownership
boundary fails. Discuss whether to fund the next concrete device checks based on
the retained result. Further remote verification of 078-B/C/D needs a new bounded
allowance; this planning turn grants none. Local design/tests can prepare that
decision. Reflect sooner after three implementation commits even without a run.

The retrospective must answer:

- Can public fields preserve identity, weights and independent information on GPU?
- What still runs on CPU, and is each transfer/control decision justified?
- Can D2 use the public boundary, or does it expose a missing operation/unsafe alias?
- Is independent author accounting ready; if not, what precise capability is missing?
- What is the next falsifiable result, its workload and requested compute bound?

## After the first programmable CUDA milestone

Retain the existing roadmap rather than add another feature catalogue:

| Work | Deciding evidence |
| --- | --- |
| 079: Normal and adaptive updates | K=2 ordinary/Fisher, actual rejection/backtracking and joint/ordered state; separate original P7 reproduction and 1.2 threshold |
| 080: required device coverage | Each required R1/R4/R5/R6/R8 cell, including classification, exposure-aware Poisson, event/right-censored AFT and multioutput; per-cell parity and persistence |
| 081: compatible train-many | Sequential CUDA reference, then real M=1/8/32 batching with independent seed/stop/failure and CPU replay |
| 070–076 / 077 | Complete source/isolation/selection obligations and every A1–A13 application; formal independent E5 attempts with original frozen gates |
| 082 | Qualified real workloads and repeated matched-quality end-to-end cost, including JIT/transfers and full ensembles; unchanged E4 thresholds |
| 083 / 084 | Reconstruct E0–E6 engineering evidence, then separately verify independent use/reuse for E7 |

Wider CPU searches and the deeper A6 CPU probe remain paused until a reflection
records the concrete product question they answer. Their requirements and failures
remain in the ledger. No new objective catalogue, speculative CPU speed rewrite,
Ray, multi-GPU or out-of-core expansion is scheduled. All required application
families remain required; CUDA-optional cells retain the canonical R-table status.

## Planning verification and handoff

This card changes planning/documentation only. Check relative links and
`git diff --check`; do not rerun the CPU suite or spend GPU budget to validate prose.
Commit this plan and corrected current-status pointers, without pushing.

Next execution starts at 078-A: implement the frozen reference fixtures, define the
public field/route ownership contracts, then implement resident aggregation. Keep
local-only implementation status separate from real-device acceptance until run 2.
