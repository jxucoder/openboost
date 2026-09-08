# OpenBoost v1 execution and reflection

This directory records the user-requested sprint plans, results and reflections. The goal is
**to help researchers and agents make correct algorithm changes using public composable components,
and verify their cost and practical value.**

Sources: [main plan](../planning/agent-boosting-foundation-plan.md),
[construction design](../planning/foundation-construction-design.md), [tasks](../planning/foundation-tasks.md),
[evaluation](../planning/openboost-v1-evaluation.md). These define scope/architecture/gates; this
folder manages execution, not a competing roadmap. R1–R9/C1–C7/A1–A13 all remain required.

## Current execution position

The user's [101 deferral](101-defer-author-evaluation.md) pauses agent-friendliness
evaluation and its accounting/isolation preparation, including the unapproved
100 live model test. F2/E5 remains unpassed but no longer gates foundation or
CUDA construction. Public composability and correctness remain required.

The [102 Normal CUDA run](102-normal-cuda-validation.md) completes at `469ca0e`:
528/529 revised cases pass and the historical cohort has exactly its 26 expected
disagreements. All 409 declared JSON artifacts are retained. The lone revised
failure is a forward fixture's best-prefix expectation after a zero-valued final
term; independent derivation requires nine, while the test expects ten. Preserve
the [raw failed verdict](../benchmarks/v1/evidence/cuda-comparison-092/README.md).
Run 8's allowance is consumed, and agent studies remain paused.

The fixture correction at `6026ebb` has three CPU checks for ordered/joint best
prefixes and accepted no-ops. [103's real T4 revalidation](103-normal-cuda-revalidation.md)
then passes all fifteen recipe cases at clean `a7173d9`, including the forward
case's later assertions. All 46 uploaded sources and eighteen pinned packages
match. [Combined evidence](../benchmarks/v1/evidence/cuda-recipe-103/README.md)
verifies 514 earlier passes plus fifteen new passes with identical production.
Bounded revised coverage is complete across two runs; the historical failed
verdict and all 420 indexed run-8 files stay intact. The retrospective is complete,
and all nine GPU allowances are consumed. No further hardware is authorized.

The user subsequently approves [104's early performance checkpoint](104-early-performance-checkpoint.md)
on the existing squared/Normal paths. Its frozen, bounded same-host CPU/CUDA runs
precede further recipe ports; compilation/profiling and quality-qualified timing
remain distinct. No external-library speed or formal E4 claim is implied.
Automatic approval review blocks its dispatch before process creation because the
exact private-source payload and Modal destination were not explicitly approved.
No run occurs or allowance is consumed by that rejected attempt. The user's next
"approve" explicitly authorizes the requested 46-file, approximately 358 kB Modal
upload for one T4, two CPUs, 8192 MiB, 900 function seconds, 600 test seconds and
zero retries. Run 10 then executes once at clean `c8f7ebc`, with all sixteen case
artifacts and matching sources/packages. Only squared 10,000 rows completes and
qualifies: 7.044 s CPU versus 2.941 s warm GPU, ratio 2.395. Four other pairs and
the profile remain incomplete after deadlines; preserve the
[raw false verdict](../benchmarks/v1/evidence/early-performance-104/README.md).
No ratio is assigned to partial 100,000-row timings. Exact synthetic input hashes
also fail to reproduce on the local macOS audit host; same-host CPU/GPU identity
and the remote quality judge remain valid. Save exact inputs in future evidence.
All ten GPU allowances are consumed. The approved
[105 construction](105-parallel-validation-and-reproducible-cost.md) now preserves
exact input/per-fit evidence, adds cooperative field validation and prepares a
[frozen 88-file T4 comparison](105-validation-run11-request.md). Full CPU regression
passes 1994 tests with one Linux-only skip; 474 real-device cases collect from an
isolated wheel. The user then approves that exact upload and invocation. Run 11
executes once at clean `dd84247`: all 474 T4 checks and three cost gates pass, all
28 fits replay exactly from retained inputs, and large squared warm fit cost falls
13.513 to 8.947 seconds. [Raw evidence and audit](../benchmarks/v1/evidence/parallel-validation-105/README.md)
retain all artifacts and exact original/candidate models. All eleven allowances
are consumed. Keep the optimization and stop at the completed 105 retrospective;
next construction returns to required 080 recipes and inference metadata. These
synthetic results do not pass formal E4 or full Normal conformance.

[106 binary/Poisson device components](106-binary-poisson-device-components.md)
completes local construction: independent numerics and complete required scope,
resident objective operations, class-aware export and two-round composition checks.
Local regression passes 2,065 CPU tests with one Linux-only skip; forty-four new
GPU cases collect without execution. [107 loss-change and recipe integration](107-glm-comparison-and-recipes.md)
then completes local construction of resident comparison and scalar acceptance,
best-validation and independent patience consumers. Full regression passes 2,241
CPU tests with one Linux-only skip. All 153 GLM GPU cases collect but remain unrun.
The [108 run-12 request](108-glm-validation-request.md) now freezes 85 files,
571 cases and 77 mandatory JSON artifacts. Isolated installed collection agrees;
45 local packet/retention controls pass. Its concrete upload/hardware allowance
is pending; all eleven previous allowances remain consumed. Execute only after
that allowance, preserve the raw verdict and stop for retrospective.

After this checkpoint, advance
[required CUDA recipes](080-cuda-required-recipes.md),
[train-many](081-cuda-train-many.md) and [real-workload quality/cost](082-end-to-end-cost.md).
No new model call is needed for
these engineering checks. Preserve every application family and all past evidence.
Resume the deferred author study only at the user's direction.

[102's local readiness check](102-normal-cuda-validation.md) passes all 37 run-8
harness cases and verifies the unchanged source closure. Automatic approval review
blocked the attempted command before process creation because "finish" did not
specifically authorize the source upload and paid GPU invocation. Both packet
authorizations are restored to pending; no run occurred or allowance was consumed.
The user subsequently replied "approve" to the exact 86-file upload and single
bounded T4 invocation. Its authorization is recorded in 102; execute the original
packet once and stop for retrospective. That run is now complete as recorded above;
earlier pending/approved statements are historical.

## Prior preparation and evidence

The following authoring continuations and 085 priority order are historical and
superseded by 101. The latest live author-accounting result is
[099's provider accounting smoke](099-accounting-result.md):
real cap exhaustion passes and cancellation remains unexercised, leaving the
frozen overall verdict failed. It follows
[097's passing corrected Linux worker smoke](097-worker-identity-result.md),
following [096's failed Linux worker smoke](096-linux-worker-result.md)
and its [local construction](096-linux-author-worker.md),
following [095 author packet and failed native isolation](095-author-packet-and-local-isolation.md)
and [094 standalone author-verifier preparation](094-author-verifier-preparation.md),
following the [093 checkpoint](093-foundation-progress-and-next-steps.md).
Standalone D1/D2 development checks and a concrete runner audit advance 069;
independent dispatch remains blocked on actual accounting/isolation and arm/settings
freezes. Run 8 is still pending and its frozen inputs are unchanged.
The separately approved 096 CPU allowance is consumed: 14/19 checks pass, but the
worker runs as root and modifies core/material files. Timeout is not reached.
The evaluator remains outside the worker and unchanged. That failure is retained.

The user's subsequent "sure" approved [097's explicit identity correction](097-explicit-worker-identity.md)
and one separately frozen CPU smoke. At clean `518eccf`, all nineteen original checks,
both identity guards and actual provider expiry pass. The original thirteen uploads
are byte-identical; only the trusted launcher is added. Core/material writes and
root restoration fail as required, and all evaluator hashes remain unchanged.
That allowance is consumed, with no retry. The retrospective is complete: next
design actual generated-token/wall-budget enforcement and fair-arm/model/settings
before any independent attempt. No model or additional remote run is authorized.

[098 request accounting](098-author-request-accounting.md) now constructs the
trusted text-request boundary locally: durable pre-dispatch reservations, final
usage validation, no automatic retries and local transport-process deadlines.
Protocol fixtures are explicitly labeled and do not count as real token evidence.
Actual provider exhaustion/cancellation, complete worker integration and the
model/input/spend freeze remain open. No model request or new upload has occurred.

[099 background accounting](099-background-accounting-smoke.md) adds cancellation
and final retrieval within a fifteen-second cleanup window. Known usage is
reconciled while stopped answers are withheld; null usage stays unknown. Its
[concrete live smoke](099-accounting-smoke.json) freezes twelve source files and
three harmless text prompts, with at most three generations / 4288 output tokens
and a $0.05 allowance. Fifty-three local checks pass. The user then approved one
live observation at clean `5c0f31a`: actual cap exhaustion uses 128 then 64 tokens
and blocks the next request. The cancellation probe completes early with 104
output tokens, so no cancellation is observed and the overall verdict fails.
[Raw evidence](../benchmarks/v1/evidence/author-accounting-099/README.md) retains
all three responses, 296 output tokens total and the unchanged frozen verdict.
The allowance is consumed without retry. The [retrospective](099-accounting-result.md)
identifies an explicit active-response cancellation trigger as the next local
design; no new live allowance or independent author attempt follows.

The next user "continue" starts [100's active-stop construction](100-active-cancellation.md).
An explicit trusted stop policy now cancels after the first validated in-progress
observation, withholds stopped answers and separately judges cancellation and
final usage. Thirteen new local cases plus all 53 earlier cases pass. The
[new one-request packet](100-cancellation-smoke.json) preserves 099's model,
prompt, output cap and work window; it proposes a $0.01 allowance without retries.
Its authorization is pending, and no new model request has run. The active source
now differs from 099's historical snapshot; its consumed freeze and archived
evidence remain unchanged and independently verifiable.

The user-approved [085 foundation-focus amendment](085-foundation-focus-amendment.md)
sets the current order: [069 authoring/accounting preparation](069-authoring-pilot.md)
and bounded [078 scalar CUDA feasibility](078-cuda-scalar-path.md). D1 is the control;
D2 cohort-feasibility is the deep change carried through the programmable CPU/device
boundary. This explicitly permits feasibility before formal F2 completion.
The [086 next execution plan](086-next-execution-plan.md) breaks construction into
named fields/routed histograms, candidate operations and resident two-round
training, with explicit acceptance and a retrospective after the remaining run.

Pause the next OpenBoost configuration-05 CPU probe and wider CPU search expansion.
[070](070-coverage-and-judging.md) remains open and supports correctness/isolation
requirements for these workstreams. Its [readiness inventory](070-readiness-inventory.md)
retains the full coverage, selection and source gaps. All R/C/A/E requirements,
formal E5, required CUDA recipes and P7/E4 remain unchanged.

CPU implementation has twelve recipes; the latest full CPU regression passes
1949 tests (one Linux-only skip), including author/evaluation support checks.
Bounded real evaluation and installed extensions exist. All 212 real T4 checks
pass at clean `af026ef`, including resident scalar training and all earlier
primitives. The shared score correction resolves run 4's 14 parity failures.
Independent author benefit, full quality/search and
adoption remain unverified. The
[064–084 roadmap](roadmap-after-063.md) retains those obligations; 085 changes their
near-term priority and the bounded device entry rule. Reflect after each 085 slice
and every three implementation commits. No independent author attempt has run.
The two original device runs and the separately approved run 3 are consumed.
[087 split composition](087-cuda-split-operations.md) passes its 55 new cases plus
all 33 previous regressions at clean `9ce790e`; see the
[raw evidence and reflection](../benchmarks/v1/evidence/cuda-splits-078/README.md).
D2 changes the selected split through public resident scores and feasibility masks.
[088 resident scalar training](088-resident-scalar-training.md) connects public
operations to owned transactions, two-round training and saved CPU inference.
The initial [run-4 failure](../benchmarks/v1/evidence/cuda-resident-078/README.md)
is retained unchanged. [089 score symmetry](089-cuda-score-symmetry.md) fixes its
numerical counterexample inside the existing scorer. All 202 original cases and
ten added diagnostics pass at `af026ef`; the
[run-5 evidence](../benchmarks/v1/evidence/cuda-score-symmetry-089/README.md) reproduces
the archived scorer's wrong winner and the corrected scorer's exact tie on the
same resident inputs, with both PTX outputs retained.
The later approved [run-6 Normal/D2 package](090-normal-run6-request.md) executes
at clean `4143d18`: [381/383 pass](../benchmarks/v1/evidence/cuda-normal-090/README.md),
including all 212 earlier cases, all Normal operation/recipe checks and all twenty
installed-D2/fresh-inference checks. Two mapped-runtime cases disagree with the
frozen reference's ordered backtracking decisions near a stationary constant base.
That run did not retain the failing states. All nineteen saved models replay without CUDA or
the training extension; the separate split near-tie remains a known limitation.

The separately approved [run 7](091-acceptance-run7-request.md) captures both failing
states at clean `80740f2`: [383/385 pass](../benchmarks/v1/evidence/cuda-acceptance-091/README.md),
with all 383 original outcomes unchanged and two successful observations. Both
fail at round zero's mean update: float64 full-loss rounding reports improvement
for an actually worsening candidate. Version/best updates and cleanup follow the
recorded decisions. Full Normal conformance remains open.

All seven allowances are consumed; no retry occurred.
[091's retrospective](091-normal-acceptance-diagnostics.md) is complete. The user
approved local [092 construction](092-normal-comparison-design.md). Its independent
[numerical experiment](092-normal-comparison-mathematics.md) precedes the public
loss-change operation and three consumers. The
[106-case evidence](../benchmarks/v1/evidence/normal-comparison-092/README.md) and
[complete historical mapping](092-comparison-cohorts.md) close 092-A.
[092-B construction](092-public-comparison-operations.md) adds the public CPU
comparison and an unverified resident implementation with 117 collected GPU cases.
[092-C consumer construction](092-comparison-consumers.md) and complete
[historical bindings](092-cohort-bindings.md) are locally complete. The
[pending run-8 request](092-comparison-run8-request.md) freezes 86 files with
385 historical and 529 revised cases, collected from an isolated wheel/snapshot.
CPU regression passes 1794 checks with one Linux-only skip. No run-8 upload or
invocation is authorized; the next boundary is its concrete allowance and
post-run retrospective.
The [093 checkpoint](093-foundation-progress-and-next-steps.md) explains current
implementation/evidence and the proposed authoring and workload priorities. It is
a planning review, with no new execution allowance.
Original P7/E4, 069 accounting/isolation and full
R/C/A scope remain open. No additional upload, run or author attempt is authorized.

The [landscape feedback addendum](063-landscape-feedback.md) identifies an external
stopping-result restriction and refines authoring comparators and development probes.

[Sprint 062](062-cpu-exit-and-gpu-entry.md) retains the preceding CPU/GPU audit.
[Sprint 038](038-goal-progress-and-plan.md) and Sprints 039–061 record the earlier
construction, installed extensions and application evidence. The new
[runtime diagnostic](../benchmarks/v1/evidence/runtime-audit-063/README.md) counts
quadratic tree replay on tiny fixed-step paths, not wall time or GPU performance.

The chronological entries below record status at each sprint's revision. Their
historical "next" statements are superseded by the current review.

| Sprint | Plan mapping | Status | Deliverable and record |
|---|---|---|---|
| 001 | B01/F0.2 scalar/tree subset | Complete; 55 tests | [Scalar/tree references](001-scalar-tree-reference.md) |
| 002 | User-requested early production retirement | Complete; namespace/build/docs checked | [Clean v1 starting point](002-retire-legacy-production.md) |
| 003 | B01/F0.2 transforms/classification | Complete; see record | [Transforms/classification](003-data-classification-reference.md) |
| 004 | B01/F0.2 A4–A6 probes | Complete; 122 total tests | [Ranking/quantile/vector](004-ranking-quantile-vector-reference.md) |
| 005 | B01/F0.2 A7–A10 probes | Complete; 183 total tests | [Positive targets/AFT](005-positive-aft-reference.md) |
| 006 | B01/F0.2 A11/A12/D4 probes | Complete; 208 total tests | [Normal/Formula](006-normal-formula-reference.md) |
| 007 | B01/F0.2 identity/A13/D5 | Complete; 229 total tests | [Identity/runs](007-identity-runs-reference.md) |
| 008 | B01/F0.2 exact D1/D3/D4 | Complete; 255 total tests | [Author-task references](008-author-mutation-reference.md) |
| 009 | B01/F0.2 full mixed/vector growth | Complete; 279 total tests | [Mixed/vector trees](009-mixed-vector-growth-reference.md) |
| 010 | B01/F0.2 finite compositions and exit | Complete; 288 total tests | [Compositions and exit](010-reference-integration-exit.md) |
| 011 | B02/F0.3 integrity subset | Complete; 336 total tests | [Integrity judge](011-artifact-integrity-judge.md) |
| 012 | B02/F0.3 A5 data/date windows | Complete; 360 total tests | [Bike freeze](012-bike-data-freeze.md) |
| 013 | B02/F0.3 A1/A11 five splits | Complete; 380 total tests | [Housing splits](013-housing-five-splits.md) |
| 014 | B02/F0.3 A2 official test/stratification | Complete; 396 total tests | [Adult freeze](014-adult-data-freeze.md) |
| 015 | Cross-cutting English prose | Complete; no phase advancement | [English repository](015-english-repository.md) |

At Sprint 015, B02/F0.3 frozen evaluation was next and public F1 construction had
not started. The subsequently approved overlap and CPU construction are recorded
below. References are not product implementation; phase exits remain evidence-based.

## Execution rules

1. Start each sprint with purpose, F/B/A/C/E mapping, an independent failing example, deliverables and acceptance.
2. Commit every independently verified slice; record commands/results/unverified scope/commit, not just file counts.
3. **Reflect at every sprint closure, every three implementation commits, phase transitions,
   and architectural/correctness counterexamples.** Record it in the current sprint; one reflection can satisfy multiple triggers.
4. Record evidence/reasons and update design before changing architecture/order. Never silently
   relax gates, discard failures, remove cases or switch to multi-GPU/feature catalog work.
   Routine small fixes do not require replanning the whole project.
5. `learnings/` stores durable cross-sprint conclusions linking here; execution details live here.
6. All repository prose must be English, per the user's instruction.

## Reflection checklist

- Which difficulty in making a correct algorithm change did this reduce? If preparation only, which component does it justify?
- Does execution follow construction dependencies? Did old API restrictions, specialized trainers or premature optimization slip in?
- Is there independent math/state evidence? Which internal simulations cannot support performance, quality or adoption claims?
- Can structurally different cases reuse the boundary, or is it being designed around one example?
- What required scope remains? What is the next smallest verifiable deliverable?

Use observation → evidence → decision → next step, not an unsupported conclusion that the direction is right.

## Overall completion

F0.1 specifications/construction and F0.2 references are delivered. F0.3 and F1–F5 remain incomplete.
Sprint 001 delivered scalar/tree; 002 retired production; 003 transforms/classification; 004 ranking/
quantile/vector; 005 positive/count/policy joins/event-right-censored AFT; 006 Normal/Formula;
007 identity/isolation/selection; 008 D1/D3/D4; 009 full mixed/vector growth; 010 finite model/state
compositions. See [F0.2 exit mapping](f0-2-acceptance-ledger.md).

Sprint 011 adds integrity judging. Sprint 012 freezes A5 data/calendar/five date
windows. Sprint 013 adds Housing inputs/five splits for A1/A11; its initially
unresolved license later received a source declaration, with provenance limits
recorded in the [review](../benchmarks/v1/datasets/housing-license-review.json).
Sprint 014 adds A2 Adult data and official-test-preserving splits. Further data,
baseline/budget, worker, selection and sealed held-out preparation exist; see the
Sprint 017 ledger and Sprint 038 for remaining integration obligations. All real
quality gates remain open. Creating these files did not pass E0–E6; E7 has no new
independent-adoption evidence. Every application requires individual acceptance.

Current sequencing review: [Sprint 017 audit](017-f0-sequencing-audit.md).
It distinguishes existing F0.3 prerequisites from later evaluation results and
proposes a bounded CPU-construction overlap. The phase gate has not been changed.

The user approved the Sprint 017 overlap on 2026-09-06. Active construction:
[Sprint 018 / B03](018-b03-cpu-state.md). F0.3 remains incomplete; no scope or
acceptance threshold was removed.

[Sprint 019 / B04 operations](019-b04-numeric-operations.md) delivers numeric
preparation and shared scalar split operations.
[Sprint 020 / B04 trees](020-b04-depthwise-tree.md) adds depthwise assembly and
validated numeric tree inference/persistence.
[Sprint 021 / B05 squared](021-b05-squared-recipe.md) adds mapped tree transactions
and the first complete squared CPU recipe.
[Sprint 022 / B05 Normal](022-b05-normal-recipe.md) separates target/raw widths and
adds joint Normal ordinary/Fisher updates.
[Sprint 023 / B06](023-b06-formula-runs.md) adds saturation Formula/full GGN and
sequential heterogeneous execution probes.
[Sprint 024 / B07 growth](024-b07-growth-policies.md) adds best-first and symmetric
numeric policies. [Sprint 025 / B07 categories](025-b07-categorical.md) adds mixed
input, category equality and typed dictionary persistence.
[Sprint 026 / B08 binary](026-b08-binary.md) adds explicit class order and binary
logistic inference/training.
[Sprint 027 / B08 vectors](027-b08-vector-multiclass.md) adds joint multiclass,
vector leaves, separate split/leaf statistics and output mappings (630 tests).
B09 ranking/quantile/penalized leaves are next; full A6 workflows, CUDA and
quality/performance evaluation remain incomplete.

[Sprint 028 / B09 ranking](028-b09-ranking.md) adds query-local pairwise/lambda
geometry and fixed-step ranking (642 tests). Quantile/penalized routed leaves
are next; real A4 evaluation and the broader incomplete gates remain open.

[Sprint 029 / B09 quantiles](029-b09-quantile-leaves.md) adds routed residual
views, weighted quantile and anchored penalized leaves (655 tests). B10 positive
target/exposure and AFT construction is next; real application gates remain open.

[Sprint 030 / B10 Poisson](030-b10-poisson.md) adds explicit exposure/count
geometry and rate/count inference (665 tests). Gamma/A8 is next, followed by
Tweedie/composition/A9 and AFT/A10. Real A7 evaluation remains open.

[Sprint 031 / B10 Gamma](031-b10-gamma.md) adds weighted positive-target means
(674 tests). Tweedie/frequency-severity composition and AFT are next; real A8
evaluation remains open.

[Sprint 032 / B10 Tweedie](032-b10-tweedie.md) adds fixed-power nonnegative
means (684 tests). Frequency-severity composition and AFT are next; real A9
quality and complete application artifacts remain open.

[Sprint 033 / B10 composition](033-b10-frequency-severity.md) adds matched
paid-loss problems and persisted two-model inference (693 tests). AFT is next;
real A9 joins/quality and joint selection remain open.

[Sprint 034 / B10 AFT](034-b10-aft.md) adds explicit event/right-censored
targets and persisted fixed-scale survival inference (716 tests). Next audit
CPU/B11 coverage; full A6, extension tasks, real application results and CUDA
remain incomplete.

[Sprint 035 / CPU coverage audit](035-cpu-coverage-audit.md) maps all required
applications/capabilities and orders remaining work. Public production subset:
228 passing tests. Next A6 multi-output regression, then shared preparation and
independent stopping/M32. F1/B11 prerequisites are not complete.

[Sprint 036 / A6 multi-output](036-a6-multioutput.md) closes the independent/
shared recipe and persisted target-scaling gap (730 tests). Next: shared
preparation and independent validation-driven stopping/M32. Real A6 remains open.

[Sprint 037 / shared preparation](037-shared-preparation.md) reuses training
binning/codes across independent M1/8/32 runs (735 tests). Next: independent
validation-driven stopping; prediction-time caching and performance remain open.

[Sprint 039 / independent stopping](039-independent-stopping.md) adds public
validation patience across all twelve recipes, separate from transaction state,
with M=1/8/32 heterogeneous stopping/failure/retry equivalence. Next: installed
public D2/D3 extensions, ordered updates and current real-data integration.

[Sprint 040 / installed extensions](040-installed-extensions.md) verifies public
cohort-feasibility and penalized-leaf packages without core changes, including
independent math checks and fresh-process inference after plugin removal. This is
partial E2/E6 development evidence; formal agent/adoption results remain open.

[Sprint 041 / ordered updates](041-ordered-updates.md) compares both Normal and
Formula update orders with independent references and verifies installed execution
and inference after plugin removal. The OrderedResult/run_many incompatibility is
retained as the next D5 integration counterexample. No core edits or E5 claim.

[Sprint 042 / result contract](042-result-contract.md) resolves the counterexample
with structural validation of completed results and installed mixed M=1/8/32
equivalence. External result types and their diagnostic payloads are preserved.
