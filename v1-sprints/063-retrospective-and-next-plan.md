# Sprint 063: Retrospective and next execution plan

Review baseline: merged [PR #24](https://github.com/jxucoder/openboost/pull/24),
`47108db5d9fe3157e59621115860aa1dd06e9bf4`. Date: 2026-09-06 (local).
Status: retrospective complete; next CPU work ordered; GPU overlap proposed,
not enacted. All R1–R9/C1–C7/A1–A13 and E0–E7 thresholds remain unchanged.

Subsequent [landscape feedback review](063-landscape-feedback.md) adds a concrete
external stopping-result probe to N1 and sharpens N3 comparator/task choices.
It does not replace the required application matrix or adopt the GPU amendment.

## 1. Verdict and product definition

**Continue investing in the foundation, but change what the next increments prove.**
We have credible internal evidence that a shared CPU substrate can express different
boosting algorithms. We have not established that unfamiliar authors can change
algorithms more cheaply, that practical training budgets are affordable, or that
this implementation can preserve those advantages on CUDA. More short application
fits would add less information than testing these three uncertainties directly.

OpenBoost is a programmable boosting foundation: readable Python algorithms over
public statistics, split/routing/leaf operations and explicit run transactions,
with verifiable execution and reusable inference artifacts. Its intended benefit
is lower total cost from an algorithm idea to a trustworthy result. Python is the
authoring surface; CPU/CUDA kernels are execution mechanisms. Python purity,
objective count and a familiar fit/predict wrapper are not the value proposition.

The torch analogy remains useful for composable operations, ordinary programs and
explicit execution. It does not require general autograd, arbitrary-Python GPU
compilation, a tensor framework or a universal training graph. Our domain-specific
contracts should make boosting changes easier to express and easier to check.

The strongest initial candidate for adoption is an author who needs to change a
split rule, routed leaf solve, coupled/ordered parameter update or run schedule.
A packaged method is the entry point; repeating a different change with the same
components is the retention test. This is a hypothesis about the first user, not
customer evidence or a reason to remove any required application family.

## 2. What we should preserve

| Decision or result | Evidence | What it actually supports |
|---|---|---|
| Clean redesign with independent math retained | [Reset](002-retire-legacy-production.md), [reference exit](f0-2-acceptance-ledger.md), current public modules | We can change physical representations without preserving obsolete APIs; historical failures remain reproducible |
| Recipes share operations and state | [Construction design](../planning/foundation-construction-design.md), [recipes](../src/openboost/recipes.py), [trees](../src/openboost/tree.py) | Scalar/vector, distributional, structured and routed-residual consumers fit a common semantic foundation |
| Explicit units, identity and transactions | [A6 scaling](047-multioutput-quality.md), [paid events](057-paid-event-binding.md), [stopping](039-independent-stopping.md) | Application details exposed real semantic requirements; they were not disposable benchmark overhead |
| External package composition found a real boundary failure | [Ordered updates](041-ordered-updates.md), [result contract](042-result-contract.md) | Replacing a built-in result-class assumption with a structural contract improved reuse; exploratory counterexamples changed the core |
| CPU optimization followed an unchanged failing input | [Profile](050-covertype-profile.md), [hash fix](051-candidate-row-hash.md), [histogram reuse](052-histogram-gather.md) | We removed redundant work while preserving candidate identities, reductions and replay; all five bounded jobs then passed |
| Failures and claim boundaries survived later success | Raw [Covertype evidence](../benchmarks/v1/evidence/histogram-052/README.md), [Sprint 062](062-cpu-exit-and-gpu-entry.md) | A timeout did not become a speed win, and real validation plumbing did not become quality acceptance |

PR #24 passed 923 local CPU tests, four Linux/macOS Python 3.10/3.12 CI jobs,
documentation, lint and package builds. These are engineering checks of that
revision, not 923 independent validations of the product hypothesis. Installed
extension-wheel evidence has a narrower recorded environment than the CI matrix.

The last nine implementation/integration slices, 053–061, mostly connected existing
components to application workflows. This is encouraging for internal reuse.
It does not establish independent authoring cost: the implementer already knew
both the architecture and evaluation harness.

## 3. Where execution drifted

### We accumulated coverage faster than decision evidence

Sprints 038 and 062 already identified the missing author, quality and GPU results.
Repeated continuation favored the next local adapter with a clear pass condition.
That was useful until the shared contracts had broad consumers; it now risks an
indefinite sequence of green integration checks while the central hypothesis stays
unmeasured. The error is treating every useful task as equally useful next work.

The correction is to give each milestone a decision it resolves: keep or change an
API, keep or change state ownership, enter or postpone CUDA, or invite a trial.
Three implementation commits without closing such a decision trigger a review.
Another status document alone is not a learning milestone.

### Known correctness-first tradeoffs became the default execution path

The [B05 learning](../learnings/2026-09-06-v1-b05-normal-recipe.md) explicitly accepted
recomputed predictions and retained trace arrays. That was reasonable for small
independent fixtures. It is still how current recipes run: an implementation
optimized for semantic checking is now being asked to serve practical workloads
without a resource acceptance test. The independent mathematical oracles remain
separate from production.

The new [runtime diagnostic](../benchmarks/v1/evidence/runtime-audit-063/README.md)
counts complete fits, not wall time. On 32 training/16 validation rows, squared
training uses 60, 216, 816 and 3,168 tree-prediction calls at 4/8/16/32 rounds;
Normal uses twice those counts. Final raw caches replay exactly in all eight cases.
The count is quadratic in rounds on these fixed-step paths. Step records retain
round arrays, with storage growing with rounds, rows and parameter width.

The call path explains this: [propose_terms/resolve/AcceptedState](../src/openboost/runtime.py)
replay candidate models; [_trials](../src/openboost/recipes.py) evaluates the whole
candidate again; [Tree.predict](../src/openboost/tree.py) transforms input features
on every call. Model construction and identity also revisit model contents.
Only tree prediction counts and step-array logical bytes were measured here;
these observations do not quantify total time, peak memory or GPU overhead.

Four-round smoke success is therefore an inadequate proxy for the frozen
[300/1000-round search families](../benchmarks/v1/search-design.json).
The earlier full-input profile still correctly prioritized histograms and row
hashing on its workload. We should measure the longer-round regime, not rewrite
that historical conclusion or assume the next bottleneck without profiling.

### CPU semantics are stronger than the current GPU execution shape

The current candidate representation is a tuple of Python records with scalar
score/feasibility callbacks. Owned arrays and validation use NumPy, and RunContext
rejects non-CPU devices. These are workable CPU interfaces; replacing `np` with
`cp` would not produce resident CUDA execution. Per-candidate Python decisions,
host conversions and full-ensemble state reconstruction need explicit treatment.

The design already calls for bulk candidates, routing and device state. Delaying
all device experiments until a large formal author study passes risks freezing
interfaces before discovering their physical execution constraints. This is a
sequencing concern, not proof that the semantic decomposition is wrong.

### Evidence has many producers and no complete acceptance consumer

[judge.py](../benchmarks/v1/judge.py) verifies artifact integrity and returns empty
E-gate results; [quality_report.py](../benchmarks/v1/quality_report.py) deliberately
keeps E3 false. The current search demonstration is synthetic A6. These are honest
boundaries, but adding more producers will not close them. We need a required
coverage inventory joined to actual verifiers and selection receipts.

The [local runner](../benchmarks/v1/process_runner.py) enforces time/threads, not RAM
or filesystem isolation. A hash seal can prove which files were selected; it cannot
prove a worker lacked access to test labels. Formal judging needs those execution
boundaries, without imposing them on every internal development smoke.

### Navigation accumulated contradictory-looking historical instructions

AGENTS and sprint navigation had grown into chronological lists of old "next"
steps. The main plan still has historical unchecked items for behavior delivered
later. That makes an agent repeatedly rediscover status or mistake a historical
pending item for current work. Preserve history, but make one concise current
execution pointer authoritative. This slice updates that navigation; it does not
retroactively mark old evidence passing.

## 4. Position against the actual alternatives

We should not sell "custom objectives in Python" as the differentiator. Current
[XGBoost documentation](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html)
exposes custom objectives/metrics. [LightGBM's Booster API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)
includes custom updates, rollback and leaf-output modification. [CatBoost](https://catboost.ai/docs/en/features/custom-loss-functions)
also documents user-defined objectives and metrics. These pages were rechecked
for this review; documented availability is not a task-specific execution result.

[Py-Boost](https://github.com/sb-ai-lab/Py-Boost) is a direct comparison: its repository
describes Python GPU boosting with CuPy/Numba and customization of training,
sampling, losses and multioutput behavior. Its stated scope overlaps our premise.
We need appropriate comparisons against its actual supported path, not only the
most restrictive way to use XGBoost. The prior [release review](../planning/boosting-release-review-2026-09-05.md)
remains the dated version/roadmap snapshot; this review did not install new opponents,
rerun their capabilities or replace frozen versions.

Our candidate distinction is composability plus verification across multiple
internal changes, with sufficient end-to-end speed and deployable outputs. It
remains unproven. An incumbent-friendly objective control may show no advantage;
that is expected evidence about the boundary. D2/D3/D4 and D5 test deeper changes,
but cannot be declared wins before appropriate controls are attempted.

Nor can arbitrary agent changes be guaranteed to beat XGBoost. Validation-only
selection can retain a strong baseline in a workflow; that protects selection
value, not a claim that OpenBoost components reproduce all incumbent capabilities.

## 5. Current capability and acceptance position

| Area | What is present | What remains decisive |
|---|---|---|
| CPU foundation | Typed data, binning, named fields, split/route/leaf operations, three growers, scalar/vector and residual leaves | Complete CPU E0/E1/E2 mapping; practical runtime/storage behavior |
| CPU recipes/runtime | Twelve recipes, composition, immutable joint/ordered transactions, independent stopping, preparation reuse and sequential M=1/8/32 | Remaining installed D5 probes; no fused/parallel training claim |
| Authoring | Four repository-authored installed packages covering D1–D4 and partial D5 integration | An unfamiliar author's completion cost; frozen E5 comparisons and sealed held-outs |
| CUDA | A construction design and explicit unsupported behavior | No current device ops, resident recipe, parity, batching or cost acceptance |
| Delivery | Standard and specialized persisted inference; fresh-process checks; CPU CI/build matrix | Complete installed workflow matrix and current-version reproducibility |
| Value/adoption | Real validation integrations, selected synthetic workflow | Real selected quality/full cost and independent repeated use; no E7 result |

Keep an individual application backlog:

| ID | Current evidence | Required next acceptance work |
|---|---|---|
| A1 | Five housing validation fits/replays | Real search, selected test RMSE and matched comparator |
| A2 | Five Adult validation fits/replays | Selected log-loss/AUC, class/missing semantics and comparator |
| A3 | Five full Covertype fits within the original short-fit cap | Practical-budget search, per-class quality and full cost |
| A4 | Synthetic pairwise/lambda adapter and source-ID tie checks | Resolve source/access terms, freeze official queries/folds, real NDCG@10 and quadratic-pair budget |
| A5 | Five rolling Bike origins, independent quantiles/replay | Selected pinball at each declared level and crossing/coverage-width reporting |
| A6 | Five grouped Parkinsons folds with verified scales | Selected per-target and standardized quality; independent/shared comparison |
| A7 | Five count/exposure folds with exact replay | Selected Poisson deviance, totals and declared group comparisons |
| A8 | Five positive claim-level folds with exact replay | Selected Gamma deviance and explicit eligibility/weight semantics |
| A9 | Five direct aggregate and matched composition folds | Joint aggregate selection across components, Tweedie/GLM controls and aggregate diagnostics |
| A10 | Five fixed-scale event/right-censored AFT folds | Source-license closure, selected NLL, applicable IPCW/calibration and supported time outputs |
| A11 | Five Normal housing folds | Ordinary/Fisher/adaptive variants, proper scores and calibration/width against suitable controls |
| A12 | Five concrete Formula folds, age separated from tree inputs | Structural/ordinary controls, interpolation/extrapolation and identification/misspecification evidence |
| A13 | Synthetic 16-trial A6 selection; shared-preparation scheduling semantics | Real full search/seal/release, selected task quality, M=1/8/32 full-set CPU/CUDA costs |

These are next acceptance obligations, not a declaration that every auxiliary is
absent. Existing references/metric helpers should be reused. All thirteen remain
required; ordering work on one source first does not drop the other twelve.
The complete requirements remain in the [application contracts](../planning/foundation-application-contracts.md)
and [evaluation](../planning/openboost-v1-evaluation.md).

## 6. Execution plan: smaller milestones that settle larger questions

### N1: Close the installed scheduling gap

Start here under the current plan. Extend the existing installed verifier with
same-seed/different-run-ID streams, exact retry/permutation, changed feature content
with unchanged row IDs, changed row identity and valid fresh preparation. Preserve
mixed K=1/2, M=1/8/32, different stop rounds and retained failures.

Acceptance: installed public wheels from outside the checkout; both stale cases
fail explicitly, fresh preparation matches independent direct runs, cross-ID streams
differ and same-ID streams match. Record source/wheel hashes and errors. No new
scheduler or core edit unless a failing probe demonstrates a defect. This closes
specific development evidence, not the full E5 experiment.

The [landscape addendum](063-landscape-feedback.md) found that result validation
still requires the concrete patience/budget StopState. Add an external terminal
record with a true policy-specific reason to this development slice. Preserve
unfinished-result rejection and existing stopping semantics; fix only the
demonstrated contract boundary. This probe is not a ScoreStop implementation.

### N2: Make practical execution an explicit design test

Before large searches, run a preregistered diagnostic on frozen Housing fold zero,
without test labels. Use the first 8,192 training rows in frozen order and 1,024
validation rows, retaining source IDs; hash those prefixes before execution.
For squared and Normal, use 4/32/128 rounds, 32 bins, depth two, learning rate 0.1
and regularization 1. Repeat 32 rounds with 2,048 training rows. These eight
fixed-step cases separate round growth from row growth. Add tiny deterministic
forced-rejection/backtracking fixtures; do not depend on a real fit accidentally
rejecting. Unsupported subset sizes fail explicitly rather than changing silently.

Each case has a 120-second hard cap, two CPU threads and an enforced 8-GiB RAM cap.
Record uninstrumented end-to-end time/peak RSS and separate instrumented stage/call
counts; profiling time is not an official timing result. Retain partial/timeout
outcomes and stop further expensive cases after the first resource failure pending
diagnosis. Record actual enforcement in the environment manifest. This diagnostic
budget is separate from the existing 300/1000-round E3/E4 searches and changes none
of their configurations or thresholds.

Use those results and the present operation-count counterexample to implement a
bounded transaction/diagnostic change if justified:

- Keep accepted raw state and compute candidate deltas from new terms; rejection
  discards candidates, acceptance advances train/validation caches atomically.
- Reuse fitted encodings during training prediction with data/binning/layout
  identity checks. Different binning/schema must miss or fail explicitly.
- Provide summary diagnostics retaining round decisions/scalars, with full arrays
  explicitly available for verification. Do not disable validation to gain speed.
- Keep full-model replay as the independent oracle/export check. Bind caches to
  immutable parents/terms; callers cannot submit arbitrary unchecked raw caches.

Acceptance: forced failure/rejection, ordered/joint updates, stale state, offsets,
best/stop and numeric/missing/category/vector persistence remain correct. For the
fixed-step counting fixture, training tree replay must grow linearly with new
terms, not quadratically with history. Summary mode retains O(N*K) run arrays
plus O(T) scalar trace and model storage, rather than O(T*N*K) diagnostic arrays.
Full/summary modes must produce the same accepted/best state and final inference.
Keep the CPU oracle independent; do not freeze a general cache framework first.

Bound this milestone to a diagnostic and one cohesive runtime design. If the
change expands beyond state/delta/diagnostic ownership, record the counterexample
and revise the milestone instead of starting a second foundation rewrite.

### N3: Obtain the first comparative authoring evidence

Prepare a frozen, clearly **exploratory** accounting smoke with one incumbent-friendly
control and one deep change selected from the existing development cards. Prefer
D2 candidate constraints or D4 ordered recomputation for the deep task; choose the
opponent using actual supported hooks/outer-loop/source paths. Do not inspect H1/H2.

Acceptance: a complete attempt record per arm includes install, first correct
result, assistance, private/core edits, tokens, blocked time and failure reason.
Run a verifier that the candidate cannot edit. Before dispatch, verify that the
available execution service can measure the declared token/time budgets and isolate
attempts; otherwise report the measurement gap rather than inventing agent costs.
This is an evaluation plan, not authorization for additional agents or outreach.

Use the [landscape addendum](063-landscape-feedback.md) when choosing the opponent:
GBNet for differentiable composition, current NGBoost authoring/learner tools for
distribution tasks, and Py-Boost for editable GPU execution. Coupled matrix leaves
and learned learner mappings are development options, not three new mandatory
paper implementations or replacements for the existing D/H evaluation scope.

An author unfamiliar with the implementation is preferable to another self-authored
package. Execution must use an authorized independent runner/agent or participant.
Do not score the designer's work as independent. Fix a demonstrated API/docs problem,
then freeze a new cohort; do not add hints to a failed attempt and count it as a
clean success. One smoke is not E5 and does not support the 30% benefit claim.

### N4: Close the evaluation loop on one real workflow, then fill the matrix

Create one current required-coverage ledger joining R/C/A cells to source revision,
verifier, raw artifact, environment, status and remaining action. E-gate entries
must distinguish missing evidence, failed evidence and verified evidence. The
inventory is a reporting layer over existing tools, not another universal runner.
Missing/duplicate/stale or producer-omitted required cells must not yield a pass.
Manual rows are navigation until a verifier actually evaluates their conditions.

Then complete one real A13 workflow on the existing A6 grouped data: train-only
preparation/scaling, the frozen 16 configurations per method, validation selection,
sealed test release, selected original-unit predictions and all required metrics.
Preflight the exact frozen search budgets; do not quietly substitute four rounds
or shrink data if a worker times out. Failed trials remain failed and block claims
according to the existing protocol. Record total selection cost, including failures.

Resolve A4 source/access and Veteran provenance as explicit items, without placing
unrelated runnable work behind access uncertainty. If access remains unavailable,
use the existing documented source-substitution process before evaluation with
equivalent semantics and frozen new provenance; never choose a source from favorable
scores. Keep those application rows open until their own checks pass.

Extend the proved selection/release path to the remaining application schemas and
A9 joint composition selection. Final selection/judging gets enforced test-label
separation and RAM limits. A valid selection receipt plus a paired score still does
not certify all E3; the full required matrix must join them.

### N5: Formal author, quality and delivery completion

Once relevant CPU contracts and judges are frozen, execute unchanged E5: five
development types and two sealed held-outs, three independent attempts per task/arm,
30 minutes or 20k generated tokens per attempt. Retain failures and appropriate
opponents; require the existing per-task correctness and deep-change cost gates.
Separate cohorts after interface or model/settings changes. No held-out inspection
by the foundation designer or reuse of exposed tasks as unseen results.

Complete E3 individually across A1–A13 and E6 installed workflows. Do not use
cross-task averaging to hide a failed application or use local CI as installed
extension coverage on every platform. Prepare a concise install/change/verify/save
trial package now; external contact waits for user authorization. E7 still needs
two independent authors and one author's repeated use on a second task.

## 7. Recommended GPU sequencing amendment

**Recommendation: allow bounded B12 feasibility to overlap N3–N5 after N1 and the
N2 state/diagnostic ownership check.** This differs from Sprint 062's default of
finishing formal F2 before entering F3. The reason is concrete: CUDA feasibility
constrains candidate layout, state caching and diagnostics, while a full formal
study is not a technical input to implementing one resident composition.

This review proposes the amendment; it does not mark it approved, start CUDA,
freeze interfaces or declare F1/F2/F3 complete. Under unchanged authorization,
continue N1–N4 CPU/development work and follow existing F2→F3 entry rules.
If approved, record the amendment in the main plan before device implementation.
All existing application, author, parity, cost and adoption obligations survive.

| Decision | Current ordering | Proposed bounded overlap |
|---|---|---|
| First resident CUDA experiment | After formal CPU/author prerequisites | After N1 and the relevant N2 ownership checks; development interfaces may change |
| Formal E5 cohort | Frozen CPU/interfaces/judges | Still frozen; use an immutable revision and separate cohort if GPU findings change public semantics |
| GPU acceptance or performance claim | Required E1/E3/E4 and full device scope | Identical requirements; a feasibility pass is only a technical milestone |
| Required application completion | Every A1–A13 | Unchanged; no representative-only v1 exit |

First device work should fit squared error end to end, immediately followed by
Normal with K=2 and actual bounded backtracking. Keep a public bulk representation
for fields/candidates/routing and explicit device buffers, stream, workspace and
capacity. Use CuPy/numba-cuda initially as designed. Keep CPU semantic fixtures;
physical layouts may differ. Scalar CPU callback objects need an explicit bulk
counterpart or unsupported capability error, not per-candidate transfers.

Use authorized Modal on one real GPU with a declared hard budget and provenance.
Before running, freeze the workload, precision/tie policy, dependencies, cold/warm
compilation policy and instrumentation. Do not quietly replace the already frozen
T4 E4 environment; a different feasibility environment is a separate record.

Acceptance of the initial feasibility slice:

- At least two rounds; compare gradients, curvature, additive fields, candidate
  optimality/ties, routing, leaves, raw states, predictions and task metrics under
  existing E1 tolerances. Numeric missing values are included.
- Inputs may be prepared on CPU and uploaded once with costs recorded. Raw updates,
  statistics and routing stay on device; no bulk CPU fallback in the training loop.
- Accept/reject, best/stop, ordered-state boundaries and keyed RNG survive; count
  metric synchronizations and rejected-trial work. CPU inference reads the export.
- Record full fit/predict, preparation, transfers, JIT, sync, workspace and device
  peak memory. A fast histogram alone does not pass.
- Abort expansion if ordinary recipes require private bypasses or if standard
  state updates require full ensemble host replay. Revise the shared boundary.

Next prove a nondefault public component on device, then all required R1/R4/R5/R6/R8
subsets and a compatible R9 group. M=1/8/32 batching follows independent same-ID
parity; reuse-only savings and actual batching savings are reported separately.
Preserve the original historical P7 reproduction requirement and threshold in
its own record. Do not count old CUDA code or measurements as current support.
Ray, multi-GPU and out-of-core stay outside this plan.

## 8. Acceptance and resource discipline

The north-star measure is total cost to a correct reusable algorithm change,
followed by that change's selected quality and full execution cost. Track these
separately; do not invent a percentage of v1 completion from file/test/sprint counts.

| Question | Deciding result | Action on failure |
|---|---|---|
| Is a change expressible through public components? | Installed D1–D5 independent math/state verification, plus held-out scope under E5 | Fix the demonstrated boundary; count exposed tasks as development |
| Is authoring cheaper? | Unchanged E5 correctness and at least two deep-change capped-median cost wins | Investigate docs/API/opponent path; retain controls and failed attempts |
| Can practical runs fit resource limits? | Long-round diagnostic, then frozen real search outcomes with time/RAM enforcement | Address measured repeated work or bound unsupported scope; no silent budget change |
| Does CUDA preserve composition? | Resident scalar/K>1/nondefault parity and visible transfers/failures | Redesign the narrow execution boundary before more kernels |
| Is GPU execution useful? | Existing E4 matched-quality cost and M=1/8/32 gates | Keep experimental status; no speed or complete-GPU claim |
| Is task value real? | Every A1–A13 passes its own E3/selection criteria | Report the failing task; do not remove it or average it away |
| Is adoption real? | E7 independent completion and repeated use | Learn why users stopped; stars/internal attempts do not substitute |

Before formal jobs, retain the existing 16-configuration and per-trial time/RAM/thread
budgets. Diagnostic pilots get separately named, frozen budgets and never change
formal thresholds. Estimate total jobs and maximum compute from the selected matrix
before launching; one failed resource pilot should stop a blind full-matrix launch.
No financial or calendar estimate is justified before that measurement.

Work in one implementation slice at a time, with at most one pending experiment
whose result changes the next slice. Planning tracks are not authorization for
parallel agents or external jobs. End each milestone with: decision resolved,
strongest counterexample, evidence path, remaining uncertainty, next decision.

Do not schedule another objective catalog, a general compiler/registry, cosmetic
wrapper expansion or more repeated four-round adapters without a named failing
consumer or unfilled required cell. Preserve all correctness fixtures and raw
artifacts. Improve their index; do not rewrite history to simplify the story.

## 9. Verification and next handoff

This review read the active design/gates, current runtime/recipes/ops/inference,
installed scheduler, search, integrity/quality layers and selected raw sprint
artifacts. The new diagnostic ran eight tiny CPU fits and checked exact raw replay;
no new real-data fit, GPU run, formal author attempt, held-out inspection or test
selection occurred. Public documentation was checked for the comparator claims
above; no new opponent version was installed or scored.

Verification results are recorded in the [learning entry](../learnings/2026-09-06-v1-deep-retrospective.md).
The package/CI result cited above belongs to merged PR #24. This review changes
planning/navigation and adds an observational diagnostic, not production behavior.

**Next implementation: N1 installed D5 probes.** Then N2 practical runtime/state
work and N3 exploratory measurement preparation. N4 has a concrete real A6/A13
closure target instead of another undifferentiated adapter queue. The bounded
CUDA overlap is the recommended sequencing decision for the user; until adopted,
the existing phase dependency remains in force.
