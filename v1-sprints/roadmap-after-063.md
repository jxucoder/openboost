# Sprint roadmap after the v1 retrospective

Planning baseline: `df23796`, 2026-09-06. Current execution: 064–065 complete, 066 next.
This decomposes Sprint 063 N1–N5 and its proposed GPU sequence into execution
cards. It does not replace the [main plan](../planning/agent-boosting-foundation-plan.md),
[construction design](../planning/foundation-construction-design.md), or
[E0–E7 protocol](../planning/openboost-v1-evaluation.md). All R1–R9/C1–C7/A1–A13 remain required.

The product decision remains: do public components reduce the total cost of a
correct algorithm change, while preserving useful quality and execution cost?
CPU runtime work, application results, author trials and CUDA each test a different
part of that hypothesis. Completing sprint cards is not a percentage of product validation.

## Sprint sequence and deliverables

Each card has an entry dependency, a first falsifiable check, work boundaries,
acceptance and a reflection question. Numbers identify cards, not calendar weeks
or a requirement to finish every lower number before starting independent work.
Sprints 064–065 are complete; Sprint 066 is the current execution assignment. Later cards are planned;
pin their actual source revision, commands and workload before execution.

| Sprint | Outcome | Entry dependency | Deciding evidence |
|---|---|---|---|
| [064](064-programmable-stopping-and-isolation.md) | Complete: public stopping completion | Current CPU contracts | Independent two-round policy, 77 focused / 943 CPU tests; installed checks follow in 065 |
| [065](065-installed-run-isolation.md) | Complete: installed independent runs | 064 | M=1/8/32 RNG, preparation, custom stop, failure/retry and plugin-free inference |
| [066](066-practical-cpu-profile.md) | Practical CPU cost diagnosis | 065 | Eight frozen Housing cases; enforced limits, uninstrumented cost and separate profiles |
| [067](067-incremental-runtime.md) | Incremental transaction execution | 066 diagnosis | New-term replay grows linearly; independent full replay and all state transitions agree |
| [068](068-trace-retention.md) | Bounded diagnostic memory | 067 | Summary/full equivalence; bounded retained arrays; matched practical rerun |
| [069](069-authoring-pilot.md) | Exploratory authoring measurement | Preparation can start now; attempts after 065 and accounting checks | Fair control/deep-change arms, measurable budgets and isolated verifiers |
| [070](070-coverage-and-judging.md) | Trustworthy coverage and selection infrastructure | Inventory can start now; execution preflight uses 068 | Missing evidence fails; test labels and resource caps actually isolated/enforced |
| [071](071-real-multioutput-selection.md) | First complete real selection workflow: A6/A13 | 068, 070 | Full frozen search, validation selection, sealed release, original-unit quality and cost |
| [072](072-regression-and-distribution-quality.md) | A1/A11 selected quality | 071 pipeline | Housing regression and Normal each judged independently |
| [073](073-classification-quality.md) | A2/A3 selected quality | 071 pipeline | Adult and full Covertype quality, probability semantics and resource outcomes |
| [074](074-ranking-and-quantile-quality.md) | A4/A5 selected quality | 071 pipeline; A4 source closure | Query-safe ranking and temporal quantiles, each with its own gate |
| [075](075-count-severity-aggregate-quality.md) | A7/A8/A9 selected quality | 071 pipeline | Counts, severity, direct Tweedie and joint composition selection in correct units |
| [076](076-survival-and-structured-quality.md) | A10/A12 selected quality | 071 pipeline; A10 provenance closure | Censored likelihood and real Formula/structural tests judged separately |
| [077](077-formal-author-evaluation.md) | Formal E5 and F2 decision | 069, 070, required CPU F0/F1 exits and frozen contracts/judges | Five D types, two sealed H types, three attempts per task/arm; correctness and cost gates |
| [078](078-cuda-scalar-path.md) | Resident scalar CUDA composition | 068 and F2 pass, or an explicitly adopted sequencing amendment | Intermediate parity, no loop fallback; prepare the original P7 protocol |
| [079](079-cuda-distribution-and-extension.md) | Resident distribution and external component | 078 correctness/residency | K=2 Normal with actual backtracking, a nondefault device component, and separate P7 reproduction |
| [080](080-cuda-required-recipes.md) | Remaining required CUDA recipe cells | 079 | R1/R4/R5/R6/R8 device matrix, persistence and task-metric parity |
| [081](081-cuda-train-many.md) | Compatible GPU train-many | 080 applicable cells, 065 semantics | Batched R1/R4/R8-compatible group, M=1/8/32 state/quality and failure isolation |
| [082](082-end-to-end-cost.md) | Formal E4 cost decision | 080–081 and quality-qualified selected workloads | Matched-quality standard, public/optimized, full-set and CPU inference gates |
| [083](083-engineering-v1-acceptance.md) | E0–E6 release-candidate audit and installed E6 | All required evidence from preceding engineering cards | Raw-artifact gate reconstruction, clean CPU/CUDA installs, honest public contracts |
| [084](084-independent-adoption.md) | Independent use and repeat use | Material preparation now; trials after F2 and authorized contact | Two external authors, one own method and second-task reuse; E7 separate from E0–E6 |

## Dependencies and work that can overlap

The immediate execution path is 064 → 065 → 066 → 067 → 068. Do not start an
unmeasured runtime rewrite. If 066 contradicts the proposed remedy, revise 067
using its profile before implementation. If a planned defect is absent, preserve
that finding and close only the relevant check; do not manufacture a code change.

069 preparation and 070 inventory/source work can proceed without waiting for
runtime optimization. Their experiments need stable, pinned inputs. After 071
proves the shared selection/release path, 072–076 are independent application
families; an unavailable ranking source must not stall Bike or another runnable
family. Each individual application still needs its own acceptance.

077 does not wait for every E3 quality comparison merely because its number follows
076. It does require all CPU capabilities/workflows in F1, the required F0 judging
prerequisites, and a frozen authoring environment. 070 records that entry audit
and every unresolved prerequisite; a coverage table alone cannot pass it.

CUDA entry retains the current F2→F3 rule. Sprint 063 recommends allowing bounded
078/079 feasibility after 065/068, but **that amendment has not been adopted**.
The earlier approved B03–B06 overlap is not authorization for B12. If an amendment
is adopted, record it in the main plan first and keep formal E5 cohorts immutable.
No acceptance threshold changes under either sequence. Once entered, CUDA work
can coexist with independent application quality work; formal E4 still needs
quality-qualified workloads. An execution track is not permission to spawn agents.

External trial material may be prepared early. Actual E7 contact requires explicit
authorization and does not block unrelated engineering. Existing authorization
for the sealed H1/H2 evaluator does not authorize a new author cohort or outreach.

## Complete scope accounting

| Required scope | Sprint responsibility |
|---|---|
| R1–R9 CPU, C1–C7 | Preserve existing implementation; 064–068 fix demonstrated boundaries; 070 audits every cell; 071–077 close workflows/author evidence; 083 reconstructs all gates |
| A1 / A11 | 072, with separate regression/distribution scores |
| A2 / A3 | 073, with separate binary/multiclass scores |
| A4 / A5 | 074, with separate ranking/quantile scores |
| A6 | 071, independent/shared multioutput and every target |
| A7 / A8 / A9 | 075, three application gates including joint aggregate selection |
| A10 / A12 | 076, separate survival/structured scores |
| A13 | 071 real selected quality/CPU execution, 081 device semantics, 082 full-set cost |
| Required CUDA R1/R4/R5/R6/R8; R9 compatible batching | 078–081; optional device cells remain explicitly separate |
| E0 / E1 / E2 | 070 indexes independent checks, every implementation card supplies affected evidence, 083 audits full required scope |
| E3 | 071–076 individually, joined across at least six independent sources in 083 |
| E4 | 082, with qualifying quality and current device correctness |
| E5 | 077; 065/069 are development evidence only |
| E6 | 065 supplies installed development checks; 083 verifies the complete CPU/CUDA delivery matrix |
| E7 | 084; no substitution with internal authors, downloads or stars |

Source/access closure for A4 and A10 starts in 070. Source substitution, if needed,
must preserve application semantics and be frozen before evaluation; poor scores
cannot trigger a favorable replacement. No new algorithm catalog, general compiler,
graph/RL/generative scope, Ray, multi-GPU or out-of-core work is added. Coupled
matrix leaves and adaptive mappings from the landscape review are optional
development probes, not new mandatory paper reproductions.

## Shared acceptance, evidence and sprint closure

1. At entry read the source, tests and docs together, pin the revision and write
   the exact focused check/CLI. An implementation sprint starts with an independent
   failing case; an evidence sprint starts with a falsifiable integrity/metric or
   resource check. Already-green behavior is not a newly fixed defect.
2. Prefer one reviewable outcome and one to three independently verified commits.
   Split a newly discovered substantial core fix from a benchmark run. If a card
   expands beyond its declared boundary, record the counterexample and add a
   follow-up card instead of silently expanding the sprint or dropping requirements.
3. Record raw artifacts under a sprint-specific path in benchmarks/v1/evidence/,
   or retain immutable existing parent artifacts with verified hashes. Include
   code/dirty state, data/splits/configuration/protocol hashes, environment, exact
   commands, time/memory enforcement and every failure. Plans and manual ledger
   rows do not count as evidence.
4. Use focused mathematical/state tests and relevant regression/lint for code
   changes, and installed/docs/build checks where their behavior is touched.
   Documentation-only planning checks links, strict docs and diff hygiene; it does
   not claim a new runtime pass. Inspect staged changes and commit locally.
5. At closure record observation → evidence → decision → next step in the card and
   a learning entry. Also reflect every three implementation commits, at a phase
   transition and after an architectural/correctness counterexample. State whether
   the work reduced authoring cost, preserved semantics, or only prepared evaluation.
6. Use the existing per-case statuses not_run/pass/fail/unsupported/error/timeout.
   A sprint may close with a diagnosed failure, but that does not pass its gate.
   Dependencies that require a pass remain unmet. Revisions after evaluation need
   fresh affected evidence/cohorts; do not edit old failures or mix incompatible runs.

Before formal searches, preflight the frozen 16 configurations per method and
five splits with the existing 300/1000-round spaces, trial time/RAM/thread caps,
and selected test-release rules. Estimate total jobs and maximum resource exposure
from that exact matrix before launch. Sprint 066's 120-second diagnostic cap is
not a replacement budget. Stop a blind matrix launch after a resource preflight
failure and investigate; never shrink a failed formal workload and retain its label.

These are scope cards, not promised durations. Use the first measured profile and
search preflight to estimate compute and calendar effort. Preserve failed trials
and include their cost. Modal is available for authorized bounded GPU work, but
this planning change launches no jobs and approves no new publication or outreach.

Engineering v1 requires every A1–A13 and all required E0–E6 cells passing. E7
independently validates adoption. No sprint count, objective count or green CI
replaces either decision.

## Planning result

The initial decomposition changed only planning files/navigation. Implementation
has since completed 064–065; follow the current card and its evidence. Original planning verification is recorded in the
[sprint-decomposition learning](../learnings/2026-09-06-v1-sprint-decomposition.md).
