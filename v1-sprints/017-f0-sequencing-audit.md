# Sprint 017: F0 prerequisite and sequencing audit

Audit base: `8565f77` (clean worktree). Status: audit complete; proposed sequencing
amendment not adopted. No production implementation or remote publication in this slice.

## Question and verdict

Must all remaining evaluation work finish before building the foundation?

No final E0–E7 result is required before F1. However, the current written F0.3
requires more than task cards and mathematical oracles: it explicitly requires
protocol/matrix/environment/agent freezes, adapters, runner and judges. F0 exit
also says missing data or judges prevent exit and F0.1–F0.3 must precede F1.
Consequently **F0.3 is not complete and F1 is not authorized by the current phase
gate**. The earlier suggestion that we could simply treat all comparator and
cohort work as later work understated that gate. Starting B03 early requires an
explicit sequencing amendment; this audit does not silently enact one.

Engineering assessment: the missing global evaluation preparation does not all
block construction of Numeric Problem, RunContext, AcceptedState and minimal
artifacts. Requiring every evaluation adapter before testing these abstractions
has delayed the product hypothesis: composable components should make correct
algorithm changes easier. Three data-binding slices improved fairness, but did
not test whether the proposed foundation works.

## Evidence inspected

- [Active plan F0–F5](../planning/agent-boosting-foundation-plan.md): F0.3 freeze/
  implementation gate; F1 CPU correctness, F2 agent trials, F3 device/cost, F4
  real quality/adoption and F5 delivery.
- [B02–B14 construction order](../planning/foundation-construction-design.md):
  B03 starts public implementation; real evaluation follows stable recipes.
- [E0–E7 protocol](../planning/openboost-v1-evaluation.md) and
  [A1–A13/D1–D5 task contracts](../planning/foundation-tasks.md).
- [F0.2 acceptance ledger](f0-2-acceptance-ledger.md), reference implementations,
  actual worker/exporter/selection/quality/integrity call paths and tests.
- Frozen dataset records, search design, CPU/CUDA locks, comparator artifacts and
  safe held-out manifest. Held-out task/verifier contents were not inspected.
- `src/openboost/` contains only `__init__.py` and `py.typed`; public v1 training
  and prediction components have not been implemented.

## Findings and phase assignment

“Before F1 as written” reports the existing plan, not a claim that the dependency
is technically necessary for B03. Later deadlines below describe execution of
results; protocol definitions must precede those results.

| Item | Actual state | Before F1 as written? | Necessary deadline / recommendation |
|---|---|---|---|
| A1–A13/R1–R9/C1–C7 contracts and independent math/state references | F0.1/F0.2 delivered; all families mapped, not production support | Yes; satisfied at specification/reference level | Use immediately as B03–B11 conformance oracles |
| Real data and split identities | Frozen inputs for A1–A3/A5–A12; A13 explicitly reuses Covertype/Housing | Yes; incomplete for A4 | Resolve A4 source/agreement and Veteran license record; do not invent a separate missing A13 dataset |
| CPU comparator capability and budgets | Pinned libraries, CPU probes, 16-config families, 1800-second/8192-MiB/two-thread budgets exist | Yes; substantial partial evidence | Stop describing all baseline/budget preparation as absent; complete task binding before its scored search |
| CUDA comparator capability | Real T4 preflight exists; LightGBM native build/isolation gaps recorded | Yes for environment freeze | Full native build closure is necessary before formal F3/E4 comparison, not Numeric Problem construction |
| Real numeric worker wiring | Eleven applications, five folds each exercised across committed artifacts; four-round validation plumbing | Adapter preparation belongs to F0.3 | Preserve evidence; no need to rerun every preceding cell merely to begin a new CPU component |
| A4 ranking wiring | Synthetic query-aware workers tested; real dataset unresolved | Yes; incomplete | Source first, then real query binding; retain ranking in B09 and full v1 scope |
| A13 scheduler/comparison wiring | Reused datasets and independent M=1/8/32 state oracles exist; real workload/scheduler/cost execution incomplete | Freeze/adapters yes; final execution no | Freeze workload membership before timed evaluation; build sequential OpenBoost semantics at B06, fused/device work at B13 |
| Composed A9, outer/coupled A11/A12 and other comparator paths | Parametric workers/diagnostics exist; complete real integrations do not | Adapter preparation is part of current F0.3 | Match controls before each scored quality/author comparison; cannot substitute ordinary GBDT for these algorithms |
| Global expected matrix | Local smoke matrices and integrity schema exist; no complete frozen recipe/task/device/fold comparison matrix found | Yes; incomplete | A real remaining protocol deliverable, not another batch of four-round fits |
| Independent gate evaluation | `judge.py` returns integrity only and empty gate results; `quality_report.py` always sets E3 false; selection seal/audit works for its declared packet | Yes for judge implementation; incomplete | Define fail-closed gate aggregation and required evidence; execute CPU conformance during F1, other gates in their assigned phases |
| H1/H2 preparation | Separate evaluator authored sealed package; hashes/status committed; local custody and independence limitations explicit | Yes; partially satisfied | Do not call the cards missing. Keep contents out of design; finish restricted execution and comparator/cohort preparation before formal F2 |
| Agent cohort and accounting | Numeric attempt limits fixed; model/reasoning/tools/docs/cache/accounting cohort not bound | Explicitly yes; incomplete | Cohort must freeze before scored F2.2; it is not an input to B03 math/state semantics |
| Full 16-trial quality searches and test results | Not completed; four-round smoke is not selection evidence | Results: no | E3/F4 after runnable recipes; comparator searches may run earlier once protocol/data/environment are frozen |
| CPU E0/E1/E2 success | References exist, public conformance does not | No; needs production code | F1 exit, not F0 entry requirement |
| GPU parity, end-to-end cost and fused train-many wins | No new OpenBoost implementation/results | No | F3/E4; no device or speed claim before passing |
| Agent advantage, package delivery, independent repeated adoption | No formal results | No | F2/E5, F5/E6, F4/E7 respectively; outreach still requires authorization |

## Concrete gaps found in the implementation audit

1. **Global freeze/aggregation is the missing integration deliverable.** Passing
   local worker cells cannot establish complete R/C/A coverage. The integrity
   judge validates a producer-declared matrix; it does not determine full recipe
   coverage or evaluate E0–E7. Quality reporting deliberately refuses E3 closure.
2. **A6 final scoring is not fully wired to the frozen scale.** Workers now save
   train-only normalization correctly. `quality.py` reports per-target RMSE;
   `quality_report.py` has no standardized-average output. `selection.py` permits
   positive selection weights but does not verify them against the training-scale
   artifact. Its comment is not enforcement. Fix this before a scored A6 search;
   it does not block constructing B03 state records.
3. **Resource and test-access enforcement differ from declarations.** Search
   budgets are present, while the local runner enforces time/threads, not its RAM
   cap or filesystem isolation. Separate test files and a hash seal do not prove
   that candidate processes could not access test labels. Formal runs need an
   execution boundary and provenance, not an invented pass from local smoke.
4. **Some status text is stale.** Historical sections still say held-outs or
   train-fitted encoders are missing, despite later artifacts. Use this ledger and
   artifact scope rather than treating every old “pending” sentence as new work.
   Do not rewrite old failure artifacts or turn later success into retroactive passes.

## Recommended sequencing amendment (proposal only)

Allow **B03–B06 CPU architecture construction** to overlap the remaining B02
preparation. Keep F0.3 marked incomplete until its explicit ledger closes. Do not
allow CPU interface freeze, formal agent comparisons, quality/speed claims or
v1 completion through this overlap. All R1–R9/C1–C7/A1–A13 scope stays required.

The bounded order would be:

1. Record adoption of this sequencing amendment in the active plan and define
   the first CPU conformance mapping. No threshold changes or source substitutions.
2. Build B03: Numeric Problem, RunContext, AcceptedState and minimal artifact.
   Acceptance: distinct run IDs, read-only input ownership, weight/offset applied
   once, rejection leaves accepted/best/RNG state unchanged, corrupt-state failure
   and fresh-process roundtrip. Use existing independent state/persistence oracles.
3. B04/B05: one shared split/route/leaf pipeline, then squared-error and Normal
   recipes. Require two-round intermediate agreement, accepted/rejected updates,
   predictions and persistence, with a replaceable operation through public APIs.
4. B06 immediately probes Formula and heterogeneous sequential train-many state.
   Do not stabilize interfaces around only scalar GBDT. Then complete B07–B11.
5. Finish the global matrix, source gaps and independent gate/selection binding
   alongside the appropriate implementation slices; freeze each formal cohort
   before measurement. Full quality and GPU/adoption work stays in F2–F5.

If the amendment is not adopted, follow the existing gate: finish B02's source,
full matrix, judge, comparator and cohort/environment preparation before B03.
That is the literal plan, but not the recommended engineering critical path.

## Verification and limits

Read-only source/artifact audit plus documentation checks; no new model fits or
production changes. Prior committed evidence records 488 passing v1 tests and
55 distinct application/fold numeric plumbing cells across 11 applications;
these are not new audit test runs or complete quality gates. No remote fetch,
push, external publication, held-out inspection or threshold change occurred.
