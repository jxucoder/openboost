# Sprint 093: Foundation checkpoint and proposed next steps

Status: planning checkpoint requested before further execution. Baseline `8a70e0b`.
No implementation, upload, device invocation, author attempt or budget change is
part of this review. [Run 8](092-comparison-run8-request.md) remains pending.
This checkpoint orders existing work; it does not replace the canonical R/C/A/E
scope or grant new execution allowances.

## Product goal and current assessment

OpenBoost is a programmable boosting foundation for researchers and agents.
A useful public boundary lets an author change objectives/geometry, statistics,
split feasibility, leaves, update order or run scheduling while reusing verified
data, tree execution, state and persistence components. Standard boosting,
Normal/NaturalBoost-style methods, FormulaBoost and train-many test those
boundaries. Every required application A1–A13 remains part of v1.

We have substantial CPU implementation and narrow real-device correctness
results. We have not established lower independent authoring cost, competitive
end-to-end GPU cost, complete real-task quality or adoption. Test counts describe
regression coverage; they are neither a completion percentage nor product evidence.
CPU supplies the semantic reference and an accessible development path. Practical
runtime matters, but speculative competition with mature CPU trainers stays paused.

## Evidence at this checkpoint

| Layer | Implemented or observed | Limit of the evidence |
| --- | --- | --- |
| CPU foundation | Public data/targets, named statistics, histogram/candidate/routing/leaf operations, three growers, scalar/vector mappings, transactions, stopping and saved inference | Twelve recipes and 1794 recorded passing CPU checks, one Linux-only skip; formal full-scope and quality gates remain open |
| CPU recipes | Regression/classification, ranking, quantiles, counts/positive targets, AFT, Normal, Formula and multi-output, plus frequency–severity composition | Individual A1–A13 real evaluation is incomplete; recipe existence is not quality parity |
| CUDA components and squared training | Python-authored numba-cuda kernels, CuPy-owned storage/streams, fields, histograms, feasibility/routing/leaves, resident rounds and CPU model export | All 212 bounded T4 cases pass at `af026ef`; this historical revision is verified, the current changed revision awaits regression |
| CUDA Normal and external D2 | K=2 ordinary/Fisher directions, joint/ordered updates, backtracking and installed cohort-feasibility extension | Run 6 passes 381/383; run 7 passes 383/385 and reproduces the same two failures. Nineteen fresh CPU model replays pass at those revisions |
| Current comparison correction | Public objective loss-change result, independent numerical bounds and separate training/current-best/patience anchors | CPU verified; actual CUDA lowering, trajectories, ownership and cost remain unrun. Run 8 freezes 385 historical and 529 revised cases |
| Train-many | Shared preparation and independent sequential CPU M=1/8/32 semantics | Compatible GPU batching and full-set speed/memory claims remain unimplemented/unverified |
| Author benefit and adoption | Installed development extensions and a clean D1/D2 author-view export exist | No independent author attempt, complete accounting/isolation runner or repeated external use |

Sources: [scalar evidence](../benchmarks/v1/evidence/cuda-score-symmetry-089/README.md),
[Normal run 6](../benchmarks/v1/evidence/cuda-normal-090/README.md),
[diagnostic run 7](../benchmarks/v1/evidence/cuda-acceptance-091/README.md),
[latest local verification](../learnings/2026-09-07-v1-comparison-hardware-freeze.md),
[author readiness](069-authoring-pilot.md), and [full evaluation protocol](../planning/openboost-v1-evaluation.md).

## Why the recent numerical work belongs in the foundation

Run 7 captured a candidate whose rounded reported NLL improved while its true
loss change at the same stored inputs was positive. The transaction consistently
followed that incorrect decision. A training system also makes related decisions
for best-model selection and patience, against different reference states.

The correction therefore exposes objective comparison and its numerical
resolution as an operation, then gives each consumer its proper anchor. The
current CPU study and all ninety independent trajectory settings pass; CUDA
acceptance is still pending. Original tests and failed archives remain intact.
The revised gate must not erase old failures or silently change their tolerances.

This was a demonstrated correctness counterexample, not speculative abstraction.
The next phase should now bring author and workload evidence closer to the
engineering work. Another large catalog of unmeasured components would leave the
main product hypothesis unresolved.

## Immediate sequence and exit criteria

### 1. Finish the existing Normal CUDA validation cycle

Use the exact pending [run-8 request](092-comparison-run8-request.md) only after
its upload/compute approval. It covers one T4 invocation, two CPUs, 8192 MiB,
900 function seconds, a shared 600-second test deadline and zero retries.

Exit: all 529 revised cases pass with installed-source, complete artifacts,
independent numerical/state/persistence checks and recorded lowering/fit costs.
The separate historical report must preserve its 385 exact cases and the declared
two original coefficient failures plus 24 obsolete byte-accounting assertions.
Those remain historical failures, not revised successes. Missing evidence or any
unexpected result prevents acceptance. A separate known split near-tie remains
explicit even after this comparison gate passes.

Always retain results and stop for retrospective. On failure, classify the first
root cause as numerical comparison, algorithm/state, ownership, fixture/reference
or instrumentation; prepare a focused correction with fresh affected evidence.
No automatic retry or tolerance change. On success, claim only the bounded revised
Normal/D2 contract, then proceed with the priorities below.

### 2. Complete author-pilot readiness and measure actual authoring

Resume [069](069-authoring-pilot.md), already a priority under [085](085-foundation-focus-amendment.md).
Preparation can proceed while a hardware allowance is pending; it need not wait
for all CUDA recipe ports or the full CPU search matrix.

Freeze D1 expectile control and D2 cohort-feasibility deep change, a justified
incumbent implementation path, model/settings/tools, documentation, wheels and
independent verifiers. D2 must later exercise the same verified public device
boundary. Refresh the author view to a pinned current revision: the existing
export predates the current CUDA implementation, and `docs/v1/index.md` still
claims training is CPU-only. Audit author-facing docs against actual call paths.

Preparation exit: a real accounting/isolation smoke visibly enforces both time
and generated-token limits and rejects verifier/test-label tampering; the packet
is reviewable and incomplete measurements cannot pass. Then use an authorized
independent runner for comparative attempts. Measure time to first correct result,
failures, tokens, hints and core/private edits. Designer-written extensions remain
development evidence. The pilot does not pass formal E5; keep its unchanged
multi-task, repeated-attempt thresholds and sealed H1/H2 handling.

### 3. Ask one concrete real-workload cost question

At the run-8 retrospective, freeze a bounded real-data study using an existing
application contract that can exercise the verified scalar/Normal path. Question:
where does end-to-end time go once data is large enough to expose kernel and
orchestration cost, and does D2 composition add material overhead?

Separate uninstrumented timing from profiling. Include preparation, transfers,
compilation policy, objective/geometry, trees, validation/backtracking, export and
CPU prediction; retain task metrics, resource outcomes and failed work. Existing
per-output histogram row scans and recorded synchronization counts give specific
hypotheses to measure, not a performance conclusion. Select any optimization from
the measured bottleneck and require the same correctness/quality afterward.

Exit: an interpretable complete cost breakdown on a frozen real workload and a
specific next engineering decision. This study is a proposed scoped feasibility
step, not authorization to resume the 400-job CPU search or a substitute for E3/E4.
Do not invent calendar or performance estimates from tiny synthetic suite times.

## Remaining required construction and validation

The following work remains mandatory, with exact order revisited after the first
Normal and author-pilot results. It is not all a prerequisite for the first pilot.

| Existing cards | Work and acceptance |
| --- | --- |
| 080 | Complete required CUDA binary/multiclass, Poisson/exposure, scoped AFT and multi-output subsets while preserving squared/Normal. For each: intermediate and task-metric CPU/CUDA parity, failure handling, no silent fallback and fresh CPU inference |
| 081 | Actual compatible GPU batching at M=1/8/32, including independent RNG, stopping, rejected updates, failures and memory limits. A loop over GPU fits is only the reference |
| 070–076 | Finish source/judging/selection readiness and individual real A1–A13 evidence under frozen protocols. Preserve all application families and existing trial budgets; the broader CPU search remains paused pending a specific justified decision |
| 077 | Formal independent author evaluation with fair incumbent paths, correct token/time accounting, immutable judges, repeated attempts and protected held-out tasks |
| 082 | Matched-quality end-to-end GPU, train-many and CPU inference cost gates, including public-composition overhead. Keep the original P7 workload/threshold separate |
| 083–084 | Reconstruct every engineering gate from raw artifacts, verify installed delivery, then independently assess external method creation and second-task reuse after authorized contact |

Optional device cells do not become mandatory merely because their applications
are required on CPU. Ranking, quantiles, Gamma/Tweedie and Formula keep their
required CPU/application evidence and explicitly scoped CUDA status. No required
family is dropped. Ray, multi-GPU, out-of-core and speculative CPU optimization
remain outside current execution.

## Review cadence and decision measures

Commit each independently verified slice. Reflect after the run, every three
implementation commits, phase transitions or a new correctness/consumer
counterexample. Report four separate measures: correct algorithm-change cost;
mathematical/state correctness; real-task quality and complete execution cost;
independent repeat use. A gain in one cannot silently pass another.

The next checkpoint is the run-8 retrospective. After that, prioritize an
independent authoring result and a real-workload cost explanation alongside
required construction. The product question is whether these public boundaries
make useful algorithm changes easier to build correctly and practical to run.
