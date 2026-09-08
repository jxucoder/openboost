# Sprint 092-C: Separate comparison consumers and their anchors

Status: CPU consumers implemented and locally verified; resident consumers and
revised hardware-cohort bindings remain open. This design recorded the ownership
decision before runtime/recipe edits. Actual CUDA verification still requires the
later 092-D freeze and a new allowance.

## Explicit policies

Training backtracking compares the actual candidate raw against its bound accepted
parent. Accept only `LossChange.improves()`. Retain the comparison on each finite
trial, including unchanged or unresolved decisions, and continue the same bounded
halving schedule after rejection. Numeric domain errors retain their existing
failure record; structural or infrastructure errors propagate. Fixed acceptance
remains an explicit choice and can commit a finite worsening candidate. The
objective comparison does not replace truthful loss/validation reporting.

Best-model selection compares candidate validation raw against the best validation
anchor, with zero minimum change. Patience compares accepted validation raw against
its own last qualifying observation, using its configured `min_delta`, once per
completed outer round. Every consumer obtains the objective's comparison directly;
no consumer infers improvement from equality/ordering of rounded reported scores.

Keep the existing reported-metric policy identifiable for historical low-level
callers and scalar regressions. A device run should select a named comparison
policy explicitly; the Normal recipe selects objective comparison. Requesting
objective comparison with no callback must fail before preparation/upload. There
is no fallback from that request to reported metrics. The runtime must not inspect
the objective's name or contain a Normal-specific branch.

## Ownership decision

### CPU

The immutable `best_model` already identifies the exact best validation predictor.
At an explicit objective-comparison transition, replay that model on the bound
validation data to obtain the comparison anchor. This temporary `N_validation*K`
array belongs to the call; successful/failed calls retain no borrowed mutable
view. Current accepted/proposal raw stays immutable. This prioritizes an auditable
CPU reference over a new best-cache optimization. Record the extra replay cost as
a limitation; it is not a CPU optimization project.

Patience retains its own immutable owned validation raw snapshot, initially the
base prediction, and replaces it only on a qualifying change. Add an explicit
`StopState.observe_change(score, change)` operation; retain `observe(score)` for
authors intentionally supplying a score-based policy. A qualifying improvement
can leave the reported score bit-for-bit equal, so the recipe must use the change
decision to replace the anchor, never `new_reference_score != old_reference_score`.
Expose trial comparisons in full and summary traces without retaining extra
per-trial arrays. Public stopping completion metadata remains unchanged.

### Resident CUDA

In objective-comparison mode, each accepted state owns an independent best
validation raw snapshot, in addition to its current training/validation raw.
Initialization copies the base validation prediction. An accepted transition
compares the candidate against the previous state's best snapshot, then copies
either the candidate validation raw or the old best raw into the new state's
owned best snapshot. Even unchanged best anchors are copied in this initial
implementation. Rejected proposals return the identical state and allocate no
new best snapshot. Reported-metric mode does not allocate this new storage.

All device computation and fallible copies must finish before publishing a state,
incrementing term references or advancing state identity. The same workspace and
atomic cleanup discipline applies to failed comparisons, failed copies, stale
parents, record release and run closure. Releasing a parent or rejected proposal
must not invalidate a surviving state's best snapshot. Prefix-based model export
remains faithful to the separately selected best terms and reporting score.

The Normal recipe separately owns one patience validation snapshot. Compare it to
an owned copy of the newly accepted validation raw after the outer sweep. Replace
the anchor only on a qualifying change; otherwise release the new copy. Release
the retained anchor on recipe completion and all exception paths. Provide a
public way to access the prepared validation problem; avoid reaching into runtime
private dictionaries from algorithm code. No host raw export is part of comparison.

The planned additional retained storage per live objective-mode state is
`4*N_validation*K` bytes for the best snapshot; the active recipe retains another
`4*N_validation*K` for patience. A temporary current-validation copy adds that
amount during observation, on top of the comparison operation's own scratch.
Historical byte-count assertions belong to their historical contract. New tests
must account for these copies and verify release; sharing is deferred until it
has its own 065/068 proof.

## Construction order and distinguishing tests

1. Generic CPU resolve/stop operations and Normal backtracking. Start with a
   constant `2^-30 -> 0` mean change whose reported NLLs are equal. Training must
   accept it and validation-best/patience must advance using the stored-input
   evidence. Keep the reporting floats truthful. Reject both captured worsening
   run-7 candidates through the new public consumer.
2. Resident parent-bound proposal comparison and owned best snapshots. Exercise
   accepted/rejected/fixed/invalid/unresolved transitions, original parent and RNG
   invariants, retained earlier states and allocation/dispatch failures. A
   candidate better than the current validation state but worse than best must
   leave the best prefix and anchor unchanged.
3. Resident joint/forward/reverse recipes and patience. Use validation means
   `2 -> 3 -> 1.95 -> 1.97 -> 1.9 -> 1.89` against target zero, fixed acceptance,
   log-scale zero and `min_delta=0.2`: current/best/patience anchors must differ
   when appropriate. Improvements below the patience threshold can still update
   best. A qualifying change advances the patience anchor even if reporting
   floats are equal. Preserve one observation per completed outer sweep.
4. Bind the [383 historical requirements](092-historical-case-mapping.json) to
   collected revised consumer tests as those tests are authored. Keep settings,
   original mathematical/metric tolerances and all original outcomes visible.
   Reuse the independent original-row oracles and captured actual stored inputs;
   do not use CPU emulation as CUDA evidence. Include installed D2 and independent
   saved CPU inference. Explain any revised trajectory before the hardware freeze.

Each implementation slice starts with its distinguishing failing test, records
verification/limitations and is committed independently. 092-C cannot close with
only a new comparison record or training acceptance: all three consumers and their
ownership contracts must be represented. 092-D then freezes sources, both test
cohorts, retained raw observations, lowering checks and end-to-end cost evidence.

## CPU consumer slice

Ten new consumer cases failed before editing: missing public comparison arguments,
missing stopping operation, and actual equal-score recipe rejection/early stopping.
They now exercise all three anchors. The two stored run-7 false improvements are
rejected by the CPU trial consumer; tiny genuine improvements advance acceptance,
best and patience while reported NLLs stay equal. The prescribed five-transition
anchor sequence preserves distinct current/best/last-qualifying observations.
Fixed/unresolved policy, stale parents, structural/dispatch failures and RNG
preservation are covered. Full and summary recipe traces retain comparison evidence.

The generic CPU runtime accepts an explicit comparison callback; other recipes
keep their score policy. StopState adds observe_change without changing completion
metadata. The private shared trial helper now returns comparison records alongside
the existing four results. TraceSummary explicitly accepts the bounded LossChange
scalar record; it still rejects arbitrary objects, states and arrays.

Focused verification: 100 tests pass across the new consumers, Normal, stopping,
incremental runtime and retention tests. Full regression results are recorded in
the linked learning entry. No frozen oracle, case, archive or tolerance changed.
This completes construction step 1 only; device consumers and conformance remain
open. The extra CPU best replay is deliberate and has no speed claim.
