# Sprint 092: Programmable, reliable loss comparison

Status: local execution approved by the user after
[091's measured retrospective](091-normal-acceptance-diagnostics.md).
092-A's independent mathematics, complete historical mapping and clean-source
evidence are complete. The slice's reflection is recorded below. Public implementation
and consumer changes remain pending. No new hardware allowance is included. All seven
device invocations are consumed. This is the next bounded correctness slice of
079/B12/R6, not a new private trainer or a change to the required R/C/A scope.

## Problem and recommended boundary

[Run 7](../benchmarks/v1/evidence/cuda-acceptance-091/README.md) captures a candidate
whose true Normal NLL worsens by about `5.90e-18`, while its device metric reports
an `8.88e-16` improvement. Ordinary float64 original-row loss subtraction also
has the wrong sign. Float32 reduction cancellation produces the small learner;
the transaction correctly follows an unreliable comparison. More precision in
that one reduction could remove this one leaf without making comparison reliable.

Keep absolute objective values as reporting metrics. Introduce an objective-owned
loss-change operation over two actual stored raw snapshots of the same problem.
Recipes consume its result through an explicit comparison policy. This is a
use-case-driven addition to the foundation: an agent changing objective geometry
can also specify how candidate changes are compared, without editing the grower,
route operations or transaction implementation.

Proposed public result: estimated NLL change, an explicitly justified numerical
uncertainty, and a status distinguishing improvement, worsening, unchanged raw and
unresolved sign. Final names/signatures should follow the CPU oracle slice.
The operation must validate shape/problem identity, use offsets once and preserve
all-row domain checks including zero-weight rows. A CPU reference implementation
and a resident CUDA implementation operate at their declared stored input precision.
The runtime has no Normal-specific branch. Missing comparison support is explicit;
it must not silently substitute the known-unreliable total-loss predicate.

Backtracking accepts a candidate only when the change is demonstrably negative
under the declared numerical policy. Identical stored raw rejects as unchanged;
unresolved sign rejects with a retained reason and continues the bounded trial
schedule. Fixed-step acceptance remains explicit and may accept worsening loss.
Do not manufacture a smaller `proposal.loss`, clamp a small gradient to zero, or
hide a tolerance inside serialization to make the old scalar comparison pass.

## Candidate mathematics to evaluate before production

The [092-A derivation and local experiment](092-normal-comparison-mathematics.md)
establish a conditional arithmetic enclosure without treating platform `exp`
accuracy measurements as guaranteed bounds. They retain the bounded-support and
unresolved-sign cases. This is independent mathematics, not a production correction.

For one Normal row let `r = mean_before + mean_offset - target`,
`p = exp(-2 * (log_scale_before + scale_offset))`, `dm = mean_after - mean_before`
and `dl = log_scale_after - log_scale_before`. The exact mathematical difference is

```text
dl + 0.5 * p * ((2*r*dm + dm*dm) * exp(-2*dl) + r*r * expm1(-2*dl))
```

Weight these row differences once and divide by total weight. The shared Normal
constant cancels; the expression uses actual candidate raw after storage rounding,
not an assumed coefficient times an unrounded tree prediction. This identity is
a candidate evaluation strategy, not yet a verified implementation or error bound.

Derive the resolution rule from the actual arithmetic: residual/offset formation,
products, exp/expm1 accuracy, weights and reduction. Check the CPU and CUDA elementary
function guarantees from primary documentation before relying on a bound. Accurate
or compensated accumulation alone does not bound row-evaluation error. A fixed
epsilon on total NLL is not the policy. If a defensible bound cannot be established
over the intended domain, narrow the verified domain explicitly or return an
unresolved result; do not label empirical precision agreement a universal proof.
No silent host fallback is permitted. An explicitly requested fallback, if ever
introduced, must report transfers and cost separately.

## Construction slices and acceptance

| Slice | Work | Distinguishing acceptance |
| --- | --- | --- |
| 092-A: mathematics and semantics | Independent stable-difference prototype, explicit uncertainty/status rules, original-row high-precision oracle and predeclared cohort mapping. No core edits. | Both measured mean trials classify as worsening; unchanged log-scale trials reject; the analytic `-2^-61` true improvement is retained. Domain/error claims have support, not a fitted threshold. |
| 092-B: public operations | Add the smallest objective-owned comparison record/operation on CPU and CUDA. Keep absolute NLL and model format faithful. | Same stored-input fixtures, offsets/weights once, value/gradient/metric independence, invalid/zero-weight domains, ownership and explicit unsupported operations. CUDA is unverified until a separately approved real run. |
| 092-C: transaction consumers | Connect Normal joint/ordered backtracking, validation-best and patience consumers through explicit comparison policies. Retain each trial's comparison evidence. | Version/term/parent/RNG/rejection invariants; independent best versus accepted versus stopping anchors; finite fixed steps; unknown and invalid trials; installed D2 without a new trainer. |
| 092-D: hardware freeze and retrospective | Freeze current-semantic cases and all historical regressions separately, exact sources, raw traces, CPU replay and cost counters. Request one concrete allowance only when ready. | No hidden case/tolerance changes. Reproduce the two historical failures as historical observations, and verify the new consumer contract independently. Archive all outcomes and reflect before more recipes. |

The first failing test in A reuses the exact stored round-zero candidate from
`normal/acceptance/forward.json`: its mean decreases one float32 value, but its
loss-change sign is positive. Replay both orders. A second test preserves the
tiny analytic true improvement that disappears from full-loss subtraction.

Additional frozen cases must cover clear improvements/worsening, changes that round
back to identical raw, cancellation with unresolved sign, positive weight rescaling,
row permutation, offsets, zero-weight extremes and domain boundaries. Include
joint and individual channels, ordinary/Fisher/damped learners and actual retried
proposals. Derive independent expected statuses before device execution; do not
choose tolerances after observing the GPU result. Use the 60/100-digit oracle as
numerical evidence and higher precision where convergence is insufficient.

## Best-model and stopping ownership

Changing training acceptance alone does not make all loss-based decisions reliable.
Validation-best compares against the retained best model, while patience compares
against its own last qualifying observation and `min_delta`. These anchors can
differ. Do not reuse the current training parent or the best prefix as a shortcut
for both. Preserve one observation per completed outer round and explicit user
stopping policies through the existing completion-status contract.

Before C, choose and document ownership of the needed validation snapshots. A
bounded initial design can retain independent owned best/stopping raw snapshots;
their additional `N_validation * K` storage/copies must be measured and released
on replacement, rejection, exceptions and run closure. An optimization that shares
immutable snapshots needs the same 065/068 lifetime proof. Compare against the
correct retained anchor before publishing a state update. Keep best-score and
reported metrics truthful even when rounding makes their floats equal.

## Historical and revised evaluation cohorts

The original test explicitly evaluates `proposal.loss < state.loss`. Updating a
recipe to call a new operation cannot repair that original caller. Preserve run 6,
run 7, their sources and both failures unchanged. Do not relabel them passing or
count a diagnostic success as conformance.

Before B, publish a case-by-case mapping from all 383 old cases to the new consumer
contract. The [092-A cohort specification](092-comparison-cohorts.md) and
[383-case mapping](092-historical-case-mapping.json) now provide that preregistration;
planned requirements are not collected or passing revised tests.
Retain the old suite as a historical regression cohort, with its status
reported separately. Add a distinct cohort for the new comparison API using the
same original datasets/settings and independent expectations at stored inputs.
Any semantic supersession of the two full-loss predicate cases must be explicit
and reviewed before the next hardware freeze. If other cases change because their
best/stopping comparison was also near the numerical boundary, expose and explain
them before claiming the new gate passes. Keep strict checks for well-separated
decisions and the original metric/leaf/raw tolerance bounds; no blanket widening.

## Scope and reflection

Do not turn this into speculative GPU fusion, wider CPU search or a universal
optimization project. Record the reduction cancellation as a separate follow-up;
its corrected precision policy must be tested on all histogram/split consumers,
including the still-open split near-tie, before broad changes. Stop after each
slice/counterexample and at the next device retrospective. Once Normal comparison
is verified, return to original P7, required CUDA families/train-many and 069's
author accounting. The user's approval authorizes local 092 construction. No
independent author, additional agent, upload or run is authorized here; all
application families remain required.

## 092-A closure and reflection

`8d0e92f` implements the independent arithmetic prototype and distinguishing tests;
`a5967b5` freezes the full historical mapping and 106-case study harness. The
[clean-source evidence](../benchmarks/v1/evidence/normal-comparison-092/README.md)
records 38 improvements, 59 worsening changes, seven unchanged raw pairs and two
unresolved comparisons. All 105 available enclosures contain the original-row
high-precision estimates; the deliberately unsupported exponent case has no bound.
Both recorded false-improvement decisions are rejected, and the analytic tiny
true improvement survives. No core, old test, tolerance or raw run artifact changed.

The main numerical discovery is that neither CPU libm wrappers nor CUDA's
test-derived exponential error table supplies the bound the original proposal
needs. The bounded Taylor/interval construction supplies a mathematical enclosure
under explicit basic-arithmetic assumptions. This is a justified restriction of
the experiment, not a declaration of universal or machine-verified accuracy.
Cancellation also requires higher precision in one diagnostic oracle, recorded
without weakening its target or modifying frozen source.

This supports an objective-owned programmable comparison boundary: the grower
and transaction storage should not learn a Normal-specific rule. The independent
prototype is deliberately not the foundation implementation. Its finite support
and polynomial cost must remain visible when constructing public CPU/CUDA
operations. Do not expand into a general math-library project or change reduction
precision to hide comparison failures.

The next approved local work is **092-B**, followed by **092-C** under the existing
local approval. Before C, make the best/patience ownership decision explicit and
bind planned requirements to concrete verifiers. A comparison record alone does
not repair the three consumers. Additional device activity still requires 092-D's
concrete freeze and a new allowance. Required CUDA families/train-many, original
P7/E4 and 069 independent accounting remain open; no CPU search or new agent work
was resumed by this slice.
