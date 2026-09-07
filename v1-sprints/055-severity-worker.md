# Sprint 055: Current A8 claim severity worker

Parent: 606487c. Status: complete; all five bounded real-data folds pass.

## Plan and acceptance

1. Reproduce missing A8 support with weighted direct Gamma recipe parity.
2. Bind positive claim targets and original sample weights, preserve scalar
   log-mean models and explicit positive severity output. Reject nonpositive
   targets and unsupported exposure/offset inputs.
3. Verify final/best selection and fresh replay, then run all five frozen
   policy-grouped claim folds at unchanged 90/30-second caps. Independently
   recompute selected Gamma objective and retain raw outcomes/failures.
4. Run regression/lint/docs, reflect across Sprints 053–055 and commit locally.

No policy-average substitution, dispersion fitting, quality/search or CUDA claim.
All required remaining applications, real searches and D5 checks remain open.

## Results and reflection

Both new A8 direct cases initially failed on unsupported application. The worker
now composes the existing Gamma recipe and positive_mean transform. Weighted
final/best selection, fresh inference and nonpositive/foreign-input rejection
pass. All five frozen policy-grouped claim folds pass bounded execution and exact
replay. Independent selected Gamma objective recomputation agrees at the declared
rtol=1e-12/atol=1e-14; source/output hashes match the
[raw evidence](../benchmarks/v1/evidence/severity-055/README.md).

Across Sprints 053–055, multi-quantile, period-count/exposure and positive-claim
workflows required evaluation adapters but no foundation production changes.
This supports keeping the present composition boundary for these consumers;
it does not establish that external authors need less work. Explicit row units,
output transforms and stopping semantics remain necessary at the adapter layer.
Do not replace outstanding D5 or quality searches with more objective counts.

Next connect A9 annualized aggregate targets and exposure weights, retaining the
separate frequency-severity composition requirement; then A10/A12 and unresolved
A4, real searches and D5. CUDA implementation and formal acceptance stay open.
No required use case is removed. See
[learning](../learnings/2026-09-06-v1-severity-worker.md).

Closure: 872 CPU tests passed; Ruff, strict MkDocs and whitespace checks passed.
No foundation production changes. No push or publication.
