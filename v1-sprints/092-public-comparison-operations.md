# Sprint 092-B: Public objective comparison operations

Status: local construction approved after 092-A. All seven hardware allowances
remain consumed. The independent study and complete historical mapping precede
these production changes. No new upload, execution or author attempt is included.
The public CPU record and Normal operation are implemented and pass 127 focused
tests; the full CPU regression at that slice is 1633 passed, one Linux-only skip.
The device operation and objective callback are now constructed. Its separate
117-case GPU verifier cohort collects locally; device execution remains pending.

## Implementation order and acceptance

1. Add immutable `LossChange` with validated bounds, method/reason and a derived
   status, estimate, uncertainty and strict `improves(min_delta)` query. It has no
   objective or runtime branch. Unavailable bounds and unchanged raw are explicit.
2. Implement `Normal.compare(problem, before, after)` on CPU. Validate both raw
   snapshots and every objective-domain row, including zero weights. Compare exact
   stored values with offsets and weights once; leave `Normal.loss` unchanged.
   Use the 106-case study and the structurally different original-row Decimal
   expressions, not production metrics or gradient values, as expectations.
3. Implement the resident device operation and an optional objective callback.
   Keep its row algebra shared with CPU, but supply explicitly directed binary64
   CUDA arithmetic instead of a platform-libm error assumption. Original f32
   domain checks remain prerequisites. Export only a bounded scalar summary and
   account for all scratch and transfers. Never fall back to CPU computation.
4. Add a separate GPU operation cohort for stored-input bounds, domains, ownership
   and errors, collected locally and explicitly unverified until a new real run.
   Commit each verified local slice and reflect before consumer construction.

The first distinguishing test calls the public CPU operation on the captured
rate-four mean candidate. It must report worsening although both independently
rounded losses report improvement. The analytic `-2^-61` improvement must survive.
Nonfinite/shape/row-domain errors must not become unchanged or unresolved; finite
inputs beyond the bounded comparison support report an explicit unresolved reason.

## Numerical boundary

The [092-A derivation](092-normal-comparison-mathematics.md) remains the mathematical
contract. Production uses a private scalar interval algebra factory shared across
backends. Python supplies outward `nextafter` expansion around separately rounded
binary64 basic operations. CUDA supplies directed double addition/multiplication/
division; those intrinsics prohibit contraction across the specified operations.
The common implementation handles finite intervals, the 18-term reduced Taylor
polynomial, its bounded remainder, and the weighted Normal difference. It does not
import the independent test implementation. The Decimal original-row oracle remains
structurally different. Compiled CUDA lowering, parity and cost are still unverified.

`LossChange` describes an objective's declared bound; it does not establish that
an arbitrary external callback supplied a correct bound. Snapshot/problem/parent
identity belongs to the public operation and its consuming transaction. CPU arrays
follow the existing explicit row-order contract. Device operations reject foreign,
released or forged prepared records/buffers before arithmetic.

## Consumer boundary

092-B does not change backtracking, best-model selection or stopping decisions.
092-C must retain separate validation anchors and explicit comparison policies,
with ownership specified before edits. A missing objective comparison callback
must raise when comparison is requested. Existing scalar contracts do not silently
inherit a Normal policy. All 383 old cases and failed archives remain identifiable;
the new operation cohort cannot count as full boosting conformance.

## Construction reflection

The public CPU operation passes the full independent numerical study, and the
device path now expresses the same bounded algebra using directed double
intrinsics. Its two kernels produce row intervals and an ordered weighted
reduction, with a separate all-row float32-domain check. Eleven additional GPU
cases cover domain/identity/rollback and callback independence beyond the 106
stored-input cases. All 117 are collected, not run.

The device API's planned cost is `32*N + 32` bytes of scratch, a 32-byte summary
export plus existing validation flags, and four launches including validation.
The counters expose those costs, but no actual device measurement or lowering
verification is available. This may be expensive; the next conformance run must
retain the cost, compilation and real end-to-end evidence. Do not describe shared
source as CPU/CUDA parity.

Local verification: 1636 CPU tests pass with one Linux-only skip; production and
changed support lint pass. Documentation builds with the existing external-tree
run-6 evidence link warning. Offline source/wheel build succeeds, and an isolated
wheel import with CUDA modules blocked exercises the public tiny-improvement
comparison. Historical source snapshots, verifiers, tolerances and raw run
artifacts remain unchanged.

092-B construction is complete; device verification remains part of the future
hardware gate. Continue with 092-C ownership/design and explicit consumers under
the existing local approval. The public operation alone has not fixed training,
validation-best or stopping decisions.
