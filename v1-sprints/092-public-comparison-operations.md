# Sprint 092-B: Public objective comparison operations

Status: local construction approved after 092-A. All seven hardware allowances
remain consumed. The independent study and complete historical mapping precede
these production changes. No new upload, execution or author attempt is included.
The public CPU record and Normal operation are implemented and pass 127 focused
tests; the full CPU regression is 1633 passed, one Linux-only skip. Device
construction and its separate verifier cohort follow before 092-B reflection.

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
