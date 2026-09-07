# Sprint 078: CPU ownership seam audit

Status: source audit, not a device implementation. Follows the approved 085
sequencing exception. No CUDA run has occurred or consumed its two-run budget.

## Findings from actual call paths

| Boundary | Current behavior | Required device seam |
| --- | --- | --- |
| data.NumericData / data._owned | Rejects device!=cpu; copies to float64 NumPy and immutable bytes | Keep host preparation; explicitly upload an owned numeric execution view |
| binning.BinnedData / binning._array | Feature-major NumPy codes/missing flags, immutable host export | Resident codes/missing arrays with the same fitted Binning identity |
| stats.RowFields | NumPy ownership and column_stack for extra fields | Resident named fields with identical weight roles and row identity; support D2 extra columns |
| ops.histogram | NumPy sums/bincount per feature/field | Device reductions with a separate missing bin and routed original rows |
| ops.candidates / choose | Python Candidate objects and scalar callbacks | Batched candidate fields, batch feasibility/scoring, deterministic compact winner |
| tree.Tree | Validated CPU topology and leaves; NumPy inference | Resident construction/inference buffers plus explicit CPU-readable artifact export |
| runtime.AcceptedState / _evaluate / resolve | CPU prediction caches and _owned copies on proposals/commit | Backend-owned accepted/proposal buffers, same parent identity and atomic resolution |

The current default scalar path cannot become resident by replacing np with cp in
one kernel. _owned and _array explicitly materialize host bytes; CPU Candidate
callbacks would force scalar synchronization or bulk downloads. Existing caches
avoid ensemble replay on CPU but do not establish CUDA buffer ownership.

## Construction decision

Keep existing CPU public behavior as the reference. Add explicit execution-owned
resident representations behind shared operation semantics, beginning at prepared
numeric codes and named fields. Do not admit CuPy into CPU-only constructors and
pretend their immutability or validation survives automatically. Public device
operations must retain identity, field roles, dtype and shape contracts.

The first implementation slice is ExecutionContext ownership and explicit
upload/export, followed by device fields/histograms. A context owns device, stream,
allocation lifetimes and transfer/synchronization accounting. A buffer belongs to
one context; foreign, closed or aliased output buffers must fail. State acceptance
must own its storage: a frozen Python dataclass is not GPU immutability. Public
extension work buffers must not alias accepted raw values. Device copies are
allowed to enforce ownership; record their cost. Do not expose mutable authoritative
accepted buffers to callbacks as though they were immutable CPU arrays.

Separate transient resident tree construction from CPU inference serialization.
Export small topology/leaves explicitly at the artifact boundary; keep repeated
raw updates and row routing on-device. Do not re-run the host ensemble for each
GPU proposal. Proposal rejection must preserve accepted raw/model/best/stop and
must not promote mutable workspace into an accepted snapshot.

D2 needs batched independent information fields and a device feasibility operation.
Its independent columns are never multiplied by objective weights. Preserve custom
component semantics through explicit device capabilities; reject CPU-only callbacks
before upload. A device implementation must not silently ignore a supplied scorer,
leaf solver, feasibility function or acceptance policy.

## First implementation acceptance

1. Preserve existing CPU constructor/ownership tests. A strict CUDA request remains
   explicitly unsupported until its complete capability path exists.
2. Test upload ownership, wrong-context/use-after-close rejection, stream ordering,
   explicit export and immutable accepted/proposal storage on actual CUDA hardware.
   No NumPy fake or skipped job counts as device conformance.
3. Match weighted fields and routed histograms, including missing and independent
   cohort fields, to existing independent CPU references before tree construction.
4. Add batch candidate selection and raw updates; then satisfy 085's two-round
   scalar criteria. Freeze E1 tolerances/fixtures before the first remote run.

The exact public execution API and dependency pins are still to be implemented
and verified. This audit is a prerequisite decision, not evidence that an execution
context, device storage or kernels already exist. The first T4 allocation remains
bounded by 085 and should wait until the local implementation is reviewable.
