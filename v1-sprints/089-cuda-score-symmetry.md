# Sprint 089: Preserve scalar score symmetry

Status: local correction and validation-package preparation after the user's
continuation of the run-4 retrospective. Baseline `f5922fb`. No new hardware or
private upload allowance, agents, CPU search, objective expansion or phase exit.

## Purpose and counterexample

This is a correctness correction within 078-C/B12/F3.1 and E1, following
[088](088-resident-scalar-training.md). Resident squared training currently fails
14 of 202 real T4 cases. In the weighted/missing fixture, mathematically tied
root candidates select different splits and two recipe predictions violate E1.
The [run-4 evidence](../benchmarks/v1/evidence/cuda-resident-078/README.md) is immutable.

The smallest failing example already exists on real hardware: depth-one weighted
root topology differs from the independent original-row oracle. A CPU arithmetic
counterexample shows that asymmetric multiply/add contraction can break equality
of swapped child summaries by one binary32 ULP. This is a hypothesis about run 4's
cause, not a measured instruction trace. No CPU simulator can accept the CUDA fix.

## Construction sequence

1. Add real-device direct score checks: swapped child summaries, reordered named
   gradient/curvature columns, finite regularization/penalty and strictly unequal
   adjacent-ULP choices. Capture the actual weighted root gradients, histograms,
   candidate summaries and score bits before assertions. A test-only copy of the
   exact run-4 scoring kernel provides a comparison on identical resident inputs;
   it is historical production code, not an independent correctness oracle.
2. Independently round child and parent score products in the existing scalar
   kernel. Keep the public boundary, exact lexicographic tie rule, strict score
   comparison, fixture semantics and all E1 tolerances unchanged. Verify local
   math, source integrity, CPU imports/regression, lint and documentation; commit
   the bounded correction without claiming CUDA acceptance.
3. Freeze a new installed-package run containing all 202 original cases plus the
   new diagnostics. Capture corrected and archived-kernel PTX in the original
   pytest output, with source hashes and score values. Validate exact source/case
   closure, immutable prior evidence, dispatch guards and packaging before asking
   for one new concrete private-upload and compute allowance. Commit the freeze.

## Numerical choice and acceptance

Use `numba.cuda.libdevice.fmul_rn` for each final score product, preserving the
existing factorization while preventing contraction of one child's multiplication
into the other child's addition. NVIDIA specifies independent nearest-even
multiplication that cannot merge with an addition. See
[libdevice](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rn.html)
and [numba-cuda inspection](https://nvidia.github.io/numba-cuda/reference/kernel.html).
Ordinary finite float32 addition of the separately rounded child scores is
commutative; no epsilon tie band or global fast-math setting is introduced.

The hardware gate requires every original case and added check to pass, no
missing/duplicate/skipped cases, exact swapped-score equality, exact tie winners,
strict ordering for unequal adjacent-ULP scores, and unchanged leaf/prediction/
metric parity. The root diagnostic must preserve both candidate summaries and
generated code even if a later assertion fails. Reproducing the old kernel's
failure is diagnostic, not required to accept the corrected behavior: if it does
not reproduce, record that limitation and investigate without rewriting run 4.

The candidate fix has no real-device acceptance until that run completes. Preserve
the 065/068 ownership checks, all R/C/A/E requirements and independent author
isolation. Reflect at the source correction, completed package and hardware result.
Any future invocation has a fresh fixed output and zero retries; the four consumed
allowances cannot be reused.

## Diagnostic construction

Ten new CUDA cases collect locally: one weighted-root comparison with both kernels
and PTX capture, six swapped-summary combinations (three regularization/penalty
settings and two field orders), and three adjacent-ULP choice scales. The root
case uses public preparation/objective/histogram/scoring operations. Direct kernel
access is confined to diagnostics on context-owned buffers and the same stream.
There is no new public kernel registry or fallback. The historical function's
exact source segment matches its run-4 hash. Twenty-one focused CPU source,
arithmetic and import/configuration checks pass; changed-file Ruff passes.
Collection is not real-device verification. Production arithmetic is unchanged
in this first diagnostic slice.

## Source correction and reflection

`13fdd83` commits the diagnostics before the kernel change. The production scalar
scorer now uses explicit `fmul_rn` for its left, right and parent products. Their
factorization, parameters and validity checks are unchanged. Choice still uses
strict `>`; there is no change to public operations, candidate ordering or tolerance.

Full CPU regression passes 1265 cases with one Linux-only skip and 212 GPU cases
deselected. Production/changed-support Ruff and documentation build pass. CPU API
imports still work with CUDA packages blocked. These checks protect CPU semantics
and import boundaries; they do not prove the new kernel compiles or passes on T4.

Reflection: a local numerical correction stays inside the shared foundation
operation. Direct historical/current score diagnostics will test the cause without
replacing the independent original-row oracle. The next deliverable is a frozen
49-file, 212-case package with pending compute/upload authorization. No broader
CUDA capability or quality claim follows from this source change.
