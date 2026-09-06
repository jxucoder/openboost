# Sprint 027: B08 vector leaves and multiclass

Starting revision: 6e5d6b0. Status: complete for the bounded slice.

## Plan and acceptance

1. Generalize shared tree payloads and output maps to L-dimensional vectors; expose
   vector Newton statistics/scoring/leaves and separate split versus leaf fields.
2. Add softmax geometry with the named diagonal upper bound and joint vector-tree
   multiclass updates, preserving class order in inference artifacts.
3. Compare all growth policies and projected splits/full leaves against independent
   mixed/vector references; verify multi-round softmax, rejection and persistence.
4. Run regressions/lint/docs/build, reflect and commit locally. No CUDA, full A6
   workflow or external-library parity claim is intended.

First failing check imports vector_newton for a two-output shared tree.

## Results

Delivered vector Newton fields, summed channel scoring and diagonal leaf solves,
all three growers with separate split/leaf fields, [L,K] mappings, joint multiclass
rounds and softmax inference. Tree-v3 stores matrix payloads; older versions fail.

- Nine focused tests pass against independent mixed/vector and softmax references.
  They cover projected/full statistics, topology and predictions across all three
  policies, three rounds, arbitrary output mapping, offsets, rejection and
  fresh-process persistence on numeric/category/missing/unseen inputs.
- CPU regression: 630 passed. Ruff, strict MkDocs and offline sdist/wheel build pass.
- All ten current documentation Python examples pass from an isolated installed
  wheel using Python 3.12.12 / NumPy 2.3.5 on macOS.
- Wheel SHA256: 19ea788e499b34ef3795ee970b3614fd28181b228a296f4bc6423190d5a2cc54.
  This identifies a local validation build, not a published release.

## Reflection

Observation: split width and leaf width can differ without changing routing,
growth policies or transaction semantics. Evidence: all three projected fixtures
produce a different tree from full split statistics while matching independently
computed full-dimensional leaves. Decision: keep leaf_fields an explicit public
input, and keep projection selection in algorithm code. Multiclass adds no new
tree trainer or state engine.

The softmax diagonal is an upper bound, not exact curvature; this distinction is
visible in names, tests and documentation. These fixtures validate mechanics, not
real A3/A6 quality or a general sketching speed advantage. Full A6 workflows remain
open. Next follow B09 with ranking/query isolation and routed quantile/penalized
leaf contracts, then complete remaining application/objective slices and CUDA
under the existing plan. F0.3 and F1–F5 are still incomplete.

Status: complete for this bounded slice. See
[learning and verification](../learnings/2026-09-06-v1-b08-vector-multiclass.md).
