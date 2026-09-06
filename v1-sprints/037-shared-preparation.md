# Sprint 037: Shared CPU training preparation

Parent: 752a2ab. Status: complete for the bounded slice.

## Plan and acceptance

1. Add owned preparation binding data identity, bin capacity and fitted codes.
2. Let every built-in recipe and RunSpec accept explicit prepared training data;
   reject mismatched data/config and preserve independent target/weight/run state.
3. Verify no repeated bin fitting, M=1/8/32 heterogeneous independent/reordered/
   regrouped equivalence and failure isolation.
4. Run regression/lint/docs/build, reflect and commit locally.

Preparation covers training binning/codes, not inference caching or fused
execution. Validation-driven independent stopping remains the next slice.
First failing check imports missing PreparedData.

## Results and reflection

PreparedData binds immutable training features, bin capacity, fitted transformer
and codes. All twelve built-in recipes resolve this explicit input without a
refit; RunSpec carries it separately from scalar options. A mismatch fails the
affected run rather than rebuilding or contaminating another run.

Five focused cases pass, including M=1/8/32 heterogeneous squared/Normal/Poisson/
multi-output independent, reversed and regrouped equivalence. Tests prohibit
Binning.fit after preparation, check distinct raw caches and retain failed runs.
CPU regression: 735 passed. Ruff, strict MkDocs, offline sdist/wheel and all
nineteen installed-wheel examples pass (macOS/Python 3.12.12/NumPy 2.3.5).
Local wheel SHA256:
61fa4f6f4c16b02711e1541aee8b1aa5df6445cdf94341da2a59ee81e569ba7a.

Observation: shared feature preparation and independent model state can be
separated without a global cache. Evidence: supplying one prepared object gives
the same per-run state as independently fitted preparation under reordered and
regrouped execution. Decision: reuse by explicit immutable identity/config
contract; never infer equivalence from shape or object address.

This does not cache validation/inference transformations, establish measured
speedup or implement fusion. M32 equivalence here uses independent fixed round
budgets, not validation-triggered stopping. Next add independent stopping and
verify different stop rounds plus failure/reordering under shared preparation.
Other Sprint 035 requirements and F0.3/F1–F5 remain open.

Status: complete for the bounded slice. See
[learning record](../learnings/2026-09-06-v1-shared-preparation.md).
