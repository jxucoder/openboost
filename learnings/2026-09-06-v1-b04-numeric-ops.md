# 2026-09-06: Public numeric preparation and split operations

## Context

B03 established immutable CPU problem/run state. B04 now needs shared operations
that external algorithm code can compose, rather than a closed trainer.

## Decision or Result

Fit unweighted quantile cuts once, retain explicit missing masks and transformer
identity, and aggregate named additive row fields from actual selected rows.
Newton weighting happens once in a declared adapter; independent cohort information
is separate. Histogram prefix/suffix candidate sums are checked against the
structurally different exhaustive original-row reference.

## Changes

- Public binning.py: immutable numeric cuts, feature-major int32 codes and missingness.
- Public stats.py: named row fields, weight roles, once-weighted Newton adapter.
- Public ops.py: histogram, candidates, feasibility/scoring callbacks, deterministic
  choice, identity-checked partition and scalar Newton leaf.
- Public documentation executes a constrained split using only public components.

## Verification

- Full suite: 527 passed, including 18 new numeric-operation cases.
- Quantiles/codes match the independent order-statistic oracle for minimum cuts,
  duplicates, constants, all-missing values, one bin and unseen numeric ranges.
- Feasible conditions, gains, chosen split, left/right rows and leaves match
  exhaustive row enumeration on weighted/missing/subset/empty fixtures.
- A public cohort constraint changes the winner; independent mass stays unweighted.
  Two-level manual composition conserves every original row and matches leaves.
- Double weighting, foreign rows/binning, invalid routes and custom NaN scores fail.
- Overflowing quantile interpolation fails explicitly instead of losing cuts.
- Ruff, strict MkDocs and offline sdist/wheel build passed. Both B03 and B04 public
  examples ran using an isolated installed wheel under Python -I outside the repo.
  Wheel SHA256: 707cfcaee9187764bbe44da08c86745bc5566307cc79ef4fd8da80be0f23b48d.

## Failed Attempts

The initial cohort-winner fixture used four quantile bins for six rows, merging
the distinguishing first two rows. Six bins restored the intended independent
counterexample; no algorithm or acceptance threshold changed. A lint check also
required combining nested test contexts.

## Risks and Follow-ups

This is the operations slice, not a complete B04 tree grower. No tree artifact,
boosting recipe, categorical splitter or CUDA implementation exists yet. RowFields
metadata prevents API-level reweighting, not arbitrary plugin mathematical errors.
Histograms use dense CPU NumPy storage; no performance or memory-efficiency claim.
Extreme quantile interpolation may require explicit feature rescaling.

Next: public depthwise assembly and validated tree prediction/persistence using
these same operations, then B05 squared/Normal recipes. B06 must probe Formula
and heterogeneous state before stabilization; F0.3 remains open.

## Commits

- This slice: feat: add composable CPU numeric split operations.
