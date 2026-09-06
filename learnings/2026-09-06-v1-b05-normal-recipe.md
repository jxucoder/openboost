# 2026-09-06: Normal geometry shares scalar tree and transaction foundations

## Context

Squared boosting worked, but Problem/state coupled observed target width to raw
parameter width. Normal's scalar observation and two raw parameters exposed the
first necessary change to that initial state boundary.

## Decision or Result

Problem now declares raw_width independently, defaulting to observed target width.
Offsets align with raw parameters; state validates against raw_width. Normal uses
ordinary gradients or diagonal Fisher directions, then fits unweighted directions
through once-weighted least-squares statistics. Both scalar learners commit or
reject jointly through the same mapped terms and runtime as squared boosting.
Shared numerical trial handling retries smaller coefficients without swallowing
structural errors. No separate distributional trainer or duplicated targets were
introduced. This validates one additional geometry, not the full v1 abstraction.

## Changes

- data/runtime: independent raw width, strict offset shapes and model validation.
- objectives: Normal weighted NLL, unweighted gradient/Fisher diagonal, offset-aware
  base and mean/scale output, plus ordinary/diagonal-natural direction operation.
- stats: public least_squares adapter with G=-w*z and H=w.
- recipes: joint Normal updates and shared configuration/trial helpers. Numerical
  backtracking failures are recorded; fixed-step numerical failures still raise.
- Public Normal example and updated construction/capability documentation.

## Verification

- Initial raw-width test failed before implementation with unsupported keyword.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run
  --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short -q`:
  572 passed, including 12 new Normal/raw-width cases. Python 3.12.12,
  NumPy 2.3.5, macOS. No CUDA validation.
- Ordinary and natural modes agree with independent three-round exhaustive
  reference gradients, Fisher diagonals, directions, coefficients, losses and raw
  predictions. Final NLL agrees with the separate density-based reference scorer.
- Offset-aware base satisfies weighted first-order conditions; geometry agrees
  with explicit shifted-raw reference. Regression curvature equals original
  weight, not Fisher curvature. Offsets remain outside accepted caches.
- Joint rejection preserves identical raw/model/best/version; two learners fit
  once per round. Nonfinite trials continue to later finite evaluations without
  committing invalid state. Wrong learner schemas raise before trial handling.
- Fresh-process persisted raw ensemble reproduces mean/scale outputs on unseen
  numeric and missing observations. Invalid shapes/modes/damping/floors fail.
- Ruff, strict MkDocs and offline build pass. Five public examples pass under
  Python -I from an isolated installed wheel outside the checkout.
  Wheel SHA256: b6181e78dbde0856109b74b9d5a6b1b92194810fe31f84266bf15882d5a6549d.

## Failed Attempts

An initial file lookup assumed a distribution-prefixed reference filename; the
reference is coupled.py. Import ordering required routine lint fixes. Review of
numerical trial handling prompted validating schemas before the retry catch, so
structural errors cannot appear as ordinary full rejection. No oracle changed.

## Risks and Follow-ups

Normal currently uses joint updates and diagonal Fisher geometry. Ordered updates,
Formula full/GGN metric, heterogeneous runs and every other required use case
remain necessary. B06 is the next construction probe before stabilization.
Raw model artifacts do not store a distribution tag, interval calibration or
resume state; callers explicitly apply Normal output conversion. Scale flooring
is initial-only; later nonrepresentable distributions are rejected, not clipped.
Predictions are recomputed during trials and trace arrays are retained; there is
no large-workload performance, quality parity, GPU or adoption claim. F0.3 and
full evaluation gates remain open.

## Commits

- This slice: feat: add Normal boosting with independent raw parameter width.
