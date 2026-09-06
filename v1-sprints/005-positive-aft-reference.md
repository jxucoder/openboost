# Sprint 005: Counts, positive targets and log-normal AFT

Starting revision: `ddf1143`. Status: sprint complete; F0.2 ongoing.
Scope: B01/F0.2, A7–A10 mathematics and minimal data probes.

## Plan and acceptance

1. Write failing tests for Poisson exposure, Gamma/Tweedie derivatives, events/right censoring and stable tails.
2. Implement independent NumPy/stdlib references; verify hand values, finite differences,
   weight replication, two-round tree updates and output units.
3. Add a tiny A9 policy/positive-payment join and two-stage product probe; regression, lint, reflection, commit.

- Exposure e is a Poisson offset but an annualized Tweedie weight; do not apply it twice.
- Gamma requires y>0; Tweedie y>=0 and 1<p<2. Reject overflow/invalid inputs without silent clipping.
- AFT accepts finite positive event times or positive lower/+inf upper right censoring, rejecting
  other intervals. Fixed sigma>0, stable log-tail/Mills and distinct median/mean/survival/quantile units.
- Every objective checks new-raw gradients, leaves and predictions over two rounds. Persistence
  and real external evaluation remain later work.
- Data probes report inconsistent/orphan/nonpositive payments. Missing payments are not automatically zero;
  paid-count must pair with paid-mean.
- No product, real insurance/survival result or E-gate is complete; other cases remain required.

## Results and evaluation

Bounded scope complete. Added positive.py/survival.py, 61 tests and production-import isolation.
**183 passed, no skips**; Ruff passed.

- Poisson hand base=log(7/5); effective all-zero counts require explicit minimum_rate. Doubling e
  doubles count mean only, not rate. Weight remains separate. Gamma base is log(weighted mean).
  Gamma/Tweedie loss/g/h match hand values and finite differences; geometry is unweighted.
- Independent algebra checks root leaves for two rounds of all three positive objectives;
  integer weights match row-replicated losses. Invalid support/power/e/weight fail. Gamma log-ratio
  retains valid geometry at y=1e308, F=log(y). Unrepresentable exponentials fail, with no hidden clipping/curvature floor.
- AFT event/right-censored derivatives pass finite differences. Lower must be finite positive;
  upper equals lower or +inf, otherwise fail. Mixed two-round updates independently match erfc formulas.
- Normal tail: erfc/log1p for z<=8; 300-level continued fraction for z>8, retaining Mills-z correction
  directly for curvature. Compare erfc over z=-10..30 and independent 64-point Gauss-Laguerre
  integration at40/100; switching continuity passes. Never subtract SF to zero before logging or fake curvature by clipping.
- Outputs: separate median/mean/quantile/survival checks; survival decreases with time. Multiplying
  time units by7 adds log(7) to event NLL through the density Jacobian, leaves censored NLL and both
  gradients/Hessians unchanged. Mathematical inverse transforms do not replace serialization checks.
- A9 joins positive payments by policy ID and retains orphan/nonpositive/count-contradiction reasons.
  Only zero-count policies without payments receive zero. paid-count/exposure×paid-mean equals
  annualized amount; raw ClaimNb cannot replace paid-count. Entity ordering is irrelevant.
  This is small-table mathematics, not real ETL or a trained model.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

Local macOS CPU/Python 3.12.12/NumPy 2.3.5/pytest9.0.2; no new dependencies. Missing-module tests
failed first; the first21 mathematical cases passed after implementation. An expanded check tried
to compare scalar loss and array g/h as one ragged array; it was corrected to compare each separately.
Lint naming issues were fixed. No real datasets/baselines, IPCW, CUDA, complete two-stage fit/save or E-gates ran.

## Reflection: Direction after three implementation commits

Sprints003–005 prepare different geometries, but production remains a namespace. All new code is
independent reference with production imports blocked; no performance, real quality or agent-cost
measurements exist. Test growth is not product value. These cases must become F1 component conformance
checks; F2 must measure algorithm-change cost.

Exposure plays different roles in A7/A9, and events/censoring have different likelihoods at the same
time. Doubling exposure, paid-count products and time-Jacobian tests distinguish them. Preserve typed
target/offset/weight/output contracts; no generic trainer should guess semantics.
Next: Normal/NaturalBoost and Formula, then identity/state/run/full categorical/vector growth and
F0.3 freezing. Every A1–A13 still needs its own implementation and real evaluation; insurance/AFT
examples do not receive privileged scope.

## Commits

- This slice: `test: add positive target and AFT references for v1`.
