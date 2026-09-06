# Sprint 029: B09 routed quantile and penalized leaves

Parent: 55f3b91. Status: complete for the bounded slice.

## Plan and acceptance

1. Add owned residual context and routed leaf views retaining global row IDs,
   original weights and current residuals; bind to the split problem identity.
2. Implement weighted quantile and anchored quadratic-penalty pinball leaf solvers,
   plus a CPU quantile recipe reusing existing split policies and transactions.
3. Verify against independent quantile/D3 oracles, including second-round effects,
   all growth policies, weights, offsets, identity and persistence.
4. Run regression/lint/docs/build checks, reflect and commit locally.

No real A5 quality, author-effort, GPU or performance claim. First failing check:
the public residual leaf context and quantile recipe are missing.

## Results and reflection

Delivered ResidualContext/ResidualView and paired row_leaf/leaf_context inputs on
all three growers. Views retain original global row IDs and weights. Quantile
splits use pinball gradients and unit pseudo-curvature, while leaf values solve
weighted residual quantiles or anchored quadratic-penalty pinball. Initialization,
offsets, fixed/backtracking acceptance and best-model selection compose existing
state operations.

Thirteen focused tests pass: 80 randomized ordinary/penalized solver comparisons,
three rounds under all three policies with and without penalties, original-weight
and row-ID routing, immutability, foreign context rejection, penalty/mass response,
atomic backtracking rejection and mixed-feature inference round trips.
CPU regression: 655 passed. Ruff, strict MkDocs, offline sdist/wheel and all twelve
installed-wheel documentation examples pass on macOS/Python 3.12.12/NumPy 2.3.5.
Local wheel SHA256:
bc2b146fbf2c86b14ea607ad4113779fabacd11ed68655f8825d8399dbb8b351.

Observation: additive statistics alone cannot recover weighted residual quantiles.
Evidence: independent three-round trees require routed residual solutions, and
changing original weight mass changes the penalized optimum. Decision: keep the
residual context distinct from weighted additive fields and enforce problem
identity at the grower boundary. This context is a scalar residual implementation,
not yet a universal feature/linear-leaf context. Generalize only against the next
concrete use case.

Split reg_lambda and leaf penalty are distinct; validation uses prediction pinball,
not a model-wide anchored penalty. No agent-effort or real A5 result follows from
implementing the D3 solver. B09 pair weights/sampling remain deferred; full A6
workflows and F0.3/F1–F5 gates remain open. Next construction slice is B10 positive
target objectives, starting with Poisson/exposure, then Gamma/Tweedie and AFT.

Status: complete for this bounded slice. See the
[learning record](../learnings/2026-09-06-v1-b09-quantile-leaves.md).
