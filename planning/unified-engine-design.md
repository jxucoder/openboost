# Unified GPU Boosting Engine — Design

Status: Phase A+B implemented; A100 speed gate passed (1229x vs NGBoost at 90K) · 2026-08-17
Evidence: `development/paramboost/spike_ggn.py` (GGN spike), CustomDistribution/loop
survey, NaturalBoost GPU-boundary survey.

## 1. Goal

One training core in which every model is a configuration, not a fork:

```
z --trees--> F (n,K raw scores) --link--> theta --formula f--> eta --loss L--> scalar
```

Per-round update rule (the only one in the engine), generalized Gauss-Newton:

```
d_i = (J_i^T M_i J_i + damp*I)^{-1} g_i        # K x K per sample, K small
```

| Model | f | L | M | K |
|---|---|---|---|---|
| GradientBoosting | identity | mse/logloss/... | scalar hessian | 1 |
| NaturalBoost | identity | NLL | analytic Fisher | n_params |
| CustomDistribution | identity | user NLL (JAX/FD) | GGN (upgrade from empirical diag) | n_params |
| FormulaBoost (new) | user formula f(x;theta) | mse or NLL | GGN, modes plain/diag/full | n_formula_params |
| Weibull AFT (new) | identity | censored NLL | Fisher/GGN | 2 |

Spike evidence (sales formula `a*x^sigmoid(bx)`, 40K samples):
plain gradient unusable (RMSE 1.05, degrades); diag GGN 0.134; full GGN 0.145
with much better parameter recovery (b corr 0.84 vs 0.70); extrapolation 19x
better than black-box GBDT (0.20 vs 3.91). Preconditioning is load-bearing.

## 2. Current state (survey summary)

- Two disjoint training loops: `_boosting.py` (1 channel, full GPU story:
  `_fit_gpu`, device-resident buffers, `fit_tree_gpu_native`, growth/GOSS) and
  `_distributional.py` (K channels, CPU-only gradients AND host `fit_tree`,
  richer eval: multi-set, incremental raw scores, `evals_result_`).
- `CustomDistribution`: user NLL + links, JAX autodiff w/ FD fallback,
  empirical **diagonal** Fisher only, no formula layer, quantile/sample assume
  Normal.
- No survival/censoring anywhere. CRPS is eval-only. No line search.
- JAX is optional (`[jax]` extra); guarded import; macOS-Intel local dev has no
  working jaxlib -> FD fallback must remain first-class.

## 3. Target architecture

### 3.1 Objective protocol (new, internal)

```python
class Objective(Protocol):
    n_channels: int
    def init_state(y, sample_weight) -> RawState        # base scores (n,K)
    def step(F, y, sw) -> tuple[d, h]                   # per-channel step dirs
    def loss_value(F, y) -> float                       # for eval/early stop
    device_capable: bool                                # can consume device arrays
```

Implementations wrap what exists today:

- `LossObjective` — K=1, delegates to `_loss.py` losses (incl. GPU in-place
  kernels). `d = g`, `h = hess` (leaf Newton step unchanged; this is the K=1
  degenerate case of GGN).
- `DistributionObjective` — wraps `Distribution.nll_gradient` +
  `natural_gradient` (analytic Fisher). CUDA kernels per family, ported
  incrementally (Normal -> LogNormal -> Poisson; digamma families stay CPU
  until worth it).
- `CustomLikelihoodObjective` — JAX (vmap grad + per-sample GGN) or FD
  fallback. Upgrade from batch-mean diagonal Fisher to per-sample GGN with
  damping (spike-validated). JAX-on-GPU interops with numba-cuda via dlpack.
- `FormulaObjective` — FormulaBoost: user `f(theta, x)` + loss; J via
  autodiff or user-supplied; modes `precond='diag'|'full'` (default full,
  spike-validated), damping `lam`, global-fit initialization for base scores.

### 3.2 Unified trainer (one loop)

Single `fit_boosting(objective, X, y, ...)` used by all model facades:

- bin once (`ob.array`), upload once when backend=cuda
- state `F`: (n, K) — device-resident when backend=cuda and
  `objective.device_capable`
- per round: `(d, h) = objective.step(F, y)`; then per channel k:
  `fit_tree_gpu_native` (cuda) / `fit_tree` (cpu); `F[:,k] += lr * tree(X)`
  in-place on device
- eval: incremental raw-score updates for all eval sets (adopt the better
  `_distributional.py` pattern), `evals_result_`, callbacks, early stopping,
  `early_stopping_rounds` sugar for every model
- D2H only at eval/checkpoint boundaries

Non-goals for the shared loop v1: multi-GPU/Ray (stays on the K=1 fast path),
GOSS for K>1, sample_weight on CUDA (unchanged limitation).

### 3.3 Facades (no backward-compat constraint — zero users today, design the
ideal API and delete what it replaces)

- `GradientBoosting` — keeps its current fast paths in Phase A for perf
  reasons only (gate: no regression on `benchmarks/check_performance.py`),
  not for API stability; rename/reshape freely if the unified trainer wins.
- `NaturalBoost*` / `DistributionalGBDT` — rebuilt on the trainer; the old
  `_distributional.py` loop is deleted, not kept alongside. API may change
  where the unified design is cleaner.
- `FormulaBoost` — new. (Naming: "formula in, boosted parameters out"; the
  statistical anchor for docs is varying-coefficient / semi-parametric
  modeling, since x follows the user formula while theta(z) is non-parametric
  via trees.)

```python
def curve(theta, x):            # jax.numpy inside for autodiff
    a, b = theta
    return a * x ** jax.nn.sigmoid(b * x)

m = ob.FormulaBoost(model=curve, n_params=2, links=("log", "identity"),
                       loss="mse", precond="full", damp=1.0)
m.fit(Z, y, model_input=x, eval_set=[...], callbacks=[...])
m.predict_params(Z)                  # (a, b) per sample — the deliverable
m.predict(Z, model_input=x_new)      # counterfactual / extrapolation
```

- `register_loss` / `register_distribution` / `register_growth_strategy`
  unchanged; add `register_objective` later only if needed.

### 3.4 Vector-leaf (research track, structural option)

The trainer's "per channel k: fit one tree" block becomes a strategy; a
vector-leaf strategy (one tree, K-dim leaves) slots in without touching
objectives. Blocked on closing the onetree research gap (alpha-tempering
showed promise); not on any v1 critical path.

## 4. NGBoost feature-parity decisions

| Gap | Decision |
|---|---|
| CRPS as training score | Defer. NLL-only training; CRPS stays an eval metric. Revisit if users ask. |
| Line search | Won't do. Damped GGN + lr replaces it (spike: stable without). Document the difference. |
| Survival / censored | Do. Weibull AFT as `DistributionObjective` (censored NLL, event indicator via `sample_weight`-style arg). Needed for capability benchmark #1. |
| Multivariate distributions | Defer (post-1.x). |

## 5. Phasing (eval-first)

- **0. Eval harness before any engine work** — the yardstick every later
  phase is measured against:
  - `benchmarks/bench_probabilistic.py`: quality suite (UCI 9 + California +
    YearMSD; NLL/CRPS/RMSE/coverage/pinball vs NGBoost, paired splits) +
    speed suite (Modal A100; wall-clock + time-to-same-NLL vs NGBoost/PGBM),
    one command, JSON results in `benchmarks/results/`
  - Capability evals defined *before* FormulaBoost exists: sales-formula
    benchmark (generalize the spike: known-truth param recovery, in-range +
    extrapolation RMSE, vs global-fit / black-box / hand-rolled XGBoost) and
    Weibull AFT dataset + metric prep (C-index, calibration)
  - Golden parity fixtures: current NaturalBoost/DistributionalGBDT outputs
    at fixed seeds, so the Phase A refactor has a regression baseline
  - **Run it once on current main** -> committed "before" numbers; every
    later phase must move these numbers, not anecdotes
- **A. Trainer + objectives, CPU-correct** — extract unified loop; port
  DistributionalGBDT/NaturalBoost onto it; `FormulaObjective` +
  `FormulaBoost` facade (FD fallback works everywhere); behavioral parity
  tests vs current models (same seeds, same NLL within 1e-6).
- **B. GPU end-to-end for K>1** — Normal/LogNormal/Poisson gradient+Fisher
  CUDA kernels; device-resident F; `fit_tree_gpu_native` per channel; A100
  benchmark vs NGBoost (target >=10x at >=1M rows).
- **C. JAX-on-GPU custom objectives** — dlpack zero-copy bridge; FormulaBoost
  and CustomDistribution get the GPU path.
- **D. Weibull AFT + capability benchmarks** — survival objective; the two
  capability benchmarks (AFT vs XGBoost AFT; sales-formula vs hand-rolled).

Each phase lands independently with a numeric gate from Phase 0:
A = golden parity (same seeds, NLL within 1e-6) + no perf regression;
B = >=10x vs NGBoost at >=1M rows, quality suite unchanged;
C = custom-objective GPU path beats its own CPU path, results identical;
D = capability benchmarks won (AFT vs XGBoost AFT; formula vs baselines).

## 6. Risks / open questions

- FD fallback cost for K-param GGN (2K+1 NLL evals/sample/round) — acceptable
  for small K; document JAX as the fast path.
- Numerical parity CPU vs GPU for Fisher math (float32 on device vs float64
  numpy today) — decide dtype policy in Phase B; eval NLL stays CPU/float64.
- `GradientBoosting` K=1 fast path: keep only as long as it is measurably
  faster than the unified trainer; fold in and delete once the trainer
  matches it. No compat reason to keep two loops.
- Golden parity fixtures (Phase 0) are a correctness tool for the refactor,
  not an API-stability promise; intentional behavior changes just update the
  fixtures.
