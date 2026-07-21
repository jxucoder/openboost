# Single-tree vector-leaf boosting for multi-parameter distributions (prototype)

## Method

NaturalBoost currently fits **one tree per distribution parameter per round**.
This prototype fits **one tree per round** whose leaves hold a vector value —
one component per parameter — by generalizing the scalar second-order formulas:

- leaf value: `v_L = -(Σ_{i∈L} F_i + λI)^{-1} Σ_{i∈L} g_i`
- split gain: `‖Σg‖²` in the `(ΣF + λI)^{-1}` metric, left + right − parent

where `g_i` is the per-sample NLL gradient w.r.t. raw (link-space) parameters
and `F_i` the per-sample Fisher information. For `k=1` both reduce exactly to
the classic scalar gain/leaf formulas. Implemented here for Normal(μ, σ) with
raw params (μ, log σ), whose Fisher is diagonal: `diag(1/σ², 2)`.

Pure-NumPy CPU prototype (`vectorleaf_normal.py`); correctness is verified by
finite-difference gradient checks and a manual-formula check of split choice
and leaf values (depth-1 case). Wall-clock is not comparable to the Numba
library; tree counts and NLL-per-round are the meaningful metrics.

## Experiments (`run_spike.py`, seed 42, depth 4, lr 0.1, λ 1)

Both models score **best-on-validation** (early-stopping regime). Library =
`NaturalBoostNormal` (2 trees/round).

| Experiment | one-tree best NLL (trees) | library best NLL (trees) | Δ (one-tree − library) |
|---|---|---|---|
| E1 coupled heteroscedastic (20k) | **2.0781** (90) | 2.0853 (156) | −0.007 |
| E2 decoupled params (20k, adversarial) | 2.0930 (140) | **2.0892** (140) | +0.004 |
| E3 California housing | 0.5453 (190) | **0.4910** (476) | +0.054 |

## Findings

1. **Tree efficiency**: on synthetics, one-tree reaches equal-or-better NLL
   with 1.5–1.7× fewer trees at the optimum; sharing tree structure across
   parameters is not harmful even when parameters depend on disjoint features
   (E2 is only +0.004).
2. **Real-data gap**: E3 trails by 0.054 nats. Not explained by scale
   overfitting — a slower scale channel (`channel_lr=(1.0, 0.5)`,
   `run_variant.py`) made E3 *worse* (0.6245). Likely a mean-capacity effect
   (the per-parameter baseline gives the mean its own 476 trees). Open.
3. **Late-training scale collapse**: past the validation optimum, the scale
   channel overfits aggressively (the joint gain actively finds low-residual
   regions; the Fisher for log σ is a constant 2, so there is no curvature
   counterweight). Interventions tried:
   - log-scale box (`s_box`): bounds the damage (E1@600r: 8.5 → 3.2), doesn't
     prevent degradation;
   - backtracking line search on train NLL: **inert** — overfitting *improves*
     train NLL, so full steps are always accepted (bit-identical runs);
   - per-channel learning rate: no help (see above).
   Early stopping on a validation set is currently mandatory.
4. **Capacity does not explain E3 either** (`run_depth5.py`): the depth-4
   comparison gives one-tree half the per-round leaf budget (16 vs 2×16), but
   equalizing it makes things *worse* — E3 best NLL is 0.5812 at depth 5 and
   0.5751 at depth 6 vs 0.5453 at depth 4 (library: 0.4910). Four hypotheses
   for the real-data gap are now falsified: structure sharing (E2), step size
   (inert line search), scale overfitting (slower scale channel hurt), and
   per-round capacity (deeper trees hurt).
5. **Remaining hypothesis (untested)**: precision weighting. The joint
   μ-channel uses gradients scaled by 1/σ², so regions with small predicted σ
   dominate split selection and leaf values; the per-parameter baseline fits
   its mean trees to Fisher-preconditioned targets (≈ raw residuals), i.e.
   effectively *unweighted*. If early σ estimates are noisy on real data,
   precision weighting amplifies their errors into the mean fit. Testing this
   (e.g. tempered weighting `(1/σ²)^α`) is the natural next experiment.
6. **Open questions** for a production version: the precision-weighting
   question above; principled regularization of the scale channel inside the
   joint gain; non-diagonal Fisher families (k×k solve per candidate split).

## Run

```bash
OPENBOOST_BACKEND=cpu uv run python development/onetree/run_spike.py
OPENBOOST_BACKEND=cpu uv run python development/onetree/run_variant.py
```

Results are written to `spike_results.json`.
