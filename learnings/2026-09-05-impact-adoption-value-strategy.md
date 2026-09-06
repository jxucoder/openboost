# 2026-09-05: OpenBoost impact, adoption, and value strategy research

## Context

Question: where should OpenBoost invest next to improve research impact, actual
adoption, and user value?

This document records research recommendations, not approved product decisions,
and does not change the AGENTS.md mission or priorities. It assumes a small team
and a 6–12 month horizon, prioritizing real use before amplifying research and
commercial value. Team size, GPU budget, industry relationships, and retention
are unknown. Suggested numbers and deadlines below are experiment gates, not
growth forecasts.

Research date: 2026-09-05. Evidence is separated into local measurements, online
source/project statements, external primary sources, and untested hypotheses.
No user interviews, willingness-to-pay tests, or new GPU/third-party quality
benchmarks were conducted.

### Versions that must remain distinct

- Local checks used `05cd8bc800595a2f40c4d08f51afb697968b9b3e`; local `origin/main`
  was `504fdd0bfc60e5d8e7518250e087fb7e4766d1b4`. The initial workspace was clean,
  11 commits ahead of that remote-tracking reference.
- Online `main` read through the GitHub connector already contained FormulaBoost,
  WeibullAFT, a unified trainer, and updated probabilistic-modeling positioning.
  Returned modification dates were 2026-08-17/18. Those files were absent from
  the local snapshot above.
- Online README, old raw.githubusercontent.com search caches, local checkout,
  and PyPI entry points disagreed. Research prioritized connector-returned source
  and limited local results to the local SHA. No branch was pulled, merged,
  reset, or pushed.
- Online sources use mutable `main` links; their immutable SHA was not frozen.
  The 26 local test passes cannot verify the newer online architecture.

Online sources: [README](https://github.com/jxucoder/openboost/blob/main/README.md),
[unified trainer](https://github.com/jxucoder/openboost/blob/main/src/openboost/_trainer.py),
[objectives](https://github.com/jxucoder/openboost/blob/main/src/openboost/_objectives.py).

## Decision or Result

Recommended direction: **build a small, complete product around testable
probabilistic predictions; use NaturalBoost to attract adoption, real risk tasks
to demonstrate value, and bounded FormulaBoost experiments to explore research
contributions.**

A possible eventual positioning:

> OpenBoost helps Python teams model and evaluate predictive distributions for
> tabular data, incorporating domain formulas when needed.

“Calibration-first” should describe training, independent calibration, diagnostics,
and deployment verification. `predict_interval` or approximately uniform aggregate
PIT alone cannot promise reliability across all groups, tails, or distribution shifts.

The immediate question is: **which external team will use OpenBoost again on its
second task, and why?** This tests the direction better than another model class.

### 1. What each goal optimizes

| Goal | Desired outcome | Evidence to observe first |
|---|---|---|
| Impact | Others complete previously difficult analyses, methods, or decisions with OpenBoost | Independent reproduction, external integrations, research use, real cases |
| Adoption | External users get started and continue using it | First-success rate, time to useful output, second use, four-week retention |
| Value | Benefits outweigh migration and maintenance | Less compute/engineering time at matched quality, or improved prespecified business decision loss |

“Independent teams × sustained use × actual improvement per team” can guide
qualitative impact assessment; it is not an estimated growth model. Stars and
downloads indicate reach. Downloads include CI and repeated installation and
cannot directly measure active users.

### 2. Existing assets and missing evidence

**Locally checked assets:** NaturalBoost, distribution-parameter predictions,
sample weights, exposure offsets for some distributions, NLL/CRPS/quantile/interval
evaluation, PIT/reliability/recalibration tools, and corresponding tests. There
were 26 focused passes. Historical implementation:
[distributional](https://github.com/jxucoder/openboost/blob/05cd8bc800595a2f40c4d08f51afb697968b9b3e/src/openboost/_models/_distributional.py),
[utils](https://github.com/jxucoder/openboost/blob/05cd8bc800595a2f40c4d08f51afb697968b9b3e/src/openboost/_utils.py); tests:
[distributional tests](../tests/test_distributional.py), [utils tests](../tests/test_utils.py).

**Existing local third-party comparison:** a CPU NaturalBoost/NGBoost raw JSON
covering three datasets, one seed, and one timing per case showed similar quality
and mixed speed results, insufficient for general superiority. Its metadata does
not cover all current AGENTS.md provenance requirements.
[Raw results](../benchmarks/results/ngboost_comparison_20260720.json).

**New online capabilities:** FormulaBoost calls formula model → FormulaObjective
→ fit_boosting. Formula loss currently supports MSE only; finite-difference
Jacobians and GGN run on CPU while trees may use GPU. The unified objective has
device paths for Normal/Poisson; the older statement that all distribution
gradients run on CPU cannot be applied to this version.
[FormulaBoost source](https://github.com/jxucoder/openboost/blob/main/src/openboost/_models/_formula.py),
[objectives](https://github.com/jxucoder/openboost/blob/main/src/openboost/_objectives.py).

**Online experiments are useful leads, not independently verified conclusions
of this study.** The benchmark page reports new GPU timings, 8 UCI datasets,
and synthetic FormulaBoost/Weibull experiments, while acknowledging 3 missing
UCI datasets and no XGBoostLSS/LightGBMLSS comparisons. Its opening calls the
results committed runs, but its ending says JSON is in ignored directories and
tables were transcribed from task records. This study did not verify frozen
raw results and environments corresponding to every table entry, so speed
multipliers are not strategic premises.
[Online benchmark source](https://github.com/jxucoder/openboost/blob/main/docs/benchmarks.md).

**Adoption entry points remain fragmented.** Local README/quickstart emphasizes
general GBDT; online README emphasizes distributional regression; the default
PyPI page still presents the old stable entry while release history lists
`1.0.0rc1`. Align source, version, installation, docs, and reproducible examples
before a stable release after gates pass.
[PyPI](https://pypi.org/project/openboost/),
[online README](https://github.com/jxucoder/openboost/blob/main/README.md).

### 3. What competition implies

| Direction | Alternatives and primary evidence | Implication for OpenBoost |
|---|---|---|
| General distribution prediction | [NGBoost](https://stanfordmlgroup.github.io/ngboost/1-useage.html) offers predictive distributions, proper scores, and survival | Distribution output alone is insufficient to motivate switching |
| Rich distribution families | [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/) includes distributions, mixtures, flows, and multiple targets | Avoid competing on distribution count; include it on relevant tasks |
| GPU probabilistic regression | [PGBM](https://github.com/elephaint/pgbm) targets large-scale probabilistic regression with GPU support | It is a reasonable task-matched baseline; calling it a reference in our docs does not exclude it |
| Modifiable Python GPU boosting | [Py-Boost](https://github.com/sb-ai-lab/Py-Boost) emphasizes extensibility, multi-output, and GPU | Readable Python needs examples measuring effort saved when adding methods |
| Intervals and risk control | [MAPIE](https://mapie.readthedocs.io/en/stable/) provides conformal/calibration/risk control; [skpro](https://skpro.readthedocs.io/en/stable/) provides probabilistic interfaces, metrics, and pipelines | Favor compatibility/integration over rebuilding a general uncertainty platform |
| Gaussian uncertainty | [CatBoost](https://catboost.ai/docs/en/references/uncertainty) supports uncertainty predictions | Normal comparisons should extend beyond NGBoost |
| New tabular models | [Official TabPFN-3 report](https://priorlabs.ai/technical-reports/tabpfn-3) emphasizes scale, inference efficiency, and calibrated distributions | Do not assume tabular foundation models only handle small samples; independently compare official performance claims |

These sources establish alternatives, not paying demand. Scenario priorities
below reflect current code fit, accessible data, and verification cost.

### 4. Choose an initial scenario that can demonstrate value

**First candidate: model development and validation for insurance frequency/loss.**
Initial users would be experimental actuarial researchers, insurance data
scientists, and risk-modeling consultancies, rather than buyers of a complete
enterprise pricing platform.

Exposure, nonnegative/count targets, distribution parameters, and tail assessment
fit existing code, with public datasets and mature baselines. The scikit-learn
freMTPL2 example demonstrates Poisson frequency, Gamma severity, and Tweedie pure
premium and can anchor a reviewable comparison.
[Official example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html).

Clarify actual needs. Expected pure premium alone may be adequately served by
GLM/GBDT. Test whether multiple quantiles/threshold probabilities, heterogeneous
dispersion research, or probability quality by exposure reduce engineering work
or improve decisions. A single threshold may be handled by a classifier; do not
assume a full distribution is necessary.

Start a real case with **Poisson frequency + exposure**, then compare flexible
count distributions. Model and validate severity separately. Local Tweedie
training uses an approximate dispersion gradient and a moment-matched Gamma
approximation for positive quantiles, while `nll()` follows a separate density
path. Interfaces and shape tests cannot validate its tails.
[Distribution implementation](https://github.com/jxucoder/openboost/blob/05cd8bc800595a2f40c4d08f51afb697968b9b3e/src/openboost/_distributions.py).

Distinguish exposure semantics: current offset checks establish mean scaling
with exposure, not arbitrary Gamma/Tweedie distribution aggregation laws. Define
count, per-claim severity, aggregate loss, annualized loss, and weighting before
comparing dispersion and tails.
[Current exposure implementation](https://github.com/jxucoder/openboost/blob/05cd8bc800595a2f40c4d08f51afb697968b9b3e/src/openboost/_models/_distributional.py).

| Scenario | Current role | Condition for greater priority |
|---|---|---|
| Insurance frequency/loss | First candidate for real value validation | An external team supplies evaluation goals, runs baselines, and wants repeat use |
| Large-sample regression for NGBoost users | Most direct developer adoption entry | Clear compute or engineering benefit on real data at similar predictive quality |
| Demand/inventory probabilistic prediction | Alternative if insurance users are inaccessible, or later expansion | Partner data and explicit stockout/inventory cost, time splits, and existing forecasting comparisons |
| FormulaBoost structured curves | Bounded research experiment | A real task has credible formulas, sufficiently varying structural inputs, and identifiable parameters |
| Weibull time-to-event | Increase priority with a domain collaborator | Independent real survival data, correct censoring assessment, and strong lifelines/NGBoost baselines |

Forecasting already has [MLForecast interval workflows](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/prediction_intervals.html).
For that alternative, provide an integrable regression component instead of
expanding into a complete time-series platform.

User access may overturn the initial priority: if no useful insurance collaboration
appears within two weeks but a demand/reliability team offers data and time,
follow the verifiable demand.

### 5. Make adoption a complete workflow

Minimum stable experience: install → run a public real case → connect user data
→ train/independently calibrate/test → compare baselines → save/reload → reproduce report.

1. **One default entry.** Choose a target type and provide data to get the first
   probability-quality report. Research primitives, other models, and experimental
   backends follow later. Build on the online README's existing positioning change
   rather than repeating the old audit.
2. **One trustworthy report template.** Include means/quantiles/intervals,
   CRPS/NLL, coverage/width, prespecified group/tail checks, model/data versions,
   and actual backend. Serve this workflow before building an all-model dashboard.
3. **Separate training and calibration.** Do not use calibration data for final
   scoring or uncontrolled repeated tuning. Evaluate distribution quality on an
   independent test set. `PITRecalibrator` currently exposes PIT/CDF-level transforms,
   not a complete saveable, sampleable predictive distribution with calibrated
   quantiles. [Local implementation](https://github.com/jxucoder/openboost/blob/05cd8bc800595a2f40c4d08f51afb697968b9b3e/src/openboost/_utils.py).
4. **A small preproduction boundary.** Verify declared seed, weights, exposure,
   missing/categories, early stopping, CPU inference, save/load, and version
   compatibility; make fallback explicit. Retain ordinary-CPU trial access.
5. **Use the ecosystem as an entry point.** Improve sklearn compatibility and
   NGBoost migration examples first, then choose skpro or MAPIE integration based
   on demand. Maintainers determine upstream acceptance; a PR does not guarantee traffic.

“Useful report in 15 minutes” is a design target to measure with new users, not
an established result. Do not make extra run counts, complex parameters, or
unnecessary dependencies concepts users must understand.

### 6. Research impact: what structure contributes

FormulaBoost merits bounded investment because it combines a domain-given
`f(theta(z), x)` with nonparametric parameter heterogeneity. Ask: **under which
data support, formula misspecification, and parameter coupling conditions does
it improve structural-input extrapolation, parameter recovery, or decisions?**

Do not claim varying coefficients or full natural gradients as inherently novel.
Existing [tree boosted varying coefficient research](https://arxiv.org/abs/1904.01058),
[gamboostLSS](https://search.r-project.org/CRAN/refmans/gamboostLSS/html/mboostLSS.html),
and NGBoost natural gradients motivate further literature and comparison work;
they do not establish that this study completed a novelty review.

Corrections relevant to publication and positioning:

- Online survival docs say NGBoost lacks censored likelihood, contradicting its
  [official survival docs](https://stanfordmlgroup.github.io/ngboost/1-useage.html#survival-regression).
  Online README also calls NGBoost a fixed catalogue, but its
  [developer guide](https://stanfordmlgroup.github.io/ngboost/5-dev.html) allows new
  distributions and scores. Correct these before promotion.
- Covariate-dependent Weibull shape is not unique across the ecosystem.
  [lifelines ancillary regression](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#modeling-ancillary-parameters)
  supports shape modeling and warns it generally loses standard AFT form. Verify
  nonlinear parameter surfaces, computation, and usability rather than unconditional uniqueness.
- XGBoost custom-objective Hessian inputs have diagonal-structure restrictions,
  but official docs discuss alternative curvature and approximations. Lack of
  full-matrix input does not prove every outer algorithm cannot use coupled
  directions. [Official explanation](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html).
- Online FormulaBoost tables show similar prediction/extrapolation errors for
  `diag` and `full`; the main full-matrix lead concerns parameter recovery.
  Synthetic cases cannot establish that real tasks require full GGN.
  [Report](https://github.com/jxucoder/openboost/blob/main/docs/benchmarks.md).

For the current scalar-MSE FormulaObjective, a direct derivation applies. For
single-sample Jacobian vector `j`, residual `r`, and positive damping `lambda`:

```text
G = j j^T
g = r j
(G + lambda I)^(-1) g = r j / (lambda + ||j||^2)
```

Thus the current per-sample full-GGN direction reduces to sample-level gradient
scaling. It differs from component-wise `diag` scaling, but this form does not
require general dense matrix inversion. Independent numerical checks at K=2/5
matched dense solves with maximum absolute errors `2.78e-16` / `5.00e-16`.
This concerns only the present scalar-MSE, positive-damping, per-sample mathematics,
not complete training equivalence, multi-output residuals, aggregated curvature,
or arbitrary numerical safeguards.
[Corresponding source](https://github.com/jxucoder/openboost/blob/main/src/openboost/_objectives.py).

Compare global, diag, full, and gradient-scaling versions of the same formula
with matched initialization/tuning, sensible structured alternatives, and
formula-free baselines. Include misspecified formulas, weak identification,
insufficient structural-input variation, reparameterization, and different noise.
A single scalar-response Jacobian has rank at most one; matrix preconditioning
cannot invent missing identification information.

If results are positive, produce a report on when structured boosting helps,
with independently runnable experiments. Consider software publication after
external use and software quality accumulate. [JOSS requirements](https://joss.readthedocs.io/en/latest/submitting.html)
provide review criteria; publication does not replace adoption evidence. If the
user prioritizes publication impact, increase this investment while narrowing
to one clear research question.

### 7. Evidence design: answer why users would switch

Keep three questions and their conclusions separate:

| Question | Experiment | Supported conclusion |
|---|---|---|
| Is general probabilistic quality reliable? | Complete the official ScoringBench protocol using its public comparison workflow | Quality within that protocol and complete dataset collection |
| Is scaling worthwhile? | At least three real datasets, multiple scales/repeats, CPU/CUDA and strong baselines | Cost/scale benefit on declared hardware at the quality threshold |
| Why adopt? | One real domain case plus external reproduction/reuse | Workflow-specific engineering/decision benefits and migration cost |

[ScoringBench](https://github.com/jonaslandsgesell/ScoringBench) is a suitable
independent quality entry; local [adapters and two protocols](../benchmarks/scoringbench/README.md)
already exist. Label official quality and larger-sample extension separately.
Benchmark completion should not block starting user interviews.

Compare relevant NGBoost, XGBoostLSS/LightGBMLSS, PGBM, CatBoost uncertainty, or
quantile + conformal. Include currently accessible TabPFN if resources permit.
Filter by target/distribution support before staged narrowing, recording
unsupported cases, failures, and budget differences.

Prespecify primary metrics, noninferiority bounds, splits, tuning budgets, and
business decisions. Report paired differences and uncertainty intervals.
`p > 0.05` does not establish equivalence/noninferiority. Calibration improvement
also needs width, proper scores, and downstream loss; infinitely wide intervals
cannot constitute a useful calibration win.

New GPU results require exact source SHA/dirty state, dataset/version/hash,
split/seed, OS/CPU/RAM/threads, GPU/driver/CUDA, dependencies, full commands,
cold/warm policy, and actual backend/fallback. Measure end-to-end fit, prediction,
memory, and transfers. GPU OpenBoost versus CPU NGBoost comparisons must report
resources/cost and include relevant GPU alternatives. Synthetic data tests
mechanisms; real data establishes scenario evidence.

Survival assessment must account for censoring. Unknown complete event times
cannot supply ordinary real-data coverage. Choose censored NLL and applicable
IPCW/Brier/calibration methods under stated identification and independent-censoring
assumptions. FormulaBoost scenario extrapolation is not automatically causal;
observational curves alone do not justify interpreting price/dose changes as interventions.

### 8. A 90-day execution plan

| Time | Main work | Deliverables | Suggested decision gate |
|---|---|---|---|
| Days 1–14 | Reconcile local/online state, correctness/evidence gaps, prepare a domain walkthrough, conduct discovery | Version/capability table, runnable case, interview outline, baseline protocol | Contact about 10 suitable people; at least 3 offer a real evaluation task or one comparison |
| Days 15–30 | Complete third-party quality runs and real Poisson/exposure baseline; observe onboarding | Raw results, failure list, installation-to-report observations | At least 2 external users succeed independently; identify the 3 most common obstacles |
| Days 31–60 | Improve workflow based on obstacles; freeze CPU/CUDA parity/scaling; bounded FormulaBoost falsification | Reusable core, real case study, structured-method comparisons | A domain benefit covers migration cost, or stop that scenario; continue research only with evidence |
| Days 61–90 | Prepare stable release, one ecosystem integration, public technical case, partner pilot according to results | Versioned tutorials/reports, upstream submission materials, reusable onboarding | 5 external teams have used real data; at least 3 continue after four weeks or on a second task; 1 independent reproduction/integration |

Outreach, upstream submissions, and releases above are future recommendations;
none occurred in this study. Upstream acceptance timing is outside our control;
distinguish completed submission materials from acceptance.

Suggested small-team allocation: roughly 60% core reliability/complete experience,
25% external use/real evidence, 15% one FormulaBoost research experiment.
Data/compute work may interleave within one person's schedule; this does not
call for launching three products.

Every new feature must address an external obstacle or a prespecified research
hypothesis. GAM/DART/linear leaves, Ray/multi-GPU/out-of-core/train-many receive
no independent product investment until the main evidence gate passes and
concrete demand exists.

### 9. Acquiring users and commercial value

Start with three materials: a real probabilistic-modeling walkthrough, a
reproducible performance/quality comparison, and an NGBoost migration guide.
Use problem-centered titles, such as “Exposure, predictive distributions, and
independent calibration assessment in one Python workflow.” Link every number
to raw results.

Prioritize authors/maintainers of existing NGBoost projects, count/loss research
teams, and risk consultancies offering validation tasks. Offer concrete paired
experiments and reproduction help; record why users leave or continue. Broad
launch traffic is not the first objective.

Interview about actual work: how was the most recent probabilistic prediction
completed; which step cost the most time or trust; which decision used its outputs;
what alternatives are used; what result would justify switching; can the team
compare on its own data within two weeks? Verbal agreement is weaker evidence
than data, engineering time, and repeated use.

Validate commercialization in order:

1. Auditable benchmark/migration/calibration services to test payment for shorter delivery.
2. If multiple teams repeatedly request them, consider maintenance support,
   private deployment, and reproducible versions/reports.
3. Evaluate hosted training/evaluation only after clear recurring paid demand.

Estimate value as “engineering time saved + compute cost change + verified
decision improvement − integration/maintenance cost.” Do not mechanically convert
CRPS percentages into revenue or invent TAM, pricing, or ARR without interviews
and costs.

Keep the open-source core easy to use. Build trust by solving shared problems,
then determine which services merit payment; hypothetical commercialization
should not introduce premature friction.

### 10. Conditions that would overturn the direction

| Observation | Action |
|---|---|
| Similar general quality but no compute/engineering benefit | Narrow to differentiated domain/custom objectives; stop general-superiority claims |
| Users only need intervals and existing models + MAPIE/skpro suffice | Integrate lightly or acknowledge the better alternative; do not expand into general UQ |
| Users want reports but consistently refuse trainer replacement | Test independent evaluation demand with a small adapter before changing product focus |
| GPU lacks real end-to-end economic benefit | Make GPU optional, optimize from profiling, and pause multi-GPU expansion |
| FormulaBoost only works on synthetic data generated by an exactly correct formula | Retain research examples and pause general extrapolation claims |
| Full-GGN benefits disappear after matched formula/initialization/tuning | Attribute contribution to structure/usability, not matrix form as independent innovation |
| Repeated onboarding rounds produce no second use | Investigate exits and change scenario/experience, rather than adding model types |
| Multiple teams repeat use and pay for support | Increase maintenance/integration for that scenario before discussing a larger product |

## Changes

- Added this study to preserve sources, version differences, recommendations,
  experiment gates, and unverified hypotheses.
- Added an entry to the learning index.
- Did not modify models, runtime code, public positioning, or benchmark implementation.

## Verification

### Focused local behavior checks

```bash
OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache \
  uv run --no-sync pytest tests/test_distributional.py tests/test_utils.py \
  -n 0 -q -k 'exposure or PIT or ReliabilityDiagram or ProbabilisticMetricsWithOpenBoost'
```

Result: `26 passed, 136 deselected in 7.21s`, Darwin, Python 3.12.12. This verifies
selected local behavior, not full regression, the online architecture, or CUDA.

### Local reproducibility finding

The independent-process experiment below fixes data and changes only the global
NumPy seed. Local NaturalBoost has no `random_state` constructor parameter and
tree row subsampling uses global `np.random.choice`. Maximum prediction change:
`0.1888485550880432`. This reproduces a local randomness-control gap; it does not
establish whether the online version has fixed it.

```python
import inspect
import numpy as np
from openboost import NaturalBoostNormal

rng = np.random.default_rng(7)
X = rng.normal(size=(240, 4)).astype(np.float32)
y = (X[:, 0] + 0.4 * rng.normal(size=240)).astype(np.float32)
preds = []
for global_seed in (1, 2):
    np.random.seed(global_seed)  # Deliberately probe reliance on global state.
    model = NaturalBoostNormal(n_trees=8, max_depth=2, subsample=0.7)
    model.fit(X, y)
    preds.append(model.predict(X))
print('random_state' in inspect.signature(type(model)).parameters)
print(float(np.max(np.abs(preds[0] - preds[1]))))
```

### FormulaObjective mathematical check

Continue from the preceding `rng` state to check the scalar-MSE rank-one identity;
this does not execute complete training of the online model:

```python
for k in (2, 5):
    J = rng.normal(size=(100, k))
    residual = rng.normal(size=100)
    damp = 1.0
    matrices = J[:, :, None] * J[:, None, :] + damp * np.eye(k)[None, :, :]
    g = residual[:, None] * J
    full = np.linalg.solve(matrices, g[..., None])[..., 0]
    simplified = g / (damp + np.sum(J * J, axis=1))[:, None]
    print(k, float(np.max(np.abs(full - simplified))))
```

Results: K=2 `2.7755575615628914e-16`; K=5 `4.996003610813204e-16`.
This is an algebra check, not a quality or performance benchmark.

Historical documentation checks: 9 relative file links existed, 2 Python examples
compiled, all 7 template sections were present, code fences balanced, and
`git diff --check` passed. This document is outside MkDocs navigation and changed
no production code; documentation checks were not reported as model regression.

## Failed Attempts

- Search-index README, raw files, and commit history disagreed. Switched to
  GitHub connector source reads and separately recorded the local tested SHA.
- `git ls-remote` failed because the local environment could not resolve GitHub.
  Used the available connector without changing network settings or repository state.
- Could not confirm full raw provenance from newer transcribed benchmark tables;
  did not repeat their GPU numbers as verified claims.

## Risks and Follow-ups

- User demand and retention are the largest unknowns, not the number of candidate
  models. Prioritize externally supplied evaluable tasks.
- Reconcile online and local code before fixes and preserve unpushed commits.
- Initial engineering checks should cover seed control, probability/calibration
  consistency, Tweedie approximation limits, persistence, and real CUDA parity.
- This study does not claim established FormulaBoost novelty, ScoringBench
  acceptance, independently verified GPU performance, or a verified paying market.
- If research publication or short-term revenue becomes primary, adjust experiment
  order instead of mechanically retaining the suggested allocation.

## Commits

- This document and the learning index form an independent documentation commit;
  Git history records its SHA.
