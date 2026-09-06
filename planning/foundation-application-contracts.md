# OpenBoost v1 application coverage and case contracts

Date: 2026-09-05. Status: user-required design/acceptance scope, **not implemented architecture**.
Complements the [main plan](agent-boosting-foundation-plan.md) and
[E0–E7 evaluation](openboost-v1-evaluation.md); code-review baseline f30c2ed.
F0.1 [task cards](foundation-tasks.md) select data, splits and mathematics; at this design baseline,
downloaded hashes, actual baseline runs and v1 implementation remain incomplete.

The user explicitly requires **every following application, with individual delivery and acceptance**.
Each has target semantics, recipe, real workflow and independent evidence. Insurance/AFT have no
special status. Keep foundation-first design with no backward-compatibility requirement.

## 0. Required A1–A13 matrix

Every row is required, not a menu. F0 may choose concrete sources but must freeze versions/hashes,
licenses, splits, feature availability and task definitions. Source choice never removes a required
application. Compose shared components rather than copying industry-specific trainers.

| ID/recipe | Required application | Data and behavior | Independent acceptance/main evaluation |
|---|---|---|---|
| A1/R1 | Continuous regression | Frozen California Housing or another real source; train, predict, save/load | Numeric/missing, weights, generalization, RMSE; state geographic limits |
| A2/R1 | Binary classification | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult); categories/missing/probabilities | Mapping, unseen policy, class weights/link; log-loss, auxiliary AUC |
| A3/R1 | Multiclass | [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype); class mapping, K parameters/probabilities | Softmax, normalization, per-class quality, multi-logloss; threads/GPU memory |
| A4/R3 | Group ranking | [Microsoft MSLR](https://www.microsoft.com/en-us/research/project/mslr/); pairwise and NDCG-weighted lambda recipes | Query/pair dependencies, official folds, NDCG@10; retain usage terms, no cross-query splits |
| A5/R2 | Quantile regression | [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275); declared quantiles/weighted leaves | Temporal split, weighted pinball; exclude target components such as casual/registered and check feature availability |
| A6/R8 | Multioutput | [UCI Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons%2Btelemonitoring), two UPDRS targets; independent trees/shared vector topology | Subject splits, original score/interpolation semantics, per-target errors; benchmark accuracy is not clinical validity |
| A7/R4 | Event counts/frequency | Poisson+exposure; count and unit-exposure rate | Offset/weights once, units, Poisson deviance, exposure scaling |
| A8/R4 | Positive amounts/severity | Gamma on severity; declare claim-level or policy-average target | Positive support, selection, claim weights, Gamma deviance; A7 cannot substitute |
| A9/R4 | Aggregate loss/pure premium | Linked frequency/severity; Tweedie and frequency×severity workflow | Zeros, power, aggregate/annualized units, Tweedie deviance; A7/A8 cannot substitute |
| A10/R5 | Censored survival/AFT | Veterans' Administration lung cancer; fixed-scale log-normal event/right-censor workflow | Censored likelihood, risk/time/survival outputs, censored NLL and applicable probability/ranking metrics |
| A11/R6 | Distributional/NaturalBoost | Real regression or official ScoringBench subtask; two-parameter Normal and direction/step variants | Proper scores, Fisher/ordinary directions, links, calibration/width; coverage alone insufficient |
| A12/R7 | Structured/FormulaBoost | UCI Concrete; recipe Z, age x, two-parameter saturation hypothesis plus recovery/misspecification experiments | Independent Jacobian/directions, nonidentifiability, real quality/structural baselines; synthetic cannot replace real |
| A13/R9 | Model selection/train-many | Multiple configurations/objectives/groups on real tasks; M=1/8/32 and selected model | Prepared identity, independent seed/stop/failure, selected quality, full-set/selection cost |

Judge by each A-ID, not a minimum task count. Require at least 6 independent sources; multiple targets
on one source count once, replicated rows do not create scale. If unavailable, F0 may choose a public
source with the same semantics and record why; the item remains pending until chosen. Never drop
cases or replace poorly performing datasets after evaluation. Each F0 card specifies data/target/output,
baseline, oracle, CPU/CUDA boundary, quality/cost, implementation phase and evidence path.
A7–A10 are expanded below; all rows share delivery/acceptance obligations.

## 1. Insurance: Define frequency, severity and pure premium separately

| Task | Target and units | Initial strong baseline | Required semantics |
|---|---|---|---|
| Frequency | Claim count over coverage period; unit-exposure rate | Poisson GLM; XGBoost `count:poisson` | Count+log-exposure offset or verified rate+exposure weights; define separately, never double exposure |
| Severity | Positive eligible incurred claim payment or policy average per claim | Gamma GLM; XGBoost `reg:gamma` | Claim-level and policy-average weights differ; record zero payments, counts and selection |
| Pure premium/total loss | Expected loss per exposure or expected period total | Frequency×severity, XGBoost `reg:tweedie`, Tweedie GLM | Distinguish annualized/aggregate/zero loss; freeze power, offset/weight and units |

XGBoost demonstrates log-exposure through base_margin:
`E[count | X, e] = e * exp(F(X))`. Baselines must supply corresponding offsets during train,
validation and prediction; rate output must clarify exposure=1.
[Official offsets](https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html#offset).
Check actual Poisson/Gamma/Tweedie settings against pinned versions.
[Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).

Candidate sources: freMTPL2freq/freMTPL2sev, OpenML41214/41215. The scikit-learn example is a
starting point for joins, targets and GLM comparisons. Its clipping/zero handling are processing
choices, not unexplained raw-data truths.
[Official insurance example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html).

F0 freezes source/version/hash, claim-policy joins, exclusions/clipping, missing/categories and splits.
Related policy/entity records cannot leak across train/test. Use temporal splits if valid time
information exists; otherwise state the limit, never invent temporal validation. Fit encoders/bins on training only.

Two acceptance layers:

- **Math/workflow:** nonunit exposure, nonunit/zero weights; offset and weight each apply once.
  Doubling exposure for fixed Poisson model doubles count but not rate. Validation, early stopping
  and loaded inference retain the same definitions.
- **Application:** appropriate deviance in matching units, totals and predefined group predicted/actual
  comparisons. Full distributions additionally need NLL/CRPS/tail metrics. Mean-only Gamma/Tweedie
  objectives do not imply calibrated distributions; exposure-scaled means do not prove aggregation laws.

Deliver A7, A8, A9 separately. Dependency order is allowed; Poisson success does not complete Gamma,
Tweedie or composition workflows.

## 2. Survival/AFT: Observations are not always event times

XGBoost `survival:aft` uses lower/upper labels for complete, right, left and interval censoring,
with declared noise family/global scale. Use the pinned version's supported entry point, e.g.
DMatrix label_lower_bound/label_upper_bound.
[Official AFT tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html).

| Observation | Labels | Continuous-time likelihood contribution |
|---|---|---|
| Event at t | lower=upper=t, t>0 | Density f(t) |
| Right censored | lower=t, upper=+inf | Survival S(t) |
| Left censored | lower=0, upper=t | F(t) |
| Interval censored | 0<lower<upper<+inf | F(upper)-F(lower) |

These definitions need independent validation, not production loss as the only oracle. Use stable
log-density/log-survival/log-CDF differences and test extreme tails/narrow intervals. Equal endpoints
mean density, not zero interval probability. Declare log-time→time Jacobian, gradient signs and
curvature approximations/clipping. Check mathematics separately from stabilization.

Early recipes fix one noise family/global scale and verify events/right censoring. F0 target
schemas distinguish every case above; unsupported censoring explicitly fails. Delayed entry/left
truncation differs from left censoring and must be rejected separately, not recoded as censoring.

Candidate real checks: NCCTG lung from XGBoost tutorial or scikit-survival Veterans' Administration
lung cancer. F0 pins one source/version, event encoding, time units, preprocessing and splits. These
are small real smoke/quality tasks, not scaling evidence. Other event-time applications include
lifetimes, equipment failures and contract termination, with actual event definitions.

Baselines include XGBoost `survival:aft`, CatBoost SurvivalAft and an appropriate parametric AFT.
Adapt CatBoost upper sentinel-1; its objective is documented CPU.
[CatBoost objectives/devices](https://catboost.ai/docs/en/concepts/loss-functions-regression).
Cox can provide ranking/risk comparisons only for supported outputs; hazard/risk is not time/probability.
Cox risk sets depend across rows, so F0 must not impose row independence on every objective.
Native Cox is not a prerequisite for F1.

Evaluation requirements:

- Censored NLL under applicable assumptions, monotone survival over time, valid probabilities and consistent quantiles.
- Distinguish ranking/probability quality. Beyond C-index, use appropriate right-censor IPCW
  Brier/integrated Brier and time calibration, declaring censoring assumptions/support. Freeze
  censoring estimates/time grids during training/development, not from final test results.
- Do not treat right-censor times as failures for ordinary RMSE/coverage, or apply right-censor
  IPCW tools to left/interval censoring without justification. Report unevaluable cells/reasons.
  [Official survival evaluation](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html).

At the reviewed historical baseline, OpenBoost WeibullAFT accepts observed time+event for right
censoring; it is not evidence of arbitrary interval support. Covariate-dependent shape is more
general distributional survival regression; the class name does not prove standard fixed-residual AFT assumptions.

## 3. Concrete foundation and execution constraints

1. **F0.1:** cover R1–R9/C1–C7/A1–A13 with inputs, targets, outputs, three-library settings, metrics
   and processing. F0.2 needs group, multioutput, links, weighted quantiles, exposure, censoring,
   distribution/formula and multiple-run oracles; each case must be independently falsifiable.
2. **F1.1:** Problem retains typed targets, offset/exposure, sample weights and bounds; shared
   splits/indexing stay aligned. Upper+inf can be valid; do not require every label finite.
3. **F1.5/F1.6:** connect case semantics and all CPU recipes. Verify at least two rounds,
   predictions, accepted state and new-format round trips without application-name runner branches.
4. **F2:** modifications may involve insurance information constraints or survival update/curvature;
   compare existing configurations first and distinguish correct implementation from quality improvement.
5. **F3:** list offset, censoring, evaluation and prediction GPU capabilities separately. Normal GPU
   success does not validate insurance/AFT; declared support needs corresponding CPU/GPU semantics and quality.
6. **F4:** every A1–A13 completes real acceptance with quality, cost, failures and unsupported boundaries.
   Any missing/failed item prevents complete-v1 acceptance.

This design update changed scope/gates only; it did not download data, run baselines or modify training code.
