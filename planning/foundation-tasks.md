# OpenBoost v1 F0.1: Algorithm task cards, alternatives and interface sketches

Date: 2026-09-05. Code-review baseline: 7b57436. Status:**F0.1 specification; implementation,
independent references and formal evaluation were incomplete at the design baseline**.
Sources:[main plan](agent-boosting-foundation-plan.md),[applications](foundation-application-contracts.md),
[v1-plan-r2 evaluation](openboost-v1-evaluation.md). All A1–A13 are required; dependencies determine order.

This file specifies tasks, mathematics, data, comparators and falsifiable results. F0.3 freezes actual
bytes/split hashes, wheels/drivers, 16 search configurations and budgets; none is claimed measured here.
F0.2 writes oracles; F1 production; F2/F3/F4 author/GPU/real-task evidence. Future test/module names
are deliverable contracts, not existing APIs. [Construction design](foundation-construction-design.md)
provides records, ops, tree/transaction/inference/device execution and B01–B14; cards do not replace it.

## 1. Shared contracts and delivery matrix

### Inputs, two-round state and failures

- Row-aligned X[N, F], stable row_id[N], typed targets, optional weight[N], offset[N, K], query/entity IDs
  and structure_input. Weights are finite, nonnegative with positive sum. Validate legitimate missingness/
  censor bounds separately from invalid NaN. Auxiliary fields never automatically become weights.
- C1 numeric/missing/CPU categories. Train-fitted mappings, dedicated missing state and declared
  unknown-as-missing route are saved. Initial F1 categories use exhaustive one-category-vs-rest,
  avoiding hidden target encoding. If quality fails, change the algorithm, not remove categories or relax E3.
- Tiny fixtures run at least raw0→geometry0→learner0→decision0→raw1→geometry1→learner1→decision1→raw2.
  Compare intermediates, trees/coefficients, train/validation predictions and saved inference.
  Correct shapes, first-round-only effects or stale gradients fail.
- Smooth objectives emit unweighted per-row g/h; statistics apply weight once. Name approximate
  curvature separately. Direction-fitting for distributions/Formula separates direction from fit weights.
- Scalar Newton: v=-G/(H+lambda); gain is child sums minus parent of0.5*G²/(H+lambda), minus split penalty.
  No implicit row-count scaling of lambda/penalty. Zero effective mass cannot split; invalid denominators fail.
  Default fixtures: lambda1, eta.1, depth2, two rounds, all rows/features, no early stopping; special cases declare deviations.
- Gain ties follow(feature_id, candidate_id, missing_direction), with fixed CPU dtype/reduction order.
  Three policies have independent topology checks, not identical-tree requirements. Symmetric chooses
  common conditions by summed active-node gains; invalid candidates cannot win via negative/NaN sentinels.
- Rejection preserves accepted raw/trees/coefficients/best. Errors identify components/fields.
  RNG follows stable run ID/seed/round/purpose, not Python hash() or scheduler position.
- Use E1 tolerances, E3 quality, E4 costs, E5 author trials, E6 installs. Real tasks record all splits;
  selection reads validation only and retains missing/failing cases.

### Implementation, devices and evidence

All CPU cells are required. CUDA has explicit subsets and no silent fallback. Future runners index
reference groups, real cases and artifacts by these IDs.

| Case | Recipe | Change/reference group | CPU phase | Required CUDA | Gates |
|---|---|---|---|---|---|
| A1 | R1 | Scalar regression/scalar, tree | F1.3/F1.6 | Numeric+missing | E0/E1/E2/E3/E4/E6 |
| A2 | R1 | Binary/category/classification, data | F1.5/F1.6 | Numeric encoding; native categories optional | E0/E1/E3/E4/E6 |
| A3 | R1 | Multiclass K/classification | F1.6 | Numeric softmax | E0/E1/E3/E4/E6 |
| A4 | R3 | Query/pair/lambda/ranking | F1.5/F1.6 | None; CPU required | E0/E1/E2/E3/E6 |
| A5 | R2 | Routed quantile/quantile | F1.5/F1.6 | None; CPU required | E0/E1/E2/E3/E6 |
| A6 | R8 | Shared vector topology/vector | F1.6 | Numeric squared error | E0/E1/E2/E3/E4/E6 |
| A7 | R4 | Count/offset/positive | F1.5/F1.6 | Numeric Poisson offset | E0/E1/E3/E4/E6 |
| A8 | R4 | Positive mean/positive | F1.5/F1.6 | None; CPU required | E0/E1/E3/E6 |
| A9 | R4 | Tweedie/composition/positive | F1.5/F1.6 | Poisson component per A7; whole workflow optional | E0/E1/E3/E6 |
| A10 | R5 | Censored likelihood/aft | F1.5/F1.6 | Numeric events/right censoring | E0/E1/E3/E4/E6 |
| A11 | R6 | Geometry/acceptance/normal, state | F1.3/F1.6 | Numeric Normal, both step policies | E0/E1/E2/E3/E4/E6 |
| A12 | R7 | Coupled formula/formula | F1.4/F1.6 | None; mixed execution separate | E0/E1/E2/E3/E6 |
| A13 | R9 | Isolated runs/shared data/runs | F1.4/F1.6 | One compatible R1/R4/R8 group | E0/E1/E2/E3/E4/E6 |

Complete every required R1–R9 row. E7 external adoption is separate; internal success is not adoption,
and mathematical/real-task checks do not wait for outreach.

## 2. A1–A13 task cards

### A1: Continuous regression

- **I/O:** X, real y, weights; real raw/predictions. L=(F-y)²/2, g=F-y, h=1; training weighted-mean base.
- **Data:** California Housing retains [historical archive/array hashes](../benchmarks/foundation/housing.json)
  and units. Seeds0–2 existed; F0.3 adds3–4 using the same rule. Three splits are not five; random
  splits do not establish geographic extrapolation. Sprint 013 supplies [five splits](../benchmarks/v1/datasets/housing.json)
  with old hashes matching; licensing, budgets/real quality remain pending.
- **Algorithm:** scalar stats, three growth policies, Newton leaves, predict/add; baseline uses no
  sampling, with separate row/column sampling seed checks. All policies compose ordinary Python.
- **Oracle:** weighted base, two-round G/H, best splits/leaves, missing routing, zero weights, ties/lambda
  scaling. Constant targets cannot create NaN. Primary RMSE; stale next-round gradients fail even with valid shape.
- **Controls:** XGBoost `reg:squarederror`, LightGBM regression, CatBoost RMSE, with reasonable per-method
  growth/leaf budgets and full preprocessing/inference costs.

### A2: Binary classification and categories

- **I/O:** two original labels, X, weights, saved label order. p=sigmoid(F), L=logaddexp(0, F)-yF,
  g=p-y, h=p(1-p). Raw/probability/label are distinct. Reject single-class training; declare extreme-base clipping.
- **Data:** UCI Adult, unchanged official test; five seeds 0–4 stratified80/20 within official train.
  ? is missing; strip test-label suffix dot. Exclude fnlwgt and do not infer weights from it.
  Main real comparison has weight1; nonunit/class weights use independent fixtures.
  Sprint 014 supplies [raw data/five splits](../benchmarks/v1/datasets/adult.json); encoding/capabilities/quality remain pending.
- **Algorithm:** CPU native categories and numeric encoding; GPU encoding is train-fitted with
  dimensions, unknown semantics and costs recorded. Opponents may use native categories.
- **Oracle:** relabeling preserves probability meaning/model mappings; extreme logits stay finite;
  no test-label encoding of missing/unknown; weight enters loss and leaf stats once. Primary log-loss,
  auxiliary AUC/group confusion matrices. Sample/class weights do not guarantee calibrated probabilities.
- **Controls:** XGBoost `binary:logistic`, LightGBM binary, CatBoost Logloss. Record missing/category
  processing and effective class weights in every artifact.

### A3: Multiclass

- **I/O:** class map, raw[N, K], stable log-softmax, probabilities/original labels. g=p-onehot(y),
  exact H=diag(p)-ppᵀ; initial tree bound h_k=2*p_k*(1-p_k), explicitly not exact Hessian. Weight once.
- **Data:** UCI Covertype, all original numeric/indicator columns, labels0..K-1, seeds 0–4 stratified
 60/20/20. Do not silently compress soil/wilderness indicators into integer categories.
- **Algorithm:** all K directions from one raw snapshot, initially independent trees/joint commit.
  K is output-parameter axis, not M runs; complete softmax updates feed the next round.
- **Oracle:** hand K=3, gradient rows sum0, probabilities sum1, exact matrix/bound separately checked,
  class permutation reversible. Missing classes/invalid labels cannot be silently truncated.
- **Controls/quality:** XGBoost `multi:softprob`, LightGBM multiclass, CatBoost MultiClass; primary
  multi-logloss and separate class metrics. Include Py-Boost for GPU/output-axis comparison.

### A4: Group ranking

- **I/O:** X, ordinal relevance, query boundaries, optional query/pair weights; within-query scores.
  No generic row→pair-weight rule. Reject row weights or require explicit caller construction, never ignore them.
- **Data:** official five-fold MSLR-WEB10K train/vali/test. Missing feature tokens mean0 in dense
  parsing, not missing. qid is not a feature; retain every real query.
- **Algorithm:** oracle enumerates same-query rel_i>rel_j pairs. L_ij=logaddexp(0,-(s_i-s_j));
  opposite row gradients, diagonal curvature sigmoid(d)*sigmoid(-d). Lambda multiplies frozen
  abs(delta NDCG@10) from current ordering, without differentiating that weight or calling it ordinary
  NDCG gradient. Initial scores0; mean pair loss per query then weighted query sum; no-pair queries
  contribute zero. Sampling estimates/normalization are declared; changing pair count cannot silently change regularization.
- **Scale:** production may use bounded seeded per-query pair sampling; small fixtures enumerate
  all pairs. Sampling count/normalization are search config and cost. gain=2^rel-1, discount=1/log2(rank+1),
  stable row-ID ties, IDCG0→NDCG1 with separately reported count, not removed queries.
- **Oracle/controls:** no cross-query pairs, query score-shift invariance, sum pair gradients0,
  two-round pair/lambda recomputation. Compare NDCG@10 with XGBoost `rank:pairwise`/`rank:ndcg`,
  LightGBM lambdarank/rank_xendcg, CatBoost PairLogit/YetiRank. Use each opponent's weight semantics;
  different optimizers are not numerical parity failures.

### A5: Weighted quantile regression

- **I/O:** real y, weights, q∈{0.1,0.5,0.9}; conditional quantile per q. r=y-F, pinball=max(q*r,(q-1)*r).
  Pseudo split g=1[y<F]-q, h=1; leaves access routed residual/weight, not Newton means.
- **Data:** UCI Bike hour.csv, only forecast-known calendar predictors. Exclude instant, casual,
  registered and measured weather/temperature/humidity/windspeed. dteday is split-only. Full-date
  rolling origins train50/55/60/65/70%, then10% validation and 10% test. Sprint 012 freezes
  [files/arrays/windows](../benchmarks/v1/datasets/bike.json) with floor(D*p/100) endpoints;
  budgets, baselines/quality remain pending.
- **Leaves:** smallest residual where cumulative positive weight>=q*sum(w); filter zero weights,
  use left ties. Add eta*leaf before new residuals. Initialize the same training-only weighted quantile.
- **Oracle:** [0,2,10], weights[1,3,1], q=.5 gives 2; weights can change optimum. At nonsmooth points,
  check subgradient optimality, not second finite differences. Report each q's pinball and crossing
  rate; independent models do not guarantee no crossing.
- **Controls:** XGBoost `reg:quantileerror`(current smooth objective), LightGBM quantile, CatBoost
  Quantile/MultiQuantile. Align q/weights/output space; do not require smooth/new algorithms to match discrete oracle trees.

### A6: Multioutput and vector leaves

- **I/O:** X, Y[N, K], row weights, K=2 initially; raw/predict[N, K]. Train-only per-target mean/std,
  saved inverse; leaf output schema separate from topology. Standardized base0; constant std1 plus flag.
- **Data:** UCI Parkinsons Telemonitoring motor/total UPDRS, both excluded from X; subject ID is
  group-split-only. Subject-level60/20/20, seeds 0–4. Preserve score/interpolation semantics; no clinical-validity claim.
- **Algorithm:** both independent trees and shared vector topology required. Sum output gains;
  leaves -G_k/(H_k+lambda). Also replaceable split-statistic projection, retaining full K leaf fields.
- **Oracle:** hand two-output split/leaves, K=1→A1, output permutation, projection cannot remove leaf
  dimensions. Compare all targets over two rounds; one scalar channel or fabricated shared topology is insufficient.
- **Controls/quality:** XGBoost independent/experimental multi_output_tree, CatBoost MultiRMSE,
  LightGBM target loop, Py-Boost native multioutput. Per-target original-unit RMSE and standardized
  average; E3 checks every target, not averages hiding failures. Retain both OpenBoost structures' results.

### A7: Counts and exposure

- **I/O:** nonnegative integer count, e>0, independent sample weight. mu=e*exp(F); count mean or
  unit-exposure rate; prediction declares e. L=mu-y*(F+log(e))+lgamma(y+1), g=mu-y, h=mu.
- **Data:** freMTPL2freq/OpenML41214; IDpol split-only, ClaimNb target. Do not copy example clipping
  of ClaimNb/Exposure. Invalid e/weight/target are recorded and rejected or preregistered exclusions,
  never silently changed during fit. Entity seeds 0–4.
- **Algorithm:** rate base=log(sum(w*y)/sum(w*e)); all-zero counts need explicit minimum-rate policy.
  Offset is raw addition, not weight, consistent across train/validation/inference.
- **Oracle:** fixed F and doubled e doubles count, not rate; integer weights equal replication;
  zero weight contributes no stats; saved model retains offset contract.
- **Controls/quality:** Poisson GLM; XGBoost `count:poisson` with base_margin=base+log(e), LightGBM
  poisson with explicit init-score/prediction adapter, CatBoost Poisson/baseline adapter. Smoke
  base/offset persistence/inference first; init_score is not automatically saved. Primary count
  Poisson deviance, auxiliary rate/aggregate bias.

### A8: Positive amounts and severity

- **I/O:** y>0, weights, mu=exp(F)>0; Gamma mean with fixed shape1, L=y/mu+log(mu), g=1-y/mu, h=y/mu.
  No fitted full-distribution claim.
- **Data:** freMTPL2sev/OpenML41215 joined to frequency via IDpol for covariates. Positive paid
  claims, weight1, policy-group splits. Report nonpositive/orphan exclusions; do not mistake claims
  for policy means. Share split IDs with A7/A9.
- **Algorithm:** base=log(weighted mean y), update log mean. Float64 references cover extreme y/mu;
  no unrecorded clipping of production overflow. Errors locate offending component/input.
- **Oracle/quality:** independent derivatives, reject zero/negative y, two-round weighted leaves/
  saved predictions, declared amount units; primary Gamma deviance.
- **Controls:** Gamma GLM, XGBoost `reg:gamma`, LightGBM gamma. CatBoost's reviewed public list has no
  Gamma; custom objectives are separate extension controls, not an invented built-in.

### A9: Aggregate loss and pure premium

- **I/O:** period positive-payment total c, exposure e, annualized y=c/e. Main path weight=e,
  mu=exp(F) annualized mean, period prediction=e*mu. No additional log(e) offset. Extra business
  weights require explicit products applied once.
- **Data:** frequency left-join severity, positive payments aggregated by IDpol. Zero only when
  ClaimNb0 and no payments. Positive-count/no-payment and zero-count/positive-payment contradictions
  are separate exclusions, not guessed missing=zeros. Target is positive paid loss, not net refunds.
  F0.3 records join quality/counts.
- **Algorithm:** Tweedie p fixed per fit(default 1.5, search F0.3).
  L=-y*mu^(1-p)/(1-p)+mu^(2-p)/(2-p), g=mu^(2-p)-y*mu^(1-p),
  h=(2-p)*mu^(2-p)+(p-1)*y*mu^(1-p).
  Also frequency×severity: Poisson counts **positive-payment records**, Gamma uses matching positive
  payment means. Raw ClaimNb including zero payments cannot be multiplied by positive severity as an equivalent target.
- **Oracle:** zero loss valid; e enters weights/unit conversion, not offset again. Hand joins/
  aggregates/products, entity-order invariance, saved two-model dependencies/predictions. Both
  Tweedie and composition need real results.
- **Controls/quality:** Tweedie GLM, XGBoost `reg:tweedie`, LightGBM tweedie, CatBoost Tweedie and
  matching paid-count two-stage baselines. Primary exposure-weighted annualized Tweedie deviance,
  auxiliary totals; means do not establish calibrated tails.

### A10: Censored AFT

- **I/O:** canonical(lower, upper), initial events/right censoring. Log-normal log T=F+sigma*Z,
  Z~Normal(0,1), sigma default 1 fixed per fit. Raw log-time location; exp(F) median,
  exp(F+sigma²/2) mean, plus survival/quantiles.
- **Data:** scikit-survival Veterans' Administration lung cancer, official Status/Survival_in_days;
  event/right-censor stratification, seeds 0–4. Small real quality task, not scaling evidence.
  F0.3 freezes training censoring estimate, IPCW grid/support.
- **Math:** z=(log(t)-F)/sigma. Event NLL=log(t*sigma)+z²/2+log(2*pi)/2,
  g=-z/sigma, h=1/sigma². Censor NLL=-log(S_Normal(z)), g=-mills(z)/sigma,
  h=mills(z)*(mills(z)-z)/sigma²; stable log-tail oracle.
- **Oracle:** event→censor changes loss/update; retain valid+inf, reject NaN/reversed intervals.
  Explicitly reject unsupported left/interval/truncation. Two-round likelihood, units, monotone
  survival/roundtrip; right-censor times are not deaths for RMSE.
- **Controls/quality:** XGBoost `survival:aft` Normal/same sigma, CatBoost SurvivalAft
  dist=Normal; scale=sigma(CPU,+inf→-1), parametric log-normal AFT. Align density Jacobian/constants;
  primary censored NLL, auxiliary IPCW Brier/C-index. No verified same-name LightGBM built-in;
  custom-loss control may be separate.

### A11: Distributional/NaturalBoost

- **I/O:** real y, weights, raw=(mu, log_sigma), two parameters/Normal distribution.
  L=log_sigma+(y-mu)²/(2*sigma²)+log(2*pi)/2;
  ordinary g=((mu-y)/sigma²,1-(y-mu)²/sigma²), Fisher=diag(1/sigma²,2), natural=Fisher^-1*g.
  Do not casually rename Fisher a Hessian.
- **Data:** same Housing inputs/splits as A1 but distinct distribution results and fixed units.
  A1 RMSE cannot replace NLL/CRPS.
- **Algorithm:** independent parameter trees fit negative ordinary/natural directions, with explicit
  fit/training weight relation. Both fixed and bounded backtracking required. Joint parameters
  use one snapshot; D4 separately checks ordered updates. Train-weighted mean/scale, declared floor.
- **Oracle:** hand Fisher solve, two-round gradients/accepted state, no rejection residue; no duplicate
  weighting in direction then tree regression. Independent NLL/CRPS evaluator.
- **Controls/quality:** NGBoost Normal+LogScore/natural, CatBoost RMSEWithUncertainty, global Normal,
  outer-loop tree controls. Smoke variance/scale/raw conversion before comparison. Primary NLL,
  auxiliary CRPS, coverage+width, PIT; coverage alone is insufficient.

### A12: FormulaBoost and real structured tasks

- **I/O:** trees read recipe Z only; structure x=age_days/28>0, target MPa. a=softplus(u), b=softplus(v),
  f=a*(1-exp(-b*x)); outputs f, a, b. Saturating monotonicity is a **testable hypothesis**, not a source-proven physical law.
- **Data:** UCI Concrete Compressive Strength, seven material columns as Z, Age as x. Group identical
  seven-column recipes for60/20/20, seeds 0–4; Age/target excluded from tree features. Duplicate identical
  inputs stay grouped. F0.3 verifies group counts/age support.
- **Math:** stable -expm1(-b*x); J_u=sigmoid(u)*(1-exp(-b*x)),
  J_v=a*x*exp(-b*x)*sigmoid(v). Half-square g=J*(f-y), GGN=JᵀJ with explicit solve damping.
  Per-row rank<=1; full matrices do not establish identifiability. Default a0=max(weighted_mean(y_train),1e-6),
  b0=1, stable softplus inverse; fixtures may declare raw0. Initialization cannot read validation/test.
- **Algorithm/oracle:** independent ordinary/diagonal/full directions, two parameters/two rounds,
  correct line-search commits. Synthetic repeated Z/different x/known a, b plus single-x nonidentifiability
  and misspecification. Real data evaluates predictions/constraints, not true-parameter recovery.
  Without real results A12 is incomplete.
- **Controls/quality:** same global nonlinear formula, fixed-revision old Formula, outer coupled updates
  with XGBoost/LightGBM/Py-Boost learners, and three-library ordinary regression with Z+x. Diagonal-h
  interfaces do not prevent external full-direction solves. Primary RMSE, auxiliary in/out-of-support
  errors/parameter stability. Do not reselect formulas from test; package inference dependencies explicitly.

### A13: Train-many and model selection

- **I/O:** prepared identity, stable run IDs, per-run recipe/config/seed/round budget, validation
  metrics. Return all states/models/costs and validation-selected model, not an average table alone.
- **Data:** Covertype primary, Housing secondary; reuse A3/A1 splits. Share matching folds/bin settings
  only; changed binning weights/mappings or folds invalidate reuse.
- **Algorithm:** M=1/8/32, sequential reference and compatible batching, independent early stop/best/
  seed/failure. CPU checks heterogeneous K/objectives; initial GPU group is compatible R1, not forced
  heterogeneous fusion. Count all model-selection costs; 32-config E4 sets are not E3's16-trial budget.
- **Oracle:** independent versus sequential reuse versus batch by run ID; reorder/regroup invariance.
  Inject failure without affecting others and retain it; ignoring failures cannot pass a required
  ensemble. Selection reads validation only; committed/saved best states agree.
- **Controls/quality:** three-library loops with actual prepared reuse; Py-Boost GPU loop. Repeated
  opponent binning versus our cache cannot be the sole speed claim. Selected model passes original
  E3, all sets E4. Old ConfigBatch is semantic reference only.

## 3. Alternatives audit: Hooks, devices and gaps

Reviewed versions:[release review](boosting-release-review-2026-09-05.md), XGBoost 3.4.1, CatBoost 1.2.10,
LightGBM 4.7.0. **Documentation/code audit only**; all new-version runtime smoke is not_run, no CPU/CUDA
pass inferred. F0.3 pins NGBoost, Py-Boost, GLM/AFT dependencies. No valid Py-Boost latest release page
was obtained; do not invent a tag.

| Path | Reviewed entry | Task/cost boundary |
|---|---|---|
| XGBoost | Built-in/custom objectives, grow policies, vector trees, outer rounds/source edits | Many A1–A10 tasks; 3.4 vector hist experimental. Hessian limits do not forbid outer preconditioning; count C++ build/debug cost |
| LightGBM | Objectives, Booster.update(fobj), rollback_one_iter, set_leaf_output | D1/D3/D4 may compose public hooks; replace leaves before next gradient. Per-candidate extra stats need source audit, total min_child_weight is insufficient |
| CatBoost | Native categories, symmetric/depthwise/lossguide, custom losses, pair/group targets | Strong A2/A4/A6/A11 controls; MultiRMSE GPU, MultiRMSEWithMissingValues not. SurvivalAft CPU; custom-loss GPU does not imply arbitrary algorithm GPU |
| NGBoost | Distribution/score/metric, natural gradients/training loop | Include A11/D4; geometry/line search are not inventions; measure actual policy-change cost |
| Py-Boost | Python/CuPy, callbacks, loss/metric, sampling, multioutput sketches | Direct foundation control, requires GPU. Applicable tasks must consider it; selected arm runs GPU while OpenBoost may begin CPU. Record E5 device costs; CPU unsupported is not failure |
| GLM/parametric AFT/global formula | Simple same-target statistics | Check real need for framework with matching units/links/inputs, not only weak default trees |

F0.3 smoke checks each path's nonunit weights, prediction space, base/offset, save/load, final metric
and reported backend. Adapter failures are error and require fixing, not quietly relabeled unsupported.
GPU installation errors are environmental, not algorithm support. GPU/CPU author arms have equal
wall/token budgets with device costs separate and comparable tools/source access.

Py-Boost DepthwiseTreeBuilder.build_tree uses multioutput_sketch for split G/H then original
grad/hess in calc_node_values. Separate split/leaf statistics are not unique. D2 source entry is
candidate selection in depthwise_grow_tree. D3 may use returned leaf indices, but leaf replacement
must update train/validation caches, not just exported models. This is call-path evidence, not
measured author time/runtime success. [Tree implementation](https://raw.githubusercontent.com/sb-ai-lab/Py-Boost/master/py_boost/gpu/tree.py).

Keep shipped/roadmap/RFC/request distinctions. XGBoost multioutput, LightGBM prediction/mappings
and CatBoost export tasks inform design/risk, not unshipped runtime baselines.

### Declared scope boundaries

| Not currently required | Existing alternatives | v1 handling |
|---|---|---|
| Native CSR/CSC, external memory, distributed/multi-GPU | Three libraries have varied sparse/scaling paths; LightGBM 4.7 adds GPU capabilities | Single-device semantics first; explicit budgeted dense conversion counts cost; reject unsupported native parameters |
| All categorical CTR/ordered combinations | Mature CatBoost category processing | One explicit tested method first, while retaining real category quality gate |
| Full Cox/competing risks/truncation/all AFT censoring families | XGBoost/CatBoost Cox and multiple AFT labels | R5 events/right censoring; schema distinguishes/rejects others; no universal row-independence restriction |
| All distributions, DART/GOSS, linear leaves/all constraints | Upstream and historical capabilities for future comparison | Components over catalogs; typed payload/statistic hooks, accurate rejection |
| Arbitrary Python compilation/serialization | Python interfaces alone do not promise this | Explicit bulk device boundaries, formula dependencies, unsupported execution errors |

These implement R/C boundaries without removing A1–A13. New requirements may revise scope, but
required cells cannot silently become optional after failed evaluation.

## 4. D1–D5 author modifications

Each task delivers an installed public-API extension/recipe without core/private edits; opponents
may use hooks, outer loops or source. E5 judges actual execution, not merely a custom-function call.

| ID/axis | Change and independent oracle | Strong candidate alternative/cost |
|---|---|---|
| D1 objective/control | tau=.8 expectile, r=y-F, loss=abs(tau-I[r<0])*r²; analytic g/h, weighted base, two rounds/raw roundtrip | XGBoost/CatBoost listed expectile, LightGBM custom loss, Py-Boost loss. Built-ins allowed, no assumed OpenBoost win |
| D2 split/statistics | Each child has every declared cohort information mass>=1; extra sums separate from weights, best feasible ordinary gain, no split if none | Py-Boost build_tree→depthwise_grow_tree or XGBoost/LightGBM source. Total min_child_weight insufficient; any mathematically equivalent solution allowed |
| D3 leaf solver | Weighted pinball+lambda*(v-anchor)²/2, lambda>0; public routed residual/weight; new leaves affect next predictions | LightGBM routing/set_leaf_output, Py-Boost callbacks/source; do not hide existing hooks |
| D4 acceptance | A11 joint→declared ordered updates; at most6 alphas .1*.5^j; finite strict decrease; no residue on rejection, next reads accepted | NGBoost loop, outer LightGBM/XGBoost learners, Py-Boost callbacks; include existing line search |
| D5 scheduling | Shared prepared, heterogeneous K=1/2, independent budget/stop/RNG, reorder invariant, isolated failures retained | Independent loops+safe reuse, Py-Boost GPU loop; no assumed fusion requirement; record prepare/execute/recovery |

Minimum counterexamples:

- D1: positive/negative/zero residuals, weight0; nonsmooth points do not use ordinary second differences;
  tau=.5 reduces to symmetric squared loss.
- D2: six fixed-bin rows, alternating A/B cohorts, g=[-6,1,1,1,1,2], h=1. Highest total gain can violate
  cohorts. Enumerate best feasible candidate, then all-A-left/all-B-right data for no feasible split.
  Cohort is not a feature; do not leak it to evade the constraint.
- D3: enumerate breakpoints/interior stationary points for unique optimum; subgradient contains 0;
  increasing lambda moves toward anchor. Ordinary weighted quantile is not the penalized solution.
- D4: descent, reverse-direction full rejection, NaN, next parameter reads new state only after success;
  verify tree/coefficient/raw/best/RNG semantics, not final loss alone.
- D5: reordering, retry, different stopping, same seed/different IDs, changed data identity.
  Same shapes or skipped failures do not pass.

H1/H2 contents stay outside interface-design material. F0.3 evaluation-side freezes cards/verifiers/
hashes; F1 designers see D1–D5 only. Two empty IDs do not make E5 ready. If held-outs guide redesign,
reclassify as development, replace them and record the change.

## 5. Minimal interface sketch and C1–C7 mapping

Names remain sketches chosen for the actual callers above.

```python
prepared = prepare(training_rows, feature_schema, binning, device=device)
problem = bind(prepared, target=target, weight=weight, offset=offset,
               query=query, structure_input=structure_input)
with runtime.run(run_id=run_id, seed=seed, device=device) as run:
    accepted = initialize(problem)
    for step in range(rounds):
        geometry = objective.geometry(problem, accepted)
        direction = direction_rule(geometry)
        # grow is ordinary Python: rewrite the composition or replace one function.
        tree = grow(prepared, direction, aggregate=aggregate,
                    candidates=candidates, score=score, feasible=feasible,
                    partition=partition, leaf_solver=leaf_solver,
                    policy=growth_policy, run=run)
        proposal = propose(accepted, tree, output_mapping=output_mapping)
        accepted = accept_or_reject(problem, accepted, proposal, run)
    model = export_model(accepted, output_schema=output_schema)
```

Expose aggregate→candidate statistics→feasibility/score→choose→partition→leaf solve. Candidates
contain thresholds/category sets/missing routes. Leaves get read-only indices/residuals/stats,
not G/H only. Algorithms may own loops; no universal trainer or registry-per-function requirement.

| Capability | Actual callers | Delivery/acceptance |
|---|---|---|
| C1 typed data/targets | A2 categories, A4 queries, A6 vectors, A7 offset, A10 intervals, A12 structure | F0.2 data, F1.1/F1.5, all identity/shape/invalid inputs |
| C2 composable trees | A1 three policies, A4 row reduction, D2 extra stats | F0.2 tree, F1.2, exhaustive splits/routes/ties/feasibility |
| C3 learners/leaves/outputs | A5/D3 residuals, A6 vectors, A11/A12 mapping | F0.2 quantile/vector, F1.2/F1.6; K scalar runs are not vector leaves |
| C4 state/runtime | A11/D4 rejection/order, A13/D5 isolation | F0.2 state/runs, F1.1, F3 visible devices/transfers/sync |
| C5 artifacts | A2 mappings, A7 offset, A9 dual models, A10 spaces, A12 formula | F1 roundtrips, F5 wheels; standard raw inference without training plugins |
| C6 eval/authoring | All A/D, H1/H2 separately frozen | F0.3 runner/judge, F2 E5, F4 E3; bad artifacts fail |
| C7 workflows | Every A install→baseline→change→verify→save/infer | F2 trials, F5 E6; missing any workflow remains incomplete |

Public raw is[N, K]; backends may use other layouts with recorded conversions. Variable topology
and scalar/vector payload replace fixed511/depth8 restrictions. Explicit proposal/accepted ownership
permits transactions/deltas without full copies each time.

## 6. Historical implementation audit

This audits code before Sprint 002 retirement; links pin50acfc6. Historical capabilities are not
current-namespace functionality.

- [Standard model](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/_models/_boosting.py)
  applies CPU sample_weight; standard CUDA rejects supplied weights. Multiclass lacks the same
  weight entry and fits per K. [Old multiclass docs](../docs/user-guide/models/multiclass.md)
  do not prove v1 weighted/mapping support.
- [FormulaObjective](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/_objectives.py)
  already computes coupled GGN before per-parameter fitting;[model](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/_models/_formula.py)
  accepts separate model_input. [Tests](../tests/test_formula.py) are mainly synthetic. Do not
  call old coupling new or full GGN automatic identifiability. A12 now has a real task definition.
- [ConfigBatch](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/_batch.py)/
  [tests](../tests/test_batch.py) already recompute loss per configuration across rounds; independent
  stop/error/RNG and compatible batching need new evidence.
- [Survival](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/_models/_survival.py)/
  [tests](../tests/test_survival.py) use Weibull event/time. A10 log-normal is a new independent
  recipe; old class names do not prove new noise/censor semantics.
- [Experimental Booster](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/experimental/_booster.py)
  limits CUDA eval/callback/early stopping. Validate v1 per case without inheriting limitations or
  compatibility obligations; preserve mathematical failures.

## 7. Sources and F0.1 boundary

Task mathematics is explicitly defined here and independently derived/verified in F0.2. Sources
check interfaces/devices/data, not replace installed F0.3 capability smoke.

- XGBoost [parameters](https://xgboost.readthedocs.io/en/stable/parameter.html),
  [advanced custom objectives](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html),
  [GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html): task/geometry/device boundaries.
- LightGBM [parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html),
  [Booster](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html): objectives, rounds, leaves.
- CatBoost [regression](https://catboost.ai/docs/en/concepts/loss-functions-regression),
  [ranking](https://catboost.ai/docs/en/concepts/loss-functions-ranking),
  [multioutput](https://catboost.ai/docs/en/concepts/loss-functions-multiregression): targets/weights/devices.
- [NGBoost development](https://stanfordmlgroup.github.io/ngboost/5-dev.html),
  [Py-Boost](https://github.com/sb-ai-lab/Py-Boost): geometry/scores/Python GPU controls.
- A1–A9 sources:[application matrix](foundation-application-contracts.md),
  [insurance processing reference](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html).
- A10 [official loader/fields](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.datasets.load_veterans_lung_cancer.html),
  [evaluation](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html).
- A12 [UCI Concrete, data/units/CC BY4.0](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).
  Actual hashes/group counts still require downloads; the formula is not supplied by that page.

**F0.1 acceptance:**all A1–A13 inputs/outputs, data/splits, algorithms, oracles/controls; full R/C mapping,
falsifiable D1–D5, explicit devices/sketches. **References delivered:**F0.2
[exit audit](../v1-sprints/f0-2-acceptance-ledger.md). **Still incomplete:**F0.3 downloads/hashes/
capabilities/budgets/held-outs/judge. This file does not pass F0 or any E-gate. Freeze comparison
protocol next; no sole production oracle or premature trainer/kernel work.
