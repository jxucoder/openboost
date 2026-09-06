# XGBoost / CatBoost / LightGBM: Release and plan review before v1 design

Review date: 2026-09-05. This is a snapshot of public primary sources; these new versions
were not installed or run. Formal comparison must pin versions, wheel/build hashes, CUDA
variants and effective configuration. Old search snippets, latest development docs and
unmerged PRs do not establish shipped capabilities.

## 1. Latest stable releases

| Project | releases/latest on review date | Release record and recent changes | Implication for OpenBoost v1 |
|---|---|---|---|
| XGBoost | **3.4.1**, tag v3.4.1,6fe8c54 | Fixes categorical-container model slicing and JVM sparse batch prediction; notes dated2026-08-14, GitHub shows Aug15 08: 30 | Category encoding, slicing and saved-model inference are correctness obligations; retain both dates without equating documentation and upload times |
| CatBoost | **1.2.10**, tag v1.2.10, b1bd2a6,2026-02-19 | JVM transposed prediction, Spark4.0/4.1; adjacent1.2.9 contains major recent Python/data-interface changes | Last-patch-only reviews miss input/inference capabilities; compare complete installed versions |
| LightGBM | **4.7.0**, tag v4.7.0,8f7036f, GitHub shows Jul18 20: 20 | Polars/Arrow, ROCm/HIP, NCCL multi-GPU, CUDA13 builds, distributed fixes; organization lightgbm-org, default branch main | Do not claim LightGBM lacks CUDA/multi-GPU; record CPU/OpenCL/CUDA/HIP separately |

Sources: [XGBoost 3.4.1](https://github.com/dmlc/xgboost/releases/tag/v3.4.1),
[CatBoost 1.2.10](https://github.com/catboost/catboost/releases/tag/v1.2.10),
[LightGBM 4.7.0](https://github.com/lightgbm-org/LightGBM/releases/tag/v4.7.0).
LightGBM retains the displayed month/day; no exact UTC timestamp is inferred from an unavailable API.

### Structural XGBoost 3.4 changes

Upstream calls3.4.0 hist vector leaves feature-complete while still experimental, covering
categories, constraints, DART, distributed execution, model inspection and batch statistics.
MAE/quantile leaf estimation changes to smooth approximations. Default binaries use CUDA13.3;
xgboost-cu12 uses CUDA12.9. [Official3.4 notes](https://xgboost.readthedocs.io/en/stable/changes/v3.4.0.html).

Design inference: multioutput/vector leaves are not an exclusive differentiator. Separate split
statistics from leaf-fitting statistics and allow leaf algorithms to change. Old quantile math
cannot serve as the latest-version oracle. Preflight coexistence of old T4 environments and
new wheels; never silently downgrade competitors or fall back to CPU.

### Do not underestimate existing CatBoost capabilities

1.2.9 adds Polars inputs with auxiliary fields, RMSPE, mmap model loading, Python 3.14 support,
and Lossguide/data-initialization improvements. These are shipped notes, not locally reproduced
speed claims. [1.2.9 release](https://github.com/catboost/catboost/releases/tag/v1.2.9).

GPU custom objectives/metrics appear in1.2.6. Python custom GPU losses are not unique to OpenBoost;
verify task-specific scope and calling conventions. [1.2.6](https://github.com/catboost/catboost/releases/tag/v1.2.6).

Official objectives include Poisson, Tweedie, MultiQuantile, RMSEWithUncertainty, Cox and SurvivalAft,
with per-objective device boundaries; Cox/SurvivalAft are CPU. AFT upper-bound sentinels differ
from XGBoost and require semantic adaptation. [Objectives/devices](https://catboost.ai/docs/en/concepts/loss-functions-regression).
Ordered boosting/category statistics are established techniques to review, not a full v1 replication
target. [Official references](https://catboost.ai/docs/en/concepts/educational-materials-papers).

### LightGBM task breadth and engineering

Public parameters cover classification, multiclass, ranking, Poisson/Gamma/Tweedie, quantiles,
constraints and custom objectives. Public interfaces support per-round updates and leaf-output
modification. [Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html),
[Booster](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html).
Compare corresponding objectives and full preprocessing/inference, not one regression fit timer.
The4.7 weighted-percentile fix reinforces the need for independent nonunit-weight leaf references.

## 2. Public plans by evidence level

Do not imply all libraries have a unified roadmap with committed delivery dates. Distinguish
formal trackers, explicit author intent, proposals and community requests. States reflect the
review date and do not guarantee scheduling.

| Project/source | Observed status/content | Use for v1 |
|---|---|---|
| XGBoost [multioutput roadmap#9043](https://github.com/dmlc/xgboost/issues/9043) | Open, type: roadmap; body updated for3.4 hist completeness; discusses multitask, broader outputs/execution interfaces | Verify completed items in releases; old checklists are not all missing features. Separate model/output/statistic axes |
| XGBoost [default-parameter RFC#12131](https://github.com/dmlc/xgboost/issues/12131) | Open; 2026-03-26 proposes learning-rate, sampling, budget changes | Proposals are not defaults; record effective defaults and fair tuned configurations |
| LightGBM [#2302](https://github.com/lightgbm-org/LightGBM/issues/2302) | Official feature-request/voting hub, no fixed delivery commitment; linked items vary in status | Suggests loading, routing, ranking, memory pain points; votes are not adoption or roadmap commitments |
| LightGBM [prediction efficiency#7326](https://github.com/lightgbm-org/LightGBM/issues/7326) | Open, assigned author explicitly plans Python predict/import cost research | Measure process startup, import, load, single/batch prediction, not only warm fit |
| LightGBM [category encoding#7361](https://github.com/lightgbm-org/LightGBM/issues/7361) | Open proposal for container-independent mappings/cross-language state; no milestone | Category semantics, persistence and unseen behavior must not depend on pandas internals |
| CatBoost [exported-model CI#3173](https://github.com/catboost/catboost/issues/3173) | Open Task, 2026-08-23; improve exported-code tests; links#3174 | Visible engineering task, not a whole roadmap; verify inference artifacts in independent environments |
| CatBoost [C++ export compatibility#3172](https://github.com/catboost/catboost/issues/3172) | Open, 2026-08-23; language-standard issue for categorical export, no release commitment | Deployment is real usage; OpenBoost still need not preserve old APIs/formats |

No current unified, time-committed CatBoost roadmap was found after reviewing releases, tasks,
contribution material and milestones. Long-lived planned/good-first-issue labels do not mean
imminent release. LightGBM's request hub likewise does not schedule every request. GitHub API
and some filtered pages failed; conclusions use successfully read releases/specific issues,
not an exhaustive review of comments, PRs or private plans.

## 3. Decisions carried into v1

1. **Target verifiable algorithm changes.** Classification/regression, probabilistic models,
   GPU custom losses and vector leaves already have alternatives; include them as controls.
2. **Evaluate algorithms and applications separately.** All A1–A13 are required; do not select
   only insurance/AFT or a few representative cases.
3. **At least five modification tasks:** objective/geometry, split/growth, leaf solver, update
   control, run scheduling, including an existing-library-friendly control and unseen tasks.
4. **Include actual engineering cost.** Categories/missingness, weights, offsets, groups, persistence,
   cold startup and inference need independent checks. Library GPU support is not per-objective support.
5. **Freeze competitors/protocol versions.** Capability smoke first; record package/version/hash,
   backend/driver, effective parameters and inference semantics. New releases require a new experiment version.

This review supports design/baseline selection, not OpenBoost speed, quality or adoption superiority.
