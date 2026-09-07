# v1 evaluation preparation

Sprint 011 adds an **artifact integrity judge**, not the full F0.3 runner or E0–E7
judge. No real dataset manifest, installed baseline capability result, frozen
resource budget or held-out author task is delivered by this slice.

Run from a repository checkout with the project environment:

```bash
uv run --no-sync python -m benchmarks.v1.judge /absolute/path/to/run
```

The command reads `manifest.json` and `cases.jsonl`, prints a JSON report to stdout,
and exits 0 only for `integrity_pass: true`; otherwise it exits 1. It does not train,
execute artifact code or modify the run directory. `gate_results` is always empty.
A producer's `pass` is a status claim, never an independently verified quality score.

## Integrity schema: `openboost-integrity-v0`

Objects use exactly the documented fields; unknown versions/fields are rejected.
JSON duplicate keys, non-finite numbers and blank JSONL records are rejected.

Manifest fields:

- `schema`: the version above; `protocol_sha256`: lowercase SHA-256 of the declared protocol.
- `provenance`: `code_sha` (40 lowercase hex), `dirty` (must be false in this version),
  `environment` (nonempty JSON object). Dirty code needs a patch digest in a future
  version; a boolean alone cannot identify different uncommitted implementations.
- `expected`: nonempty list of cells. Each has `id`, `application` (`A1`–`A13`),
  `required` (boolean), `backend` (`cpu`/`cuda`), `model`, `fold` (nonempty strings),
  `seed` (nonnegative integer), `dataset_sha256`, `split_sha256`,
  `preprocessing_sha256`, and `config` (JSON object).

IDs must be unique; every A-ID needs a required CPU cell. This only checks the
**declared matrix**. The future frozen protocol must independently define actual
fold counts, methods, datasets, budget and device requirements; one cell per A-ID
is not a sufficient experimental design. This schema does not yet validate
package/wheel metadata, real data hashes, license, row IDs, target units, resource
limits or the authenticity of recorded provenance.

Each JSONL case has exactly:

- `id`, `status` (`not_run/pass/fail/unsupported/error/timeout`), `cache_key`;
- `backend` (`cpu/cuda/none`), `fallback` (boolean), `exit_code` (integer or null);
- `artifacts`, `metrics` (finite numeric values by metric name), `reason` (string).

`cache_key(manifest, cell)` hashes the canonical JSON of the **entire manifest and
cell**. Code, environment, protocol, matrix, data, split, preprocessing, config and
seed changes invalidate old keys. This conservative version has no cross-matrix
cache reuse. It cannot detect an input that a producer failed to declare.

All cells, including optional cells, must have a record and a hashed `log` artifact.
Required non-pass records fail integrity. Optional non-pass records remain visible
with a nonempty reason; they cannot establish GPU or quality success. A pass record
requires exit code 0, the expected backend without fallback, nonempty metrics, and
hashed `predictions`, `model`, `log` artifacts. Artifact entries have `path` (relative
inside the run directory) and `sha256`; path escapes and symlinks outside are rejected.
Model and log bytes are hash-checked but not interpreted. Prediction bytes must be
JSON containing a nonempty rectangular finite numeric array. Matching prediction
rows/output semantics to targets and recomputing metrics belong to the forthcoming
independent evaluators. No dataset data is confused with prediction validation:
legitimate censored-label infinity remains part of the AFT target contract.

## Verification and next steps

[Adversarial tests](../../tests/v1/test_artifact_judge.py) generate clearly synthetic
bundles in temporary directories. They exercise missing/duplicate/unknown cases,
input changes with stale cache keys, false pass claims, worker failures, unsupported
GPU, missing/corrupted artifacts, invalid prediction values and the CLI exit code.
Synthetic manifests are test inputs, not committed benchmark results.

Next F0.3 slices must acquire and hash real datasets, implement their split/target
adapters, smoke-test pinned baselines, freeze resources/configurations and held-out
verifiers, add the execution runner and independent metric/gate evaluation. Full
benchmark runs remain prohibited until actual hashes and budgets are frozen.

## A5 real data preparation

[Sprint 012](../../v1-sprints/012-bike-data-freeze.md) freezes the UCI Bike Sharing
hourly archive and five full-date rolling splits in [datasets/bike.json](datasets/bike.json).
Source: Hadi Fanaee-T (2013), [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
DOI 10.24432/C5W894, CC BY 4.0. The download has **17,379** hourly rows, versus
17,389 on the UCI page checked on 2026-09-06; archive and CSV hashes identify the
actual data used. Raw data is downloaded locally and is not vendored in Git.

```bash
curl --fail --location 'https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip' -o /tmp/openboost-v1-bike.zip
uv run --no-sync python -m benchmarks.v1.bike /tmp/openboost-v1-bike.zip --verify benchmarks/v1/datasets/bike.json
```

`load_archive` verifies both pinned archive and `hour.csv` hashes before parsing.
`parse_hour` exposes X (float64 calendar columns), y (hourly total count), original
integer row IDs, and ISO dates. No fitted preprocessing is performed. Calendar
values and chronological unique timestamps are checked. All records are retained;
missing hours are not synthesized. IDs/dates support alignment and splitting only.
Observed weather, temperature, humidity, wind and casual/registered counts never
enter X. Numeric/calendar encoding is explicit; later adapters may choose categorical
representations without learning from validation/test.

For D=731 dates, each origin uses floor(D*p/100), floor(D*(p+10)/100),
floor(D*(p+20)/100) endpoints for p=50,55,60,65,70. Each date stays whole.
Origins overlap; they are not independent trials. The unused final 10% of dates
is intentionally outside the predeclared windows, not an extra selection set.
The freeze contains per-part row counts, date boundaries and row-ID hashes, plus
feature/target array hashes and adapter source hash. `--verify` refuses altered
source or data/splits; replay provenance (revision/environment/argv) may differ.

This **data preparation record is not an integrity-v0 run manifest**. It honestly
records a dirty checkout and identifies the adapter bytes separately. It does not
freeze training budgets/configurations, run any quantile model, or pass E3/A5.

## A1/A11 Housing data preparation

[Sprint 013](../../v1-sprints/013-housing-five-splits.md) adds
[datasets/housing.json](datasets/housing.json). The local archive matches the historical
source hash. The independent v1 adapter exactly reproduces the old X/y hash and
seeds 0–2 split hashes, then adds seeds 3–4 without altering the historical record.

```bash
curl --fail --location 'https://ndownloader.figshare.com/files/5976036' -o /tmp/openboost-v1-housing.tgz
uv run --no-sync python -m benchmarks.v1.housing /tmp/openboost-v1-housing.tgz --verify benchmarks/v1/datasets/housing.json
```

There are 20,640 rows and eight features; each seed has 12,384/4,128/4,128 rows.
Source row positions define IDs. Household ratios are computed per row, and targets
are divided by 100,000 before float32 conversion. No preprocessing is learned from
the complete dataset. The hash format deliberately retains the legacy raw-byte
convention; it differs from Bike's dtype/shape-prefixed hashes and is named explicitly.
Independent tests check column mapping, ratios, units, RNG isolation and partitions.

A1 regression and A11 Normal distribution predictions share inputs and splits but
require separate scores and acceptance; this counts as one data source. Random
partitions do not establish geographic generalization. The
[scikit-learn description](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
provides dataset context. The original freeze recorded an unresolved license.
A subsequent [dated source review](datasets/housing-license-review.json) verifies
that the exact downloaded file matches Figshare's published MD5 and carries its
uploader's CC BY 4.0 declaration. Attribution: Nelson Liu (2016), *scikit-learn
california housing dataset cal_housing.tgz*,
[Figshare version 2](https://doi.org/10.6084/m9.figshare.3829992.v2). Original StatLib
rights provenance was not separately verified. The historical data/preprocessing
freeze is preserved; this review changes no bytes or folds. Model quality is pending.

## A2 Adult data preparation

[Adult freeze](datasets/adult.json) preserves official test and creates five
80/20 stratified partitions of official training. Source:
[Becker and Kohavi (1996), UCI Adult](https://archive.ics.uci.edu/dataset/2/adult),
DOI 10.24432/C5XW20, CC BY 4.0. Download and verify with:

```bash
curl --fail --location 'https://archive.ics.uci.edu/static/public/2/adult.zip' -o /tmp/openboost-v1-adult.zip
uv run --no-sync python -m benchmarks.v1.adult /tmp/openboost-v1-adult.zip --verify benchmarks/v1/datasets/adult.json
```

The adapter exposes mixed numeric/string/None records with 13 features; fnlwgt is
excluded, and weights remain unit. Test labels lose exactly one suffix dot. IDs
combine official filename and physical line number. All records remain present.
No fitted categorical encoding is included; subsequent encoding must learn only
from each training partition. Shared predictor values are audited in the artifact
but are not assumed to identify repeated people. The official test is unchanged.

## Remaining real-data preparation

`real_data.py` verifies the pinned [source catalog](datasets/sources.json) before
parsing. With the isolated evaluation environment installed:

```bash
uv venv build/v1-env --python 3.12
uv pip sync --python build/v1-env/bin/python --require-hashes benchmarks/v1/requirements-cpu.txt
build/v1-env/bin/python -m benchmarks.v1.real_data concrete build/v1-data --verify benchmarks/v1/datasets/concrete.json
```

Download each catalog URL into its named file under `build/v1-data/`. Substitute
`covertype`, `parkinsons`, `veteran`, or `insurance` in the replay command. Raw data
is not committed. Hashes include little-endian dtype, shape, and row order.

- [Covertype](datasets/covertype.json): all 581,012 rows, 54 original features,
  seven labels, five stratified splits. The indicators remain separate columns.
- [Parkinsons](datasets/parkinsons.json): 5,875 rows, 19 inputs, two UPDRS targets;
  subject IDs determine partitions and cannot enter features.
- [Concrete](datasets/concrete.json): 1,030 rows, seven material inputs; identical
  recipes stay together. Age/28 is a separate structural input and MPa is the target.
- [Veteran](datasets/veteran.json): 137 rows, six inputs; observed times and event
  indicators stay separate. Source-declared categorical orders are fixed mappings,
  not a vocabulary learned from held-out rows. IPCW support preparation remains open.
- [Insurance](datasets/insurance.json): 678,013 policies. Claims retain shared
  policy partitions. The audit records 195 orphan claims and 9,116 positive-count
  policies without a payment; 668,897 policies remain for aggregate targets.
  Categories stay separate string columns; train-fitted encoding is still pending.

UCI [Covertype](https://archive.ics.uci.edu/dataset/31/covertype),
[Parkinsons](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring), and
[Concrete](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
pages declare CC BY 4.0. OpenML metadata for
[frequency](https://www.openml.org/api/v1/json/data/41214) and
[severity](https://www.openml.org/api/v1/json/data/41215) declares CC0; downloaded
files also match their published MD5 checksums. Housing now has a matching hosted
CC BY 4.0 declaration in the dated review above. Veteran's original-source license
evidence remains unresolved; StatLib endpoints returned HTTP 403.
Microsoft's linked MSLR download/agreement could not be retrieved; A4 remains
required and unresolved. No missing case is converted into a passing result.

## Independent evaluation machinery

- `preprocessing.py` fits numeric medians/missing indicators and categorical
  one-hot vocabularies on training rows only; unknown categories use the missing
  indicator. Target standardization preserves constant outputs. Training-only
  reverse Kaplan-Meier support uses event-before-censor tie handling.
- `freeze_preprocessing.py` freezes encoded arrays for five splits, including
  separate claim-level and eligible-policy insurance encoders. Replay with
  `build/v1-env/bin/python -m benchmarks.v1.freeze_preprocessing --verify benchmarks/v1/datasets/preprocessing.json`.
- `quality.py` independently recomputes prediction-space primary metrics and
  applies the v1-plan-r2 five-fold thresholds. A6 checks every target and A5 every
  quantile. Probabilities are validated before the fixed 1e-15 log-loss clipping.
- `quality_report.py` reads hashed NPZ targets/predictions with exact row-ID
  alignment. A run directory contains `quality-manifest.json` with declared
  paired cells. `python -m benchmarks.v1.quality_report DIRECTORY` reports paired
  comparisons; it deliberately does **not** certify E3, because selection receipts
  and complete required recipe/device coverage still need integration.
- `process_runner.execute` runs an argv in a fresh directory, captures logs,
  enforces wall time and thread settings, kills timed-out process groups, and
  requires model/prediction artifacts. Container memory enforcement is separate.
  A zero process exit or an execution pass is not a quality pass.
- [Search design](search-design.json) fixes 16 configurations per listed method
  family and resource budgets before real quality runs. It remains a partial
  design until task adapters, the full matrix, and the agent cohort are bound.

None of these modules is OpenBoost production training code. Missing ranking
inputs, licenses, held-out tasks, and full runner/judge integration still prevent
F0.3 exit; existing data and metric checks cannot waive those requirements.

## Installed comparator preflight

[Capability evidence](evidence/README.md) records the installed CPU and real T4
matrix, native-build failures and corrected isolated runs. Built-in support is
scoped per task/device. It does not establish real-data quality or OpenBoost GPU
execution. The CUDA environment is hash-locked at the package level; complete
native build provenance remains a final protocol requirement.

## Numeric validation worker

`baseline_worker.py JOB.json` writes `predictions.npz` and a trusted local pickle
`model.bin` in its working directory. Jobs name application/library/configuration,
seed, threads, device and input NPZ. Arrays contain training X/y, validation X/IDs,
optional training weights, and task-specific exposure or censoring fields. Unknown
options and test arrays fail. Reloaded predictions must match before files are
written. Use a fresh process through `process_runner.execute`.

[Worker evidence](evidence/worker-cpu.json) verifies 30 synthetic CPU task/library
fits and in-process reload, including external exposure doubling in all three
count adapters. `worker_smoke.py` reproduces the checks with the locked interpreter.
The original artifact covers fixed-round fitting. Native early stopping is now
supported as described below. Ranking, composed/structural controls and A13
execution remain required integration work; selection/test release has a separate
audited layer but is not yet bound to the full real-task matrix.

## Validation selection and sealed test release

`selection.audit(protocol, records, directory, pinned_protocol_sha256)` independently
recomputes validation metrics for exactly 16 configurations per declared method.
The protocol is held and hashed by the trusted orchestrator before training; it
binds application/fold, code/data/split/preprocessing/environment/search identities,
training row IDs, validation truth, test-feature hash, methods/configurations and
selection weights. Every task primary must have a positive frozen weight. A6
weights must derive from training target scales; final quality still checks each
output separately. A4 maximizes NDCG; the remaining tasks minimize their scores.
Ties use the lexical trial ID. No test features are opened during selection.

Trial records contain exactly `id`, `config`, `status`, `exit_code`,
`protocol_sha256`, `prediction`, `model`, and `log`. Artifact descriptors contain
relative `path` and `sha256`; validation predictions retain exact row identity.
The evaluator rejects missing/duplicate/failed trials and producer-supplied scores.
It checks every model/log hash, not only the eventual winner. Receipt contents
include all recomputed metrics, selection scores, artifact descriptors and the
selected trial. Reordering records does not change the receipt.

`selection.seal` exclusively creates a receipt file and returns its byte hash;
keep that hash separately under orchestrator control. `selection.release_test`
checks this receipt and reruns the complete audit before loading hashed test
features. It rejects training/validation row overlap, test targets, invalid
encoded features and changed artifacts. It returns features and the selected
model descriptor; downstream inference must recheck the model hash before loading.
The audit does not deserialize model files or execute artifact code.

This is an evaluator access sequence, not an operating-system security boundary.
The protocol/digests must not be chosen by the producer after seeing results.
The audit cannot prove when an external process accessed files or which code it
executed; restricted worker mounts and execution provenance remain necessary.
Query/entity identity constraints belong to frozen dataset adapters, in addition
to this layer's row-disjointness checks. Neither a selection receipt nor the paired
quality report certifies complete E3 coverage.

Reproduce the actual-process synthetic integration check with the pinned baseline
environment, using a fresh output directory:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -m benchmarks.v1.selection_smoke build/v1-selection-smoke-new
```

It executes 16 small fixed-round XGBoost jobs, seals validation selection and runs
selected-model test inference in a new process. The generated packet includes raw
inputs, predictions, models, records and receipt under ignored `build/` output.
The committed [summary](evidence/selection-cpu.json) records source/environment
hashes; it is synthetic harness evidence, not a real-data quality/performance result.


## Native baseline early stopping

Set `early_stopping_rounds` to a positive integer and supply `y_validation`;
`weight_validation` defaults to unit weights. A10 additionally requires
`event_validation`. Without stopping enabled, validation target/weight fields are
rejected rather than silently used or ignored. Targets, weights and survival
indicators are validated before fitting; the test array prohibition is unchanged.

Native objective metrics determine stopping within each trial. Cross-method
configuration selection still recomputes the frozen primary metrics independently.
The native history preserves its metric names and weighted validation values:

- XGBoost retains its fitted trees and stores `best_iteration + 1` as an explicit
  prediction limit, including vector and quantile models.
- LightGBM records each fitted model's best iteration; per-output/per-quantile
  baseline fits stop independently and preserve their individual limits.
- CatBoost uses the validation pool and `use_best_model`, truncating the saved
  model to its selected tree count.
- NGBoost receives explicit validation arrays and weights, avoiding an implicit
  split. The saved bundle predicts with `best_val_loss_itr + 1`.

Count validation includes exposure offsets in all three libraries. CLI workers
write `training.json` with stopping histories and prediction limits alongside the
model and predictions. Histories describe native selection, not independent
quality acceptance. The selected limit is part of model replay semantics.

[CPU stopping evidence](evidence/early-stopping-cpu.json) covers 30 synthetic
supported task/library cells with nonunit validation weights, vector/quantile,
exposure and survival cases. Every selected count matches its history's minimum.
Overfitting counterexamples select round 1 and reproduce in fresh processes.
No GPU stopping or real-data quality result is established. Reproduce with:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -m benchmarks.v1.early_stopping_smoke
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -m benchmarks.v1.selection_smoke build/v1-selection-early-stop-new --early-stopping-rounds 3
```

The second command runs the synthetic 16-trial selection/release smoke with
stopping enabled, using a fresh output directory. Its patience of three is a
small-fixture check; the preregistered real-search patience remains 50.

## Query-aware ranking worker

A4 now requires contiguous `query_train` and `query_validation` IDs; partitions
must have disjoint query IDs of the same type. Supply one `query_weight_train`
per contiguous training group and, with stopping, one `query_weight_validation`
per validation group. Defaults are unit query weights. Generic row weights,
fragmented query blocks and invalid relevance labels fail before training.

The adapters use XGBoost rank:ndcg, LightGBM lambdarank and CatBoost PairLogit.
XGBoost receives per-group weights; LightGBM and CatBoost receive the explicit
native equivalents. Native NDCG@10 histories drive stopping, and prediction
replay preserves the selected model. Independent final scoring uses the fixed
v1 exponential-gain/unit-query NDCG convention; native weighting/gain conventions
are not asserted numerically identical. They remain visible in recorded histories.

[Ranking evidence](evidence/ranking-cpu.json) covers three CPU fit/stopping/reload
checks. Reproduce with the pinned interpreter and two numerical threads:
`build/v1-env/bin/python -m benchmarks.v1.ranking_smoke`.
The initial smoke assertion selected the last metric in CatBoost's history,
which is PairLogit rather than NDCG; its source/error record is retained. The
corrected check locates NDCG by name. Real MSLR data/agreement and CUDA execution
of this worker remain unverified; these probes do not pass real A4 quality.

## Parametric and composed validation workers

`parametric_worker.py JOB.json` runs the `glm` A7/A8/A9, `paid_composition` A9,
or `formula_global` A12 controls through the same fresh-output process runner.
The strict job has `application`, `method`, `config`, and `input_npz`; test arrays
and unsupported fields fail. It writes predictions, a trusted local model bundle
and training configuration after verifying replay. GLMs treat convergence warnings
as failures; nonlinear optimization must report successful finite convergence.

- GLM A7 consumes period counts plus exposure; A9 consumes period paid totals plus
  exposure. Both fit annualized targets with exposure times business weight once.
  A7 emits period count predictions; A9 emits annualized premium predictions.
  A8 consumes positive individual payments and emits positive payment means.
- Paid composition additionally requires `paid_count`, `claim_policy` indices and
  `claim_amount`. Their exact counts and summed positive payments must reconstruct
  policy targets. Orphans, nonpositive payments, and mismatches fail. Frequency is
  paid-record frequency, not raw ClaimNb. Severity weights inherit each policy's
  business weight once per claim. Annualized predictions multiply paid frequency
  and severity; period totals multiply exposure once.
- Global formula consumes `age_train`, `age_validation` already in days/28 and
  training MPa targets. It fits positive global amplitude/rate through softplus
  and records training age support. It is a structural comparator, not FormulaBoost
  or a claim of parameter identifiability on arbitrary real datasets.

All numeric GLM scaling fits training inputs only. [Search design](search-design.json)
now includes 16 paid-composition penalty pairs fixed before real quality runs.
Existing GLM/global-formula grids map directly to worker configuration. These
adapters still need binding to complete real-data/search manifests.

[Hand-worked evidence](evidence/parametric-cpu.json) verifies weighted means,
paid-record composition, exposure scaling and known-curve parameter recovery.
[CLI evidence](evidence/parametric-worker-cpu.json) verifies all five controls in
bounded processes and exact output units/IDs. Reproduce with the locked CPU
interpreter and two numerical threads:

```bash
build/v1-env/bin/python -m benchmarks.v1.parametric_smoke
build/v1-env/bin/python -m benchmarks.v1.parametric_worker_smoke build/v1-parametric-workers-new
```

These are synthetic correctness checks. Real A9/A12 quality, outer coupled tree
controls, support-stratified reports and full F0.3 integration remain unfinished.

## Auxiliary quality diagnostics

`auxiliary.py` adds weighted classification accuracy/Brier/per-class counts,
binary AUC with half credit for score ties, Normal PIT decile mass, survival
IPCW Brier/Harrell C, structural errors by frozen training-age support, and an
exact five-fold empirical bootstrap of paired mean differences. Absent classes,
empty structural strata and no comparable survival pairs produce explicit null
statistics. These diagnostics cannot replace primary gates.

The [Brier definition](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html)
uses the training censoring distribution. We use its frozen right-continuous G(t)
with the existing event-before-censor risk convention. Grid points must lie
strictly inside positive training support. For each grid point, only observed
deaths by that point need G at their event times; later observations use G at the
grid point. This avoids extrapolating G for longer test follow-up. It is a direct
formula implementation, not a call to scikit-survival's more restrictive API.
No-contribution grids fail rather than returning a misleading zero.

Harrell C uses unit comparable pairs, risk = negative log-time location and a
1e-8 risk-tie tolerance. An event tied with a censor is comparable; tied deaths
are not. It is explicitly **not** an IPCW C-index. See the
[concordance definition](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_censored.html).
Nonunit weights are rejected by this survival auxiliary contract.

The paired quality report now includes all measured fold metrics and descriptive
paired-difference intervals. Its optional per-cell `auxiliary` entry is:

- A10: `{"censoring": {"path": "censoring.json", "sha256": "..."}}`, pointing
  to the frozen training `censoring_support` object.
- A12: `{"structure": {"path": "structure.npz", "sha256": "..."}}`, with aligned
  `row_ids`, `age`, and scalar `train_min`/`train_max` arrays.

Missing A10/A12 auxiliary inputs are listed in `auxiliary_missing`; invalid or
corrupt inputs produce errors. Primary paired comparisons can be reported while
auxiliaries are missing, but `E3_pass` remains false. Training provenance and full
expected coverage must still be bound independently. Bootstrap intervals enumerate
all 5^5 empirical resamples and are descriptive; overlapping folds are not IID
replications and do not support a population-superiority claim.

A6 workers accept original-unit matrix targets and fit training-only per-column
mean/std normalization (unweighted, constant-column std one). Native stopping
metrics use standardized targets and zero standardized initialization. Saved
bundles and training receipts retain `target_scale`; exported predictions and
replay are in original units. Final A6 scoring must supply the same training std
for standardized-average RMSE.

## Frozen real-data worker packets

`worker_data.py` exports all five Housing (A1/A11), Adult (A2), Covertype (A3),
Bike (A5), Parkinsons (A6), insurance (A7/A8/A9), Veteran (A10), and Concrete
(A12 ordinary GBDT) folds. It checks source arrays, reader hashes, recomputed
training encoders, exact partition hashes, group disjointness and A6 target scale
against the existing freezes. Other applications are explicitly unsupported by
this exporter and remain required work.

```bash
build/v1-env/bin/python -m benchmarks.v1.worker_data A6 build/a6-packets
build/v1-env/bin/python -m benchmarks.v1.worker_data_smoke build/real-worker-smoke
```

Use fresh output directories. Each fold contains a validation worker packet with
explicit early-stopping labels, separate train-row IDs, validation truth, test
features and test truth. The preparer is evaluation-side trusted code; these files
share a directory, so this is not OS-enforced test isolation. The execution runner
must control mounts/access before formal candidate trials. A12 appends age/28 to
ordinary GBDT features and emits separate support artifacts; this packet is not a
FormulaBoost learner input. A6 targets remain in original units; the worker owns
normalization and the frozen scale is retained for independent checking.

The smoke uses four rounds and patience three, checks row identity and finite
output shapes, and verifies the saved A6 scale. The worker checks reload before
emitting artifacts. It does not read test truth, select a model, or certify E3.

Adult packets retain official source/physical-line IDs; its official test set
is unchanged across the five stratified training splits. Categorical vocabularies
are fitted on training rows and checked against the freeze. Covertype retains
all seven classes and source row positions. Bike preserves `instant` source IDs,
expanding chronological windows and date-disjoint boundaries. Later observations
are excluded from each earlier origin; they are not forced into that fold's test
set. Preprocessing hashes bind positional source indices, while prediction/truth
packets carry the original row IDs where provided.

The exporter validates all folds before writing packets, then materializes one
fold at a time to limit memory use on Covertype. It still emits dense controls;
this is not a memory or throughput claim for the future foundation.

The exporter also binds policy counts (A7), positive individual paid claims (A8),
eligible-policy annualized paid totals (A9), and event/right-censored AFT inputs
(A10). Insurance applications inherit the same policy partitions; claims and
eligible policies use their separately frozen training encoders. A7 preserves
raw integer counts and supplies exposure for the worker's offset. A8 uses unit
weights per paid claim; its row ID is the retained joined claim position, while
policy ID controls grouping. A9 uses paid total/exposure and exposure weights,
with no exposure offset; separate period artifacts retain totals and exposure.
These numeric worker packets are not inputs for the parametric composition worker.

A10 restores the source-declared categorical features, exports event indicators
and hashes a `censoring.json` containing the frozen training reverse-KM estimate
and supported grid. Veteran original-source license review remains unresolved;
local adapter checks do not resolve that source gate or certify survival quality.

```bash
build/v1-env/bin/python -m benchmarks.v1.worker_data_smoke build/positive-survival-smoke --applications A7 A8 A9 A10
```

The smoke accepts an explicit subset of supported applications and records it in
its command. Without that option it checks every supported application. A4 and
A13 remain unsupported here and required in the full evaluation plan.

## Current OpenBoost worker (A1/A11 integration wave)

`openboost_worker.py JOB.json` consumes the same encoded train/validation packet
keys emitted by `worker_data.export`. It currently accepts A1 squared and A11
joint natural/ordinary Normal only, library=openboost, device=cpu, threads=1.
Run through `process_runner.execute(..., threads=1)` to set process thread limits.
Both tasks require explicit validation labels, even without patience. Unknown
options, unsupported tasks and test arrays fail rather than being ignored.
Other required application adapters remain pending.

A fixed budget exports the final model. Enabled patience exports the strict best
validation snapshot, including the initial base if no step improves it. Training
metadata records outer rounds, accepted commits, stop reason and selected model
identity; these counts are not interchangeable. A1 selection uses half squared
loss (same ordering as MSE); A11 uses Normal NLL. Internal training loss governs
Normal backtracking separately from validation selection.

`model.bin` is a versioned **JSON** evaluation bundle, not a pickle: a core raw
model plus application/output semantics. A1 returns mean `[N]`; A11 returns
mean and standard deviation `[N,2]` in the supplied target units. No target
scaling or offsets are added by these encoded-data adapters. The separate
`openboost_predict.py MODEL FEATURES OUTPUT` loads a packet with only `x` and
`row_ids` and does not import training recipes. It rejects mismatched output
semantics. The benchmark bundle is not a new stable public persistence API.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-current-worker-044
```

This runs four-round A1/A11 trials on each of the five frozen housing folds and
checks exact validation replay in a new process. It reads no test truth during
training or scoring, performs no configuration search and makes no quality or
performance claim. Exported packet separation is not an OS access boundary.
Formal test-label isolation, all application bindings and E3 remain open.

### A6 multi-output integration

The current worker also accepts A6 finite matrix targets and `mode=shared` or
`mode=independent`. It computes the same **unweighted training-population** mean
and standard deviation as the frozen evaluator/comparator protocol, even when
training weights are supplied. This is intentionally different from the general
public `TargetScale.fit`, which uses weights. The worker constructs a public
TargetScale from the frozen convention and uses it for both training and validation.
Sample weights still apply to loss/derivatives. Constant target scales use one.

Best-model selection and patience use the row-weighted mean of the sum of
standardized half squared errors across outputs. For a fixed output width this
has the same ordering as their mean; the recorded score retains the sum convention.
The saved A6 bundle includes `target_scale` (mean/std/constant), and restored
predictions apply the inverse transform once, returning original target units.
Training metadata records the convention and scale. This does not implement
cross-configuration A13 selection or the full A6 quality protocol.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-multi-worker-045 --applications A6
```

The smoke verifies all five grouped Parkinsons folds and exact scale equality
with the freeze. Default smoke applications are now A1/A6/A11; explicit application
selection permits bounded reruns. Four-round results are integration evidence only.

### Scale-bound A6 selection and current search

A6 selection protocols now require `train_targets` (hashed NPZ with row_ids/y)
and `target_scale` (hashed JSON with mean/std/constant). The independent audit
aligns targets exactly to training row IDs, recomputes the unweighted population
scale and rejects differences. Every `rmse_k` selection weight must equal the
inverse frozen standard deviation. The reported selection score is the mean
of these standardized RMSEs, not division by the sum of inverse scales.
Other applications retain their existing score definitions.

Protocol digests must still be pinned by the trusted orchestrator. This checks
internal consistency with supplied training data, not external dataset provenance
or filesystem isolation. All 16 configurations per method must finish successfully;
missing/failed/tampered trials cannot yield a receipt or selected test release.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.current_selection_smoke /tmp/openboost-selection-046
```

This synthetic A6/A13 integration runs 16 current configurations (shared and
independent trees, two learning rates and four depths), audits validation scores,
seals/re-audits the winner, and invokes fresh-process inference only after feature
release. All trials and failures are retained. It scores no test labels and is
not the frozen real-data quality grid, a speed benchmark or fused train-many.
A6 final comparative quality reporting and the remaining real searches remain open.

The same smoke accepts `--protected` on Linux with a root evaluator. It places
protocol, validation/scale records, test features and the selection receipt under
an evaluator-owned mode-0700 directory. Workers receive read-only train/validation
packets and job files, run as UID/GID 65534 with an 8-GiB address limit and a
1800-second deadline, and retain the current worker's one-thread contract. Completed
trial directories are reclaimed by root before the next trial starts. Installed
Python/package/source paths and output ancestors must be traversable by that UID.
Unsupported hosts fail before creating output; there is no privilege fallback.

The [Linux selection integration](evidence/protected-selection-070/README.md) passes
all three focused tests and a retained 16-trial run. The saved Linux receipt now has a macOS replay regression. Score-only
recomputation permits at most eight times the smaller binary64 spacing of each
finite value. All other receipt fields and the selected winner remain exact;
changes to ordering across a near tie still reject release. The original receipt
byte pin and every artifact hash remain exact. This bounded rule does not promise
portability across arbitrary numerical libraries or metric changes.
It is a synthetic four-round grid, not the full 300/1000-round search. The earlier
standalone permission probe does not validate this call path. Network/new-session
restrictions and separate author containers remain outside this mode's guarantee.

### A6 paired quality reporting

A6 quality cells require all `rmse_k` primary metrics followed by
`standardized_rmse`. Their `auxiliary` object must supply hashed `train_rows`
(NPZ row_ids), `train_targets` (NPZ row_ids/y), and `target_scale` (JSON
mean/std/constant). The report verifies aligned, unique training IDs, disjointness
from evaluation rows, matching target width and an exactly recomputed unweighted
training-population scale. Missing or inconsistent support is an error.

The standardized metric is the arithmetic mean of per-target original-unit RMSE
divided by each verified training standard deviation. Constant targets use scale
one. It is reported for each fold and compared across five paired folds alongside
every target. A passing average cannot override a failed target. As before, this
layer never marks E3 complete: selected-model provenance, full required coverage
and trusted source/protocol identity remain separate obligations.

### A2/A3 current classification workers

Current classification jobs require `classes=2` for A2 or an integer count of at
least three for A3. Targets are integer codes in `[0, classes)`; every declared
class must occur in training. Class order is persisted as `0,1,...,K-1`. The
prediction loader rejects missing or reordered class schemas. A2 emits P(class=1)
as `[N]`; A3 emits `[N,K]` probabilities in canonical encoded order. Patience/best
selection uses weighted log loss, and fixed-budget selection uses the final model.

External validation IDs may be unique integers or strings. Workers validate and
preserve them in emitted artifacts, while public data records use local integer
indices. These indices are execution-local and never replace exported source IDs.
Unsupported fields and label/class mismatches fail explicitly.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-classification-048-fixed --applications A2
```

A2 has five-fold frozen Adult validation integration evidence. A3 has weighted
synthetic direct-recipe/fresh-process checks in this slice; full Covertype runs
remain pending. The smoke accepts explicit A2/A3 selections, with seven classes
for Covertype. Defaults remain A1/A6/A11 to avoid silently broadening existing
runs. These four-round probes establish no classification quality/calibration,
search, CUDA or speed claim.

## Evaluator-owned execution freeze (Sprint 070)

For a pinned execution, provide a manifest independently frozen by the evaluator:

```bash
uv run --no-sync python -m benchmarks.v1.judge /path/to/producer-run \
  --frozen-manifest /path/to/evaluator/frozen.json --frozen-sha256 EVALUATOR_PINNED_FILE_SHA256
```

Both options are required together. The file must be outside the producer run
directory and match the evaluator-supplied raw-byte hash. Strict JSON parsing and
manifest validation apply to both inputs. The producer manifest must match the
entire evaluator manifest, including the expected cells, protocol and provenance.
Rehashing a producer-shrunken matrix cannot satisfy the independent freeze.
The Python API accepts `frozen_manifest=` from a trusted caller; its reported
`frozen_manifest_sha256` hashes canonical JSON and can differ from the CLI raw-file pin.

Without this argument, integrity remains relative to the producer's declared
matrix and `frozen_manifest_match` is null. A true match does not validate the
experimental design, authenticate provenance, prove all R/C/A/E obligations or
establish quality. `gate_results` stays empty. The evaluator must control the pin,
invocation and reference file. An outside-directory check is not an OS permission
boundary: process/container isolation remains separate work in Sprint 070.

### Current OpenBoost trial retention

`openboost_worker` explicitly runs its A1–A12 recipe and independent A5 quantile
paths with `retention="summary"`, reporting `diagnostic_retention` in training
metadata. This is a worker policy, not a new search hyperparameter or a change to
the public recipe default. Frozen model configurations, selected-model semantics
and prediction artifacts are unchanged. It reduces stored round arrays without
qualifying the full search's resource or quality gate. Other worker families must
be audited separately before large jobs.

### Explicit Linux worker identity and address limits

`process_runner.execute(..., address_limit_bytes=8 * 1024**3, unprivileged=True)`
requires a Linux root evaluator. It launches with hard/soft RLIMIT_AS limits,
UID/GID 65534, no supplementary groups, no_new_privs and a minimal explicit
environment. The fresh output directory belongs to that worker; its ancestors
must permit traversal. Evaluator-private inputs need separate root ownership and
permissions. Unsupported setup is rejected; there is no advisory fallback.

The runner records actual launch commands, configured limits, identity/environment
policy, logs and errors. It kills remaining same-process-group descendants before
artifact inspection in this mode. This is not a general hostile-code sandbox:
new sessions and network are not restricted. Use separate containers for independent
attempts and never share same-UID outputs across them. RLIMIT_AS limits virtual
address space, not measured resident memory; enclosing-container policy is separate.
Default execution keeps the existing inherited-identity/environment behavior.

`python -m benchmarks.v1.access_preflight /tmp/fresh-output` runs a bounded Modal
CPU probe with synthetic protected fixtures; it uploads no real datasets or sealed
tasks. Actual permission/resource errors remain distinct from the earlier injected
judge statuses. Passing this probe does not qualify full-search or author-eval gates.

### Full A6 OpenBoost resource planning

`python -m benchmarks.v1.a6_preflight_plan OUTPUT.json` compiles the frozen
300/1000-round configurations into 160 OpenBoost jobs (shared/independent topology,
five folds, sixteen configurations each). It writes once and launches nothing.
The plan records source-freeze hashes, fit-only upper bounds, first resource probes
and remaining requirements. It does not represent the full comparator matrix or
a completed full-search resource check. See `v1-sprints/070-a6-resource-plan.json`.

### Explicit comparator bin budgets

The numeric baseline worker accepts optional `config.bins`, an integer in [2, 256].
XGBoost and LightGBM receive `max_bin=bins`; CatBoost receives
`border_count=bins-1`, since that parameter counts split borders rather than
intervals. This applies to the worker's finite encoded inputs; it does not align
native quantization algorithms. NGBoost explicitly rejects this setting. Omitted
bins retain native defaults. `bin_budget_smoke.py` checks installed effective
parameters, stopping records and fresh-process A6 replay at 7 and 255 bins.

`a6_resource_preflight --comparators` runs only the three frozen fold-zero
configuration-00 comparator resource probes. It verifies the updated A6 plan's
input pins, uses the protected worker policy and fresh A6 replay, and stops on
failure without retries. Profile, paired and comparator modes are mutually
exclusive. This mode does not execute or certify the 400-job search.
