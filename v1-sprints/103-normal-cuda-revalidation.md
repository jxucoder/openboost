# Sprint 103: Revalidate the corrected Normal CUDA recipe fixture

Status: local preparation; upload and compute authorization pending. No new
hardware run has executed. Run 8's allowance is consumed.

## Purpose and plan

[Sprint 102](102-normal-cuda-validation.md) retains the real run-8 result:
528/529 revised cases pass at `469ca0e`. The forward best-prefix assertion expects
ten after a zero-valued tenth term, while strict best selection requires nine.
The test-only correction at `6026ebb` checks ten accepted terms separately and
preserves all subsequent prediction, comparison, stopping and ownership checks.
Three CPU transaction tests independently verify ordered/joint no-op semantics.

1. Freeze the corrected recipe file and its complete import/package closure;
   verify production identity against run 8 and collect the installed snapshot.
2. Verify the existing dispatch/provenance judge locally and commit this packet.
3. After explicit approval for this concrete upload and GPU invocation, execute
   once, retain raw results, and stop for a retrospective. Never reuse run 8's
   consumed allowance or replace its failed verdict.

The smallest outstanding check is the forward case's best-prefix assertion and
the assertions after it. Rerun the entire fifteen-case recipe group to cover its
joint/reverse siblings, tiny improvements, full rejection, injected failures and
zero-round ownership. Production, numerical tolerances and test IDs stay unchanged.
The existing single-cohort runner suffices; no new device implementation or
remote orchestration is needed for this correction.

## Concrete run request

The [protocol](103-recipe-run9.json) lists **46 files, approximately 1.01 MB**:
45 prefrozen sources and the protocol, whose own hash is recorded at dispatch.
Request one Modal T4 invocation, two CPUs, 8192 MiB memory, one container, at most 900
function seconds, a 600-second pytest deadline and zero retries. These are the
existing runner's upper bounds, not predicted runtime. Image construction is
outside the function deadline. Use CUDA 12.6.3 / Python 3.12 and the same eighteen
pinned packages as run 8. Install the frozen core and run from outside its source
directory. No external dataset, model API, D2 installation or separate CPU replay
environment is needed for these fifteen recipe tests.

The closure includes the core, package metadata, runner, corrected test, imported
helpers and their numerical study JSON. The generic runner also includes its
existing histogram/reference support files. Support tests are not selected for
execution. Git, hidden directories, sealed tasks and author material are excluded.
Retain the full pytest log, JUnit, literal verdict and manifest with installed and
snapshot hashes, package versions, hardware details and worker duration.

After approval, record it in this sprint and set both protocol authorizations to
approved in a clean commit. Execute exactly:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.cuda_recipe_preflight benchmarks/v1/evidence/cuda-recipe-103
```

The fixed output must not exist before dispatch. A started attempt consumes this
single allowance even if it fails. Preserve partial evidence; no automatic retry,
additional upload, push or model invocation follows from this packet.

## Acceptance and evidence interpretation

All fifteen exact cases must pass on actual CUDA with no skipped, duplicate,
missing or extra cases and exit code zero. Installed core sources, every uploaded
snapshot hash and all pinned package versions must match. In particular, the
corrected forward case must reach all later best-score/raw-prediction, comparison
and ownership assertions and close with zero live bytes. Joint/reverse retain
best prefix ten; forward retains nine; all three accept ten terms.

Run 9 can establish fifteen recipe passes. Combining it with run 8 requires a
separate explicit provenance review: the other 514 revised cases already passed,
production and all reused support sources must be identical, and the only changed
previously uploaded source must be this corrected test. Report combined coverage
as 514 earlier passes plus fifteen new passes, never as a single 529/529 run.
Run 8's historical failures and raw overall false verdict remain immutable.
Its comparison/PTX/trajectory/replay/cost artifacts are earlier evidence, not
re-executed run-9 results. The local collection report is not a hardware result.

## Local verification

The [isolated collection report](103-isolated-collection.json) verifies all 31
production files against the measured run-8 sources. Among reused uploads, only
the corrected recipe test differs; its current bytes match `6026ebb`. The raw
run-8 manifest/verdict hashes match, all other 514 revised cases passed, and all
fifteen recipe IDs remain identical. An offline wheel built from the exact upload
closure imports from the extracted installation and collects all fifteen cases.
No GPU test executes locally. Repeat this read/build/collection check with:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.check_recipe_run9
```

The existing aggregation, Normal and comparison manifest suites pass 57 checks
in 0.33 seconds. They cover failed/incomplete case matrices, source/package
identity, authorization and artifact accounting. Production/support Ruff passes.
Full CPU regression remains the unchanged `6026ebb` result: 1949 passes and one
Linux-only skip. This preparation changes no production or recipe test source.
The pending dispatch command rejects before importing Modal or creating output.
Run 8's analyzer and all 420 archive-index hashes verify unchanged. New Markdown
links and whitespace checks pass; MkDocs builds with the existing 090 evidence-link
warning.

## Retrospective boundary and next work

Stop after this invocation to audit its exact result and the combined evidence.
An unexpected failure remains a failure and requires diagnosis before further
hardware. If it passes, continue [080 required CUDA recipes](080-cuda-required-recipes.md),
then [081 compatible train-many](081-cuda-train-many.md) and
[082 workload quality/cost](082-end-to-end-cost.md). This narrow correction neither
passes all R/C/A requirements nor resolves the known split near-tie limitation.
The [101 author-study deferral](101-defer-author-evaluation.md) remains in effect.
