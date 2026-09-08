# Sprint 108: Frozen binary/Poisson validation request

Status: the user explicitly approves this exact upload and hardware packet after
reviewing the 85-file / single-T4 request. No remote invocation has occurred yet.
All eleven previous GPU allowances remain consumed. This is the concrete device
gate following the [107 construction retrospective](107-glm-comparison-and-recipes.md).

## Exact payload and execution

The [protocol](108-glm-run12.json) freezes 85 files totaling 1,443,454 bytes
(about 1.38 MiB) in its pending form: 35 production Python files, fifteen selected
test files, 25 support files and ten fixed packaging/protocol/reference files.
Every file except the protocol has a frozen SHA-256. The final protocol bytes
are hashed at dispatch after its two authorization fields change. The payload
includes public code, synthetic fixtures, independent numerical oracles and the
existing Normal comparison study; it excludes Git history, credentials, author
task cards, local reports and the rest of the working directory.

Request one Modal T4 invocation, two requested CPU cores, 8,192 MiB container
memory, one container, 900-second function cap, 600-second pytest cap and zero
retries. The image is CUDA 12.6.3 / Ubuntu 22.04 / Python 3.12 with the same
eighteen pinned packages as run 11. Requested resources differ from guest-visible
hardware. Image building precedes the function cap; this is a resource/invocation
bound, not a guaranteed dollar cap. No additional allowance follows a failure,
timeout or incomplete result.

The existing bounded aggregation launcher supplies upload, source/package
verification, isolated installed execution and partial-result retention. The new
entry point selects this protocol; it changes no launcher or production code.
The authorization guard runs before importing Modal. Output location is fixed to
`benchmarks/v1/evidence/cuda-glm-108` and must not already exist.

After the separate allowance and a clean source commit, the exact command is:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.cuda_glm_preflight benchmarks/v1/evidence/cuda-glm-108
```

## Correctness matrix

All 571 collected case IDs are explicit in the protocol and reproduced from an
isolated installed wheel/copied snapshot. Collection executes zero CUDA cases.

| Cohort | Cases | Required observations |
| --- | ---: | --- |
| Scalar/storage | 212 | Storage, histograms, candidate selection, routing, leaves, trees, resident transactions, score symmetry and inference |
| Normal regression | 167 | 23 objective operations, 117 comparison controls, twelve runtime consumers and fifteen corrected recipe consumers |
| Field validation | 39 | Per-column flags, tail lanes, all-row invalid domains and failure recovery |
| Binary/Poisson components | 38 | Independent geometry/base/loss/fields, offsets/exposure/weights and explicit support rejection |
| Prescribed GLM rounds | 6 | Two-round fields/splits/leaves/raw/metrics and class-aware persisted inference |
| GLM comparison | 77 | 59 independent stored-input cases, domain/identity/cleanup, no host/reporting fallback and actual PTX |
| GLM recipes | 32 | Sixteen default-grower trajectories, retries/rejections, current/best/patience anchors, equal-score improvements, zero rounds, failure cleanup and saved inference |

Existing regression assertions and tolerances are unchanged. The old compared
Normal runtime, installed D2, complete trajectory and cost cohorts are not repeated;
their historical evidence remains immutable and separate. This bounded repetition
does not establish full Normal conformance or all required R1/R4 capabilities.

GLM fields/leaves/raw use relative tolerance `1e-4` and absolute tolerance `1e-5`;
recipe reported metrics use `1e-3` relative and absolute tolerance. Discrete topology,
trial decisions, versions, best prefix and stop state must match their independent
references exactly. Comparison bounds must contain the direct 220-digit objective
difference for all 59 numerical cases and every measured recipe comparison; expected
signs and unchanged classifications stay frozen. No reporting-loss subtraction or
post-run tolerance change can replace those controls. Both families must retain
actual PTX containing directed double addition, multiplication and division in
both rounding directions.

The sixteen recipe settings cover depth one/two, fixed steps, ordinary and large
backtracking steps and zero steps for both objectives. Their saved models replay
in a separate environment containing NumPy 2.3.5 and the installed core, with
CUDA and training imports additionally denied. The independent oracle operates on
original rows and explicit float32 boundaries; it is not the production backend.

## Evidence, acceptance and retrospective

Retain exactly 77 named JSON files within 64 MiB: 59 numerical input/bound/direct-
likelihood/counter reports, two complete PTX records and sixteen recipe reports
containing lossless actual fixture bytes, final/best models, trial/stop histories
and every comparison's stored inputs and direct likelihood difference. Numeric
and PTX reports are written before their result assertions, allowing diagnostic
retention when those assertions fail. A trajectory that fails earlier may leave
no final recipe report; incompleteness fails the frozen verdict rather than
silently reducing the artifact requirement.

The manifest records clean revision, exact command, source/package identities,
installed source hashes, requested resources, OS/CPU/GPU/driver/CUDA details and
raw logs/JUnit. A successful verdict requires every declared case without skips
or duplicates, every source and package identity, the CPU replay environment and
all 77 retained artifacts. Preserve partial outputs and the literal failed verdict
on any failure. There is no automatic retry or modification of earlier freezes.

One pytest process includes first-use compilation and may warm shared kernels
through earlier regressions. Decimal calculations, diagnostic exports and CPU
replay instrument these correctness checks; their durations are not fit-cost or
matched-quality speed evidence. The existing related cohorts fit within earlier
bounded runs, but new GLM compilation cost and completion within this cap are
unverified. Formal E4 remains open.

Stop after this invocation for a source/evidence audit and retrospective. Correct
any demonstrated numerical/ownership/persistence failure before further recipes.
If it passes, return to remaining required 080 CUDA recipes, then compatible
081 train-many and evidence-led 082 quality/cost work. Multiclass, AFT, vector
topology and all other R/C/A requirements remain in scope. Author studies stay
deferred under 101.

## Local verification

The [isolated collection](108-isolated-collection.json) matches all 571 case IDs
and every installed core/copied source hash. The local freeze and existing
retention/manifest controls pass 45 tests in 0.51 seconds, including 25 new checks
for source/allowance/reuse guards, exact verdict requirements, artifact routing,
literal PTX retention and lossless missing-value fixture snapshots. No hardware
test has executed. The preceding production closure passed 2,241 CPU tests with
one Linux-only skip; this packet changes only test support, evidence routing and
planning. Production/changed-file Ruff and the documentation build pass.
