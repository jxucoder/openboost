# 2026-09-07: Real Normal comparison validation and a no-op prefix counterexample

## Context

After deferring agent evaluation, the user explicitly approved the existing
86-file Modal upload and single bounded T4 invocation. The earlier ambiguous
`finish` dispatch was blocked before process creation; no allowance was consumed
until the separately approved run at `469ca0e`.

## Decision or Result

Run 8 returns 528/529 revised passes and exactly the 26 preregistered historical
failures. The numerical correction passes its actual-device operations and full
revised trajectories. The remaining failure is an unconditional ten-term expected
best prefix in a forward recipe fixture. Its tenth term is zero-valued; strict
improvement requires keeping prefix nine. Exact-rational fixture analysis agrees
with the observed state. Preserve the failed verdict and correct this test
separately; do not alter production mathematics or loosen tolerances.

## Changes

- [Raw run](../benchmarks/v1/evidence/cuda-comparison-092/README.md): preserve
  complete logs, JUnit, manifest, literal verdicts and all declared JSON outputs.
- [Offline analysis](../benchmarks/v1/evidence/cuda-comparison-092/analysis.json):
  verify raw hashes, dispatch source identity, recorded bounds, PTX hashes and
  CPU replay bindings. Derive the fixture's joint/forward/reverse best prefixes
  as ten/nine/ten without a new device run.
- [102 retrospective](../v1-sprints/102-normal-cuda-validation.md) records scope,
  the fixture counterexample and next correction. Run-8 authorization is consumed;
  source/case/resource freezes remain untouched.
- Guidance and public CUDA documentation distinguish real passing subgroups from
  the incomplete 529-case gate. Agent evaluation stays deferred.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_cuda_comparison_manifest.py -n 0 -q`:
  37 local checks pass in 0.15 seconds before dispatch.
- The exact frozen GPU CLI executes once on Tesla T4, Python 3.12.1, driver
  580.95.05, CUDA runtime API 12090 / driver API 13000. All 86 source hashes and
  eighteen package versions match the clean dispatch revision.
- 416 raw artifact hashes verify, including all 409 declared JSON files. No
  skipped/error cases or unexpected historical outcomes. Both old false
  improvements are classified as worsening. All 2,548 recorded trajectory
  comparisons pass their independent numerical audit.
- Nineteen models in each cohort replay in the independent CPU environment with
  CUDA and the D2 extension absent. All six directed-double lowering checks pass.
- Four uninstrumented fixture fits preserve repeated identities and reference
  NLL/CRPS; each exports 480 comparison bytes for fifteen comparisons and closes
  to zero owned live bytes. These are synthetic, potentially warmed timings.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python benchmarks/v1/evidence/cuda-comparison-092/analyze.py --check`
  reproduces the derived artifact audit. Ruff and documentation checks accompany
  the evidence commit; no production source changes occur in that slice.
- All 420 indexed original/analysis/documentation files are staged byte-identically.
  MkDocs builds with the existing 090 evidence-link warning. Authored whitespace
  checks pass; four raw pytest log/JUnit files retain their original trailing
  whitespace so their measured hashes do not change. The credential-pattern
  scan finds no credential patterns in the archive.

## Failed Attempts

The old best-prefix assertion fails after its stopping observations pass. Later
assertions in that same case are not reached. A final no-op can advance the
accepted model without advancing its best prefix; the test conflated these two
states. A fixture diagnosis is not grounds to rewrite raw results as passing.

The earlier authorization rejection is preserved at `dffcfd5`; the subsequent
explicit approval resolves it. No rejected attempt was retried indirectly.

## Risks and Follow-ups

The original 529-case acceptance remains incomplete until the corrected fixture
is frozen and tested on real hardware. All eight allowances are consumed. No
retry, wider source upload or further remote invocation is included. Other
required CUDA recipes, train-many, real application quality and P7/E4 remain open.
The planned retrospective is complete; the next local slice is the narrow fixture
correction plus independent CPU no-op prefix coverage.

## Commits

- `f7f3c60` — defer author evaluation and prioritize foundation execution.
- `b58b168`, `dffcfd5` — preserve the initial authorization interpretation and rejection.
- `469ca0e` — explicitly approved clean execution revision.
- This commit archives the failed run, independent analysis and retrospective.

## Local correction after the evidence commit

Archive commit `c36f96a` precedes the source change. The CUDA fixture now checks
ten accepted terms and the order-specific strict best prefix separately. Three
new CPU tests use public transactions to check all joint/ordered commit boundaries,
accepted no-op retention and equal final predictions. Production code, tolerances,
all run-8 artifacts and its consumed freeze remain unchanged. The frozen old test
hash still identifies the actual failed execution; current test source differs.

The focused consumer file passes all thirteen checks in 0.26 seconds. Fifteen CUDA
recipe cases collect with the original IDs, without device execution. The archived
analyzer reproduces its original report. Full regression and final checks are
recorded before the correction commit:

- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short`:
  1949 pass, one Linux-only skip in 13.25 seconds.
- Production/changed-test Ruff, Markdown link checks and MkDocs build pass; the
  existing 090 evidence-link warning remains. No frozen artifact or consumed
  protocol changes; only the active recipe test differs from run-8 sources.

Actual CUDA validation of the changed
assertion and its previously unreached checks remains outstanding.
