# Run 7: Capture the original Normal acceptance failures

Status: frozen locally; upload and compute allowances are pending. All six prior
allowances are consumed. [Exact protocol and hashes](091-acceptance-run7.json).
Harness construction: `e0a043b`, with offline CLI provenance completed in this
freeze. Production code and every old test/reference match run 6 byte for byte.

## Concrete request

Upload the **70 explicitly frozen files to Modal** and execute **one T4 invocation**:
**2 requested CPUs, 8192 MiB requested memory, at most 900 seconds for the function
and 600 seconds for pytest, no retries and one container**. These are the previous
resource ceilings. Image construction installs the same eighteen pinned packages,
the exact core snapshot and external D2 package, and builds the same separate CPU
inference environment. No external dataset is needed.

The 70 files comprise 28 production Python files, the original verifier/reference
closure, two new diagnostic cases and their arithmetic/observation helpers, the
three-file D2 project, build/replay/dispatch scripts, package metadata and the JSON
protocol. Its own hash is recorded at clean dispatch; the other 69 hashes are
prefrozen. The upload excludes Git, hidden directories, sealed task cards and
independent-author material. No broader workspace upload or push is included.

## What the run answers

The original 383 cases run unchanged, followed by two new observation cases that
invoke the exact failing forward/reverse tests. The original tests still use their
original float64 oracle and strict full-loss comparison. Observation wrappers call
each original operation once and restore all patched methods. They record the
actual prepared inputs, initialization, raw/metric bits, gradient/Fisher, named
fields, root totals, leaf, coefficient, acceptance, version and best prefix before
the coefficient assertion stops execution. Raw copies are released and ownership
guards check that observation leaves state and uploaded data unchanged.

The new cases retain the original coefficient assertion as a conformance failure
inside their diagnostic JSON. They do not suppress the failure of the original
unwrapped case. Unexpected exceptions/assertions remain diagnostic failures.
High-precision analysis runs only after the original test ends and never drives
an algorithm decision. The extra synchronization makes these trace cases unsuitable
for product timing; no performance claim follows.

## Acceptance and interpretation

- **385 exact cases**, no missing, duplicate or skipped cases. The first 383 match
  the original case list. The shared all-pass judge is unchanged: any original
  failure keeps the overall verdict false. The two known failures are expected
  to remain because this package changes no production acceptance logic.
- Diagnostic completion requires both new cases to pass and retain complete,
  internally consistent traces with exact bit encodings, ownership checks and
  final zero live bytes. Their embedded original failures must identify the known
  coefficient assertion. Any other failure or a non-reproduced original failure
  requires reflection; it is not evidence of a repaired algorithm.
- Installed core/D2 sources, snapshot sources, eighteen package versions, separate
  CPU environment and all nineteen saved-model replays remain required. Retain
  pytest output, JUnit, manifest, strict verdict and **78 declared JSON files**:
  the prior 76 plus `normal/acceptance/forward.json` and `reverse.json`.
  The total JSON cap remains 2 MiB; partial failed outputs are retained.
- Bounds remain 8192 rows / 32 features / 32 bins for aggregation and at most
  eight rows / depth two / 24 rounds for training regressions. The new observations
  use the existing six-row, depth-zero, three-round case and a 16 MiB private pool.
- Preserve the earlier failed archive and separate split near-tie. Do not revise
  acceptance semantics or the original oracle based only on a passing diagnostic.

## Dispatch and stop boundary

After explicit approval of this concrete package, mark both authorization fields
approved, commit, recheck the freeze and dispatch once:

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.cuda_acceptance_preflight benchmarks/v1/evidence/cuda-acceptance-091
```

The entry point rejects pending/consumed authorization, a dirty checkout, source
changes, another output location or reused output before importing Modal. Archive
the result even if the expected overall verdict is false, consume the allowance,
and analyze each trace at its actual stored state:

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.normal_acceptance_trace benchmarks/v1/evidence/cuda-acceptance-091/normal/acceptance/forward.json /tmp/openboost-091-forward-analysis.json
```

Repeat offline analysis for reverse order. Compare kernel geometry/reduction/leaf
and mapped raw values first, then measured absolute losses against original-row
float64 math and 60/100-digit differences at identical inputs. Distinguish this
from the original reference's different initial precision and trajectory. Stop
for reflection and a concrete numerical-policy proposal before a correction or
another hardware run. This allowance includes no retry.

Normal conformance, original P7/E4, all remaining CUDA families and independent
author/accounting/application gates remain open.

## Local verification

**1,474 CPU tests pass**, with one Linux-only skip. Forty-five focused oracle,
analysis and run-6 archive tests plus seven new freeze/dispatch checks pass.
Production/changed-file Ruff and MkDocs pass; the pre-existing execution-page
evidence-link warning remains. The exact 70-file snapshot collects all 385 cases
under isolated Python outside the repository against installed core files whose
28 hashes match the freeze. Collection is not GPU execution. See the
[learning record](../learnings/2026-09-07-v1-normal-acceptance-diagnostics.md).
