# Run 8: Validate Normal objective comparisons on real CUDA

Status: **executed once; allowance consumed** at `469ca0e` after explicit approval.
The [102 result](102-normal-cuda-validation.md) records 528/529 revised passes,
the exact 26 historical disagreements and all 409 declared JSON artifacts. The
overall verdict is false. The original request below remains the execution record;
it authorizes no retry. Local preparation was committed at `dbeff8a`, following
the complete bindings at `c881259`.

[Exact protocol and source hashes](092-comparison-run8.json) ·
[Isolated collection](092-isolated-collection.json) ·
[Construction and reflection](092-cohort-bindings.md)

## Concrete request

Upload **86 explicitly frozen files to Modal** and execute **one T4 invocation**:
**two requested CPUs, 8192 MiB requested memory, one container, at most 900 seconds
for the function and a shared 600-second deadline for both pytest cohorts, with
zero retries**. Image construction uses CUDA 12.6.3 / Python 3.12 and the same
eighteen pinned packages as run 7. It installs the exact core and D2 extension,
then builds the separate CPU inference environment. No external dataset is needed.

The upload is approximately 1.46 MB: 31 production Python files plus the verifier,
reference, numerical study, installed D2, build/replay/dispatch and package metadata
closure. The protocol lists every file; its own hash is recorded at clean dispatch,
and the other 85 hashes are frozen. Git, hidden directories, sealed H1/H2 cards and
independent-author material are excluded. This allowance includes no push or retry.
Retain up to **32 MiB of declared JSON artifacts**, plus logs, JUnit and manifests.

## Questions and acceptance

Two separate pytest processes execute the same installed production snapshot.
They produce separate logs, JUnit files and literal verdicts.

| Cohort | Cases | Required interpretation |
| --- | ---: | --- |
| Historical | 385 | Preserve the original 383 cases plus two run-7 diagnostics, byte-identical sources and tolerances. Expect the two original coefficient failures and 24 old recipe memory assertions to fail at those exact assertions. All other cases must pass. |
| Revised | 529 | All 383 bound requirements, 117 comparison checks, 27 consumer checks and two lowering/cost checks must pass. No missing, skipped, duplicate or extra cases. |

The historical memory assertions omit the newly owned best validation snapshot,
`4*N_validation*K` bytes. Their disagreement is declared before hardware, not
rewritten into a passing historical result. A different failure or unexpected pass
requires review. The overall completion flag requires the exact historical
disagreements and full revised acceptance; it never means historical all-pass
conformance. The 236 shared operation checks execute twice and are not additional
unique requirements.

The revised gate checks actual directed-double comparison lowering, numerical
bounds at stored device inputs, gradients, splits, leaves, complete trajectories,
acceptance/best/patience distinctions, ownership and final prediction/NLL/CRPS.
Installed D2 must exercise the same comparison boundary. Both cohorts retain
nineteen saved models and fresh CPU inference without CUDA or the D2 extension.
All original mathematical and float32 tolerances remain unchanged. The independent
float64 reference has identical historical/revised summaries for all ninety
settings; this does not predict successful real-device validation.

Retain **409 declared JSON files**: 78 historical model/replay/acceptance artifacts,
76 revised model/replay artifacts, 106 operation observations, 147 instrumented
case trajectories and two lowering/cost reports. Missing evidence fails completion.
Preserve partial files and all failures. Actual stored-input audit runs after the
real comparison and cannot supply the algorithm's decision.

Instrumented correctness timing includes diagnostic exports and high-precision
work. Separately time two weighted and two installed-D2 fits without comparison
instrumentation, including fixture/context creation, preparation, training, export
and any triggered compilation; time CPU prediction separately. Earlier tests may
warm kernels. This is bounded execution-cost evidence, not process-cold timing,
matched-quality speed, real-data value or author benefit.

## Dispatch and retrospective boundary

Only after approval, change both authorization fields to `approved` and commit.
Keep all frozen sources, cases, limits and resource settings unchanged. The
protocol's authorization-only hash change is recorded in the dispatch manifest;
the local collection's original pending-protocol hash remains preserved.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_cuda_comparison_manifest.py -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.cuda_comparison_preflight benchmarks/v1/evidence/cuda-comparison-092
```

The CLI refuses pending/consumed allowance, dirty source, source/budget drift,
a different output location or reuse before importing Modal. An attempted run
cannot be regenerated as a fresh allowance. Archive its output, mark the allowance
consumed and stop for retrospective even if validation passes. No automatic retry
or eighth-to-ninth-run carryover is authorized.

The retrospective must reconcile every case and artifact, inspect all unresolved
or changed comparison decisions, separate instrumentation from actual fit cost,
and review installed inference and ownership. Preserve the original failed runs
and known split near-tie. Decide the next construction/evaluation question from
that evidence before another run. Formal P7/E4, other required CUDA families,
independent author/accounting and A1–A13 application gates remain open.

## Local verification

**1794 CPU tests pass, one Linux-only skip.** Thirty-seven focused harness checks
pass, including actual archived JUnit and injected accounting/provenance failures.
The exact 86-file snapshot builds a wheel offline and collects all 914 executions
under Python `-I` outside the checkout, using the extracted package. Zero CUDA
tests have run. Ruff, MkDocs and wheel/sdist builds pass; MkDocs retains its existing
execution-page evidence-link warning. See the
[verification and failed attempts](../learnings/2026-09-07-v1-comparison-hardware-freeze.md).
