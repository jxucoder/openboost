# Run 9: Normal CUDA recipe correction verified

All **15 recipe cases pass** on a Tesla T4 at clean revision
`a7173d980cb5ea925de0b8b8f1ffab9b2595456a`. No failures, errors, skipped, missing,
duplicate or extra cases occur. All 46 uploaded hashes, 31 installed production
sources and eighteen pinned package versions match the approved freeze.

This is the one explicitly approved execution of the
[Sprint 103 packet](../../../../v1-sprints/103-normal-cuda-revalidation.md).
Its allowance is consumed; there is no retry or additional hardware authorization.
The embedded protocol in [manifest.json](manifest.json) retains the approved
dispatch state; the active sprint protocol separately records consumption.

## Corrected behavior and coverage

The forward three-anchor case now completes all assertions: ten terms are
accepted, best prefix nine is retained after a zero-valued tenth term, patience
uses its independent anchor, and best scores/predictions and trial comparisons
match. Joint/reverse best prefixes remain ten. Owned live bytes match before
closure and reach zero after closure. The same group verifies tiny improvements
despite equal reporting scores, rejection, zero rounds and five injected failure
paths. These are assertion results, not separately exported prediction traces.

The only source reused from run 8 that changes is the corrected recipe test,
whose bytes match `6026ebb`. Production and reused support are byte-identical.
The [offline audit](analyze.py) verifies actual source revisions, installed/uploaded
identity, package freezes, raw artifact hashes and JUnit case identities across
both runs. [analysis.json](analysis.json) records the resulting combined coverage.

| Evidence | Result | Interpretation |
| --- | ---: | --- |
| Run 8 revised cohort | 528/529 | Original raw failure remains unchanged. |
| Run 8 revised cases outside the recipe group | 514/514 | Carried forward with identical production. |
| Run 9 corrected recipe group | 15/15 | Actual new GPU execution, including the formerly failing case. |
| Combined revised coverage | 514 earlier + 15 new | All 529 bounded revised requirements have passing evidence across two runs. |

This is **not a single 529/529 invocation**. Run 8's overall verdict remains false,
and its historical cohort retains all 26 expected disagreements. All 420 indexed
run-8 files are unchanged. Its comparison/PTX/trajectory, installed-D2, fresh CPU
replay and tiny-fixture cost evidence are earlier observations, not rerun here.

## Environment and execution

| Field | Recorded value |
| --- | --- |
| GPU | Tesla T4, reported 15360 MiB |
| Driver | 580.95.05 |
| CUDA image | 12.6.3 development image, Ubuntu 22.04 |
| CUDA runtime / driver API | 12090 / 13000 |
| Python | 3.12.1 |
| OS | Linux 4.19.0, gVisor, x86_64, glibc 2.35 |
| Requested resources | 2 CPUs, 8192 MiB, 1 T4, 1 container |
| Visible CPUs | 18; distinct from the two requested CPUs |
| Limits | 900 function seconds, 600 test seconds, zero retries |
| UTC dispatch start / finish | 2026-09-08 06:28:47.686017 / 06:30:30.303569 |
| Dispatch elapsed | 102.617552 seconds, including image construction/setup |
| Worker elapsed | 15.462807701 seconds |
| Pytest elapsed | 13.509 seconds in JUnit; log rounds to 13.51 |

The manifest records all package versions, source hashes, installed paths and exact
test arguments. Pytest emits thirty low-occupancy warnings for the tiny fixture
launches. No skips or hidden CPU fallback substitute for CUDA execution. No timing
here measures end-to-end training throughput or matched-quality performance.

Approved command:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.cuda_recipe_preflight benchmarks/v1/evidence/cuda-recipe-103
```

Do not rerun that consumed command. Reproduce the offline verification with:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python benchmarks/v1/evidence/cuda-recipe-103/analyze.py --check
```

## Artifacts and limits

- [manifest.json](manifest.json): clean revision, approved protocol, source/runtime
  identity, timing and raw artifact hashes.
- [junit.xml](junit.xml), [pytest.log](pytest.log), [verdict.json](verdict.json):
  literal outputs and complete fifteen-case result.
- [analysis.json](analysis.json), [analyze.py](analyze.py): repeatable combined
  coverage audit that preserves the original failure.
- [archive-index.json](archive-index.json): hashes of all other files in this archive.

The recipe fixture blocker is closed. This bounded result does not establish full
Normal/application conformance, resolve the separate split near-tie limitation,
pass P7/E4 cost gates or implement other required CUDA recipes. The next planned
work is 080 required recipe coverage, followed by compatible train-many and real
workload quality/cost. Agent/author evaluation remains deferred by Sprint 101.
