This change adds a programmable Python/CUDA boosting foundation: owned device
storage, named fields, histograms, split/feasibility/routing/leaf operations,
trees and resident transactions now support squared, Normal, binary and Poisson
recipes. Objective-owned loss-change bounds keep training acceptance, best-model
selection and patience separate even when reported losses round to the same value.

CPU execution gains verified incremental proposal evaluation, reusable encodings,
structural stopping results and bounded trace retention. The branch also preserves
the planning, failures and raw evidence accumulated since PR #24; most changed
files are evidence rather than production code.

## Validation and evidence

- Local CPU regression: 2,284 passed, one platform skip, 880 GPU/benchmark and
  ninety explicitly historical full-loss deselections. All ninety settings have
  active objective-comparison replacements, verified by collection and execution.
  Full production/test Ruff, strict MkDocs and wheel/sdist builds pass.
- [Run 12](https://github.com/jxucoder/openboost/blob/c965cb6/benchmarks/v1/evidence/cuda-glm-108/README.md):
  571/571 real T4 cases pass, including 153 new GLM cases and 418 regressions.
  All 77 artifacts are retained; offline audit verifies 246 loss-change bounds and
  replays all 32 final/best models from lossless input bytes.
- [Normal comparison/revalidation](https://github.com/jxucoder/openboost/blob/c965cb6/benchmarks/v1/evidence/cuda-recipe-103/README.md):
  bounded revised coverage is complete across two runs, with the original failed
  verdict and historical disagreements preserved.
- [Run 11 cost evidence](https://github.com/jxucoder/openboost/blob/c965cb6/benchmarks/v1/evidence/parallel-validation-105/README.md):
  all 474 T4 checks and three frozen internal cost gates pass. Parallel field
  validation lowers synthetic 100,000-row squared warm fit time from 13.513 to
  8.947 seconds (33.79%), preserving model/prediction bytes. This is an internal
  synthetic comparison, not external-library parity or formal E4.
- CI now fetches full history for source-provenance replay, selects CPU checks on
  hosted CPU runners and builds documentation strictly. The manual GPU workflow
  remains non-executing; no new paid GPU/model invocation is part of this PR.
- The [first hosted CPU run](https://github.com/jxucoder/openboost/actions/runs/34240154310)
  exposed nine platform-sensitive legacy reporting/decision assertions. Current
  tests now exercise objective comparisons under native and controlled reporting;
  original full-loss sources and hardware evidence remain unchanged. The matrix
  retains all host/version outcomes with fail-fast disabled.

## Scope and remaining work

This remains an experimental foundation, not a drop-in replacement or completed
v1 release. Required multiclass, AFT and vector-topology CUDA cells, compatible
device train-many, real application quality and formal E4 remain open. Authoring
and adoption studies are deferred under Sprint 101; retained study infrastructure
does not constitute author/adoption evidence. All R/C/A requirements remain.

## Merge strategy

Use **Create a merge commit**. Evidence tests resolve the original execution Git
SHAs; squash or rebase merging would break their availability in future clean
checkouts. Review the hosted Linux/macOS, Python 3.10/3.12 and documentation checks
before merging. This PR does not publish a package or dispatch hardware.
