# 2026-09-08: Native GLM comparison and recipe validation

## Context

The user explicitly approves the exact 85-file run-12 Modal packet. The preceding
local construction has independent mathematical and CPU evidence but no device
execution. Approval is committed at `fe12beb` before the single clean dispatch.

## Decision or Result

All 571 T4 tests pass, including all 153 new GLM cases. The
[raw archive](../benchmarks/v1/evidence/cuda-glm-108/README.md) and
[108 retrospective](../v1-sprints/108-glm-validation-result.md) preserve the scope.
Keep the native binary/Poisson operations and shared scalar recipe. All twelve
hardware allowances are consumed; stop at the retrospective before multiclass.

## Changes

- Retain the original manifest/log/JUnit/verdict and all 77 declared JSON reports.
  Only the current protocol's two authorization fields move to consumed; the
  manifest preserves approved execution bytes and all frozen source hashes.
- Add an offline source/artifact/numerical/model audit and negative controls.
  Recompute direct likelihood differences from stored float32 values, decode
  lossless fixture bytes and replay final/best models against the independent
  original-row trajectory oracle. No production or frozen test changes follow
  hardware observation.
- Update execution claims and planning to distinguish the bounded passing GLM
  matrix from full required device/application/quality/cost coverage.

## Verification

Hardware: 571 passes, no failures or skips, 52.93 s pytest / 55.324 s worker /
382.977 s dispatch. All 85 uploaded files, 35 installed core files, eighteen pinned
packages and 77 JSON artifacts match; retained JSON totals 1,013,949 bytes.
Offline audit: eighty raw hashes plus manifest binding; all 246 comparison bounds
enclose direct 220-digit differences; all 32 final/best models replay from the
actual retained input bytes. Both actual PTX records contain six required directed
double instruction forms. No speed ratio or formal E4 pass is inferred.

Fifty-six local evidence/freeze/retention controls pass in 12.99 seconds, including
eleven new audit checks. The controls reject corrupted provenance, changed raw
artifacts, false interval/identity/method claims and altered direct likelihoods.
Final CPU regression passes 2,277 tests with one Linux-only skip in 18.37 seconds.
Production and changed-file Ruff pass; the documentation build passes with its
existing historical Normal evidence-link warning. Local replay uses macOS 26.3
x86_64, Python 3.12.12 and NumPy 2.3.5. No new production change requires another
hardware run at this closure.
The final archive index binds 84 files plus its own separately retained index.
The repository ignores `.log` files, so stage the declared raw `pytest.log`
explicitly; verify all 85 archive paths are in Git before committing evidence.

## Failed Attempts

No remote retry, case failure or post-run tolerance adjustment occurs. The new
offline analyzer initially needed an import-order lint correction before running;
its retained-input and mathematical replay then pass on the first actual artifacts.
All historical failed GPU verdicts remain unchanged.

## Risks and Follow-ups

The tested shapes, domains and T4 compilation do not prove universal accuracy,
full multiclass/AFT/vector topology, train-many scaling or real-data quality.
Convex bounds can conservatively reject finite steps; that behavior remains part
of the declared numerical contract. Multiclass is the next required construction
cell. Derive its geometry and reliable comparison before recipe consumers, retaining
all application families and the separate author-study deferral.

## Commits

- `3f90252`: exact run-12 freeze and retained-evidence controls.
- `fe12beb`: approved clean execution revision; production is unchanged from 107.
