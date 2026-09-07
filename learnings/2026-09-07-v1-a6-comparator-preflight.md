# 2026-09-07: Real A6 comparator resource preflight

## Context

Explicit bins now pass installed synthetic checks, but the three comparator
paths still lack real frozen-budget Linux resource evidence on the current A6
packet. The complete 400-job search stays pending.

## Decision or Result

Reuse a6_resource_preflight with a mutually exclusive comparators mode. Select
exactly XGBoost, LightGBM and CatBoost fold-zero configuration 00 from the pinned
400-job plan; preserve 300 rounds, patience 50, 255 bins, seed and input schema.
Missing or duplicate selected jobs fail. Validate every planning input hash and
the compiled evaluator plan before uploading. This is explicitly a bounded
preflight despite the full plan's dispatch_ready=false state.

## Changes

- The existing UID/resource fit path dispatches baseline_worker and records
  installed package versions with peak guest RSS. Native dependencies are pinned
  in the image; actual transitive versions are retained in each fit resource file.
- baseline_predict replays trusted A6 comparator bundles in a fresh process using
  only validation features/row IDs. No evaluator labels are needed for inference.
- Each fit retains 1800 seconds, 8-GiB address space and one thread. Each Modal
  function requests two CPUs and 8192 MiB with 1900 seconds and zero retries.
  Three functions run sequentially, stopping at the first failed result. Each
  replay has 60 seconds. The bound is 5400 fit-worker seconds plus setup/replay.

## Verification

Twenty-eight focused checks pass, covering exact probe selection, omitted and
duplicate methods, and incompatible mode combinations. The new replay CLI is
checked against committed installed synthetic models before the implementation
commit. Full regression and remote results follow below.

## Failed Attempts

None in preparation. Remote failures, if any, must remain in the evidence.

## Risks and Follow-ups

Only three of 240 comparator jobs are selected. Deeper/1000-round cases, other
folds, full selection and real quality remain open. No test data is uploaded.
Guest RSS and address space are distinct from requested container capacity; host
cgroup enforcement is not independently observed. Use the existing approved
train/validation packet and allowlisted public source upload only.

## Commits

Preparation commit precedes remote execution; no push.

Preparation validation: 1120 CPU tests passed, one Linux-only skip; 28 focused
checks, lint and docs pass. All three new CLI replays exactly reproduce the
committed 255-bin synthetic predictions. Remote execution is pending.
