# Sprint 105: Frozen validation comparison request

Status: approved and executed once at clean `dd84247`; all 474 cases and three
cost gates pass. The [raw evidence and audit](../benchmarks/v1/evidence/parallel-validation-105/README.md)
are retained. All eleven allowances are now consumed; the current protocol marks
this one consumed while the manifest preserves its approved execution bytes.
All 87 frozen source hashes remain unchanged. This request implements the approved
[Sprint 105 construction](105-parallel-validation-and-reproducible-cost.md).

## Exact payload and execution

The [protocol](105-validation-run11.json) lists 88 uploaded files, totaling
1,457,669 bytes in the pending freeze (about 1.39 MiB). Every file except the
protocol itself has a frozen SHA-256; the protocol's final bytes are hashed at
dispatch after its two authorization fields change. The payload contains the
31-file candidate core, two exact original source overlays, installed D2 extension,
selected tests/oracles and benchmark support. No author task cards, credentials,
local replay reports, Git history or whole working directory are uploaded.

Request one Modal T4 invocation, two requested CPU cores, 8,192 MiB container
memory, one container, 900-second function cap, 600-second pytest cap and zero
retries. Requested cores/capacity remain distinct from guest-visible hardware.
The image is CUDA 12.6.3 / Ubuntu 22.04 / Python 3.12 with the same eighteen pinned
packages as run 10. Image building precedes the function cap; this is a resource
and invocation limit, not a guaranteed dollar price.

## Correctness and timing matrix

The candidate collects 474 cases from an isolated installed wheel and copied
snapshot: 431 existing revised cases, 39 new field-validation checks and four
cost/profile cases. Existing cases cover scalar gradients/splits/leaves/predictions,
transactions, buffer ownership, Normal operations and comparisons, both recipe
cohorts, installed D2 and fresh CPU inference. Assertions and tolerances are
unchanged. The prior 96-case compared Normal runtime matrix and two old
lowering/cost cases are excluded from this bounded repetition. Their historical
evidence is preserved; this packet does not establish full Normal conformance.

Each row below generates inputs once on the worker, saves their lossless bytes,
and supplies the same snapshot to each child. Each GPU arm requires one first fit
and three warm fits. Each declared CPU control requires one first and one warm fit.

| Workload | CPU child cap | Original GPU cap | Candidate GPU cap | Candidate/original warm fit limit |
| --- | ---: | ---: | ---: | ---: |
| Squared, 10,000 rows | 25 s | 30 s | 30 s | 1.10 |
| Squared, 100,000 rows | 50 s | 95 s | 95 s | 0.80 |
| Normal, 10,000 rows | Not in this packet | 65 s | 65 s | 1.10 |

Settings stay at sixteen features, 32 bins, twenty rounds, depth three, learning
rate 0.1, regularization 1, fixed steps, seed 7 and a 512 MiB private device pool.
Data includes missing features, offsets and zero/nonuniform weights. Both arms
must produce byte-identical saved models and exact predictions, with unchanged
launch/flag-export counts and zero live owned bytes after cleanup. Independent
scores must replay from saved input bytes. Squared CPU controls also require
at most 1% relative metric difference and normalized prediction RMSE. No new
Normal CPU cost/quality comparison is claimed.

The original installed wheel is reconstructed from run-10 revision `c8f7ebc`
using two frozen source overlays; all 31 original source hashes must match the
run-10 manifest. Only `device.py` and `_device_kernels.py` differ in the candidate.
Both installed GPU cores must see the eighteen pinned package versions. A separate
NumPy/core-only environment supplies the candidate CPU controls.

Each arm uses a fresh child and private empty CuPy/driver disk caches. First-fit
timing includes context setup, host binning, training, model export, cleanup and
triggered compilation. Per-fit replay, prediction timing, scoring and artifact
serialization follow the fit timer and precede the next repetition. Fixed arm
order is CPU where declared, original GPU, candidate GPU. There is no randomized
order, confidence interval or external-library claim.

A separate two-arm profile permits fifteen seconds per arm for eleven calls on
100,000-by-two fields. It records kernel-name counts, host dispatch time, CUDA
event intervals and wall time. Event intervals include host enqueue gaps and
blocking flag checks; they are not exclusive kernel execution time. Profile
measurements cannot enter fit-speed ratios.

All measurement/profile child caps total 485 seconds. Baseline installation has
thirty seconds, leaving 85 seconds for correctness, setup and auditing within the
600-second pytest cap. The selected existing cases took about 69 seconds in the
prior T4 JUnit report; that observation motivates the bound, not a completion
guarantee. Failed or incomplete cases remain failures and retain completed fits.

## Evidence and decision

Retain exactly seventeen declared JSON files, capped at 64 MiB: input snapshots,
each arm's complete/partial fits and saved models/predictions/quality, paired
judgments, baseline installation and both profiles. The three local input-size
estimates total 15,998,594 bytes; actual worker bytes and identities are recorded
there. Existing regression tests retain JUnit outcomes, not a new full trajectory
archive. The manifest also records source/package identities, GPU/driver/CUDA,
host details, requested resources, actual commands, logs and failures.

All 474 cases, provenance, declared artifacts, model/quality checks and cost gates
must pass. A 20% warm improvement on squared 100,000 rows is the engineering target;
ten-percent regressions fail the two smaller-workload gates. Partial repetitions
have no complete-case ratio. Formal E4 and required recipe/application scope are
unchanged. Stop after this invocation to inspect correctness, achieved speed and
profile evidence before choosing another optimization or returning to 080/081/082.

## Local verification

- [Isolated collection](105-isolated-collection.json): 474 collected, zero executed,
  installed candidate and snapshot hashes checked.
- [Original installation](105-baseline-install-local.json): all 31 original hashes
  match; both saved Normal CPU fits replay. Local NumPy 2.3.5 / Hatchling 1.32.0
  verification is separate from the pending eighteen-package Linux/CUDA check.
- Full CPU regression: 1,994 passed, one Linux-only skip, 12.95 seconds. Local
  controls cover actual child timeout retention, incomplete/profile/source/model/
  quality/ownership failures and the source/authorization guards.
- Production and changed support/test Ruff checks pass. GPU collection and CPU
  replay do not validate the new CUDA barrier or establish acceleration.

After exact approval, update the two pending authorization fields, commit that
approval state, dispatch once from clean Git, preserve every raw result and mark
the allowance consumed. No push, release or external leaderboard update is included.

## Approved execution plan

1. Check the unchanged source freeze and local guards, then commit approval.
2. Dispatch the fixed packet once within the declared limits; retain all failures.
3. Audit exact inputs/models/quality and timing eligibility, commit raw evidence
   and the retrospective, and stop at the planned decision boundary.
