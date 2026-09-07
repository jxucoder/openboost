# Normal CUDA run 6: 381 passed, two acceptance failures

The single approved T4 run at clean
`4143d188b9635749308ef52382a1562928ae2612` completed all **383** preregistered
cases: **381 passed, two failed, none skipped or errored**. This is a failed
acceptance run. Its raw verdict remains false; no fixture or tolerance changed.

All 212 run-5 regressions pass. Normal operations pass 23/23, mapped runtime
94/96, recipes 31/31, and installed D2/fresh inference 20/20. The remaining
passing case is the separately labelled near-tie diagnostic; it establishes
measurement capture, not repaired structural parity.

The failures are `test_frozen_three_round_transactions` with
`[False-8.0-forward-ordinary-0-conflict-0-None]` and
`[False-8.0-reverse-ordinary-0-conflict-0-None]`. Both observed trial coefficients
`(8,4)` where the reference requires all six `(8,4,2,1,.5,.25)`. Full Normal
acceptance remains open. No rerun was made; all six device allowances are consumed.

## Unmodified raw evidence

- [Manifest](manifest.json): dispatch revision, exact commands, 67 uploaded source
  hashes, installed core/extension hashes, 18 pinned packages, environment and
  all 79 raw artifact hashes.
- [Verdict](verdict.json): exact case accounting and failing overall outcome.
- [JUnit](junit.xml) and [pytest output](pytest.log): both failures, passing cases,
  score/PTX diagnostics, Normal measurements and 407 warnings.
- [Saved inference artifacts](normal/): 76 JSON files for eighteen D2 trajectories
  and a separate missing-input Normal model, each with model, inputs, measurements
  and a fresh CPU replay without CUDA or the training extension.
- [Frozen protocol](../../../../v1-sprints/090-normal-run6.json) and
  [request](../../../../v1-sprints/090-normal-run6-request.md).

All 67 source hashes match Git at the dispatch revision; all 79 raw artifact
hashes verify. Recomputing the verdict reproduces the stored failure. Installed
core, extension and package versions match, the CPU environment built, and all
76 declared inference artifacts are present. Only the live protocol marks its
allowance consumed; the raw manifest retains the approved dispatch protocol.

Local archival checks replay all nineteen saved models and compare exact fresh
CPU predictions plus the frozen device-to-CPU tolerance. These do not rerun CUDA.
The detailed retrospective is recorded in Sprint 090 and the learning log.

## Numerical interpretation and remaining uncertainty

The [derived analysis](analysis.json) is separate from the raw manifest. Reproduce
it without CUDA using `uv run --no-sync python -m benchmarks.v1.analyze_normal_run6
/tmp/openboost-run6-analysis.json`. It records the raw input hashes and analysis/
reference source hashes. This is retrospective CPU math, not another GPU run.

The conflict fixture's float64 constant optimum is
`[0.3401637243387091, 1.0863723616655747]`, with gradient sums about
`[-5.55e-17, -1.11e-16]`. Both reference orders reject every depth-zero trial.
A separate passing D2 model, whose training arrays equal the conflict fixture's,
records the device base `[0.3401637375354767, 1.0863723754882812]`. Evaluating the
independent CPU mathematics at that saved base gives gradient sums approximately
`[9.20e-9, 1.66e-7]`. The two reference loss values differ by one float64 ULP.

This supports investigating initialization/storage rounding and acceptance near
stationarity. It does **not** establish the exact failed-device mechanism. The
two failed tests did not emit their current round/channel, raw values, geometry
or trial loss bits before asserting. The next diagnostic must capture those
values before proposing a correction. Neither a blanket epsilon nor changed
expected decisions has been applied. These original failures remain required
regressions and keep the run's verdict false.

The preregistered split near-tie is a different issue. Measured gains for
`(0,0,True)` and `(0,3,False)` are both `1.115696907043457`; the strict chooser
selects the first. CPU histogram gains differ slightly. Full-tree prediction for
zero-weight row 0 differs by **0.009251285171136714**. The diagnostic successfully
records the known structural limitation; it does not establish its repair.

## Measured cost scope

Tesla T4, 15360 MiB, driver 580.95.05; Python 3.12.1, Linux 4.19.0 gVisor x86_64.
Image tag CUDA 12.6.3; loaded runtime 12090 and driver API 13000. Requested resources
were two CPUs and 8192 MiB host memory; 18 visible CPUs do not imply exclusive
allocation. Private pool bounds exclude context/driver/JIT memory.

Dispatch-to-finish wall time was **363.47 seconds**, including image setup and
remote orchestration. The worker interval was **68.76 seconds**; pytest reports
**66.84 seconds**. There was one invocation and zero retries.

The eighteen D2 configurations produce 36 first/repeated tiny fits. First-in-case
fit times range from 0.268 to 1.523 seconds; repeated times from 0.268 to 0.464
seconds. They include context creation, device preparation, cohort upload, every
trial and model export. CPU fixture creation and supplied binning precede the
timer. Earlier tests can compile kernels, so these are not process-cold timings.
The 36 fits include 186 rejected trials, retained in the measurement artifacts.

For the repeated natural/undamped backtracking fits (six training rows, six
validation rows, three rounds), the raw measurements are:

| Update | Fit seconds | Trials / rejected | Synchronizations | Validation-check exports | Metric exports | Device copies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint | 0.464 | 9 / 6 | 1066 | 2792 B | 160 B | 2232 B |
| Forward | 0.363 | 18 / 12 | 1241 | 3152 B | 304 B | 2664 B |
| Reverse | 0.348 | 18 / 12 | 1213 | 3080 B | 304 B | 2616 B |

All three upload 660 bytes during the measured fit. Recorded peak live allocation
across all D2 fits is 2282 bytes and peak private pool use is 28672 bytes; all end
with zero owned live bytes. The synchronization count includes operation-level
validation and copies, not just adaptive metric decisions. It exposes significant
small-call overhead worth profiling later, without justifying removal of ownership
or validity checks. CPU prediction timings and fresh load/prediction timings are
separately retained; there is no GPU prediction speed claim or matched-quality
comparator here. Original P7 and E4 remain unfulfilled by this tiny diagnostic.

## Retrospective boundary

The architecture now has real evidence that an installed external component can
constrain splits in a Normal GPU recipe through shared public fields, candidate
operations and transactions. This is a useful foundation result, not independent
author benefit or adoption evidence. Recipe counts and passing subsets do not
replace the remaining R/C/A obligations.

The immediate follow-up is a documented acceptance-policy investigation with
actual failing-state telemetry, preserving this failed run. Full Normal acceptance
must pass before extending the required CUDA recipe matrix. A future hardware
package requires a new allowance. Stop here for the planned retrospective; no
source correction, tolerance change, retry, push or broader evaluation followed
this run.
