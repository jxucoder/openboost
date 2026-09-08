# Normal acceptance run 7: failures reproduced with complete diagnostics

The single approved T4 invocation at clean `80740f27c07c9d676531edf2467bd82a7ef97da6`
finishes with **383 passes, two failures, no skips/errors**. The exact two run-6
failures recur and all 383 original results are unchanged. Both new diagnostic
cases pass. The overall [verdict](verdict.json) remains **false**; successful
measurement does not repair conformance. The allowance is consumed, with no retry.

The [manifest](manifest.json), [pytest output](pytest.log), [JUnit](junit.xml) and
78 declared JSON outputs retain all measured evidence. All 70 source hashes match
the clean dispatch revision; installed core/D2 sources and all eighteen pinned
package versions match. Every one of 81 raw artifact hashes verifies. All nineteen
saved models replay without CUDA or D2 in the separate CPU environment; their
model bytes also equal run 6's models. Partial outcomes and the original failures
have not been rewritten. Derived analyses below are separate from the raw manifest.

## What the failing-state observations establish

Both [forward](normal/acceptance/forward.json) and
[reverse](normal/acceptance/reverse.json) fail on **round zero's mean update**.
Forward reaches it immediately; reverse first rejects all six log-scale trials.
The mean update rejects coefficient 8 and accepts coefficient 4, while the frozen
float64 trajectory requires six rejections. Both records end at the original
coefficient-list assertion and retain the preceding values/bits.

| Observed mean trial | Coefficient 8 | Coefficient 4 |
| --- | ---: | ---: |
| Raw mean | 0.3401636779308319 | 0.3401637077331543 |
| Change from initial mean in float32 values | -2 ULP | -1 ULP |
| Reported device NLL difference | 0 | -8.881784197001252e-16 |
| Original-row float64 NLL difference at those inputs | 0 | -4.440892098500626e-16 |
| 100-digit original-row NLL difference | +1.150597798185516e-16 | +5.904866222924252e-18 |
| Device decision | Reject | Accept |

Initial device mean is `0.3401637375354767`, log-scale `1.0863723754882812`.
At coefficient 4 the measured NLL changes from `2.5053108948702483` to
`2.5053108948702474` (two float64 ULPs), but high-precision mathematics at the exact
stored inputs shows a small worsening. Both 60- and 100-digit estimates agree.
This is the actual failed candidate, not an inference from a different saved model.
The float64 original-row calculation also gets its sign wrong at these inputs.
Two-precision agreement is numerical evidence, not an interval-arithmetic proof.

The preceding gradient/Fisher arrays exactly match the float32-rounded independent
geometry at the stored inputs. Their rounded mean-gradient field sums exactly to
zero in an accurate CPU sum, but the device's ordered float32 accumulation produces
`4.470348358154297e-08`. With curvature 6 and regularization 1, the measured leaf is
`-6.38621200366174e-09`. Thus aggregation cancellation creates a small nonzero
direction, and absolute-loss rounding then admits its worsening candidate. These
are distinct numerical boundaries; changing only the leaf or only counting shape
parity would not establish reliable loss comparison.

Reverse order's first log-scale trials at 8, 4 and 2 worsen training loss; trials
at 1, .5 and .25 leave stored raw unchanged. All six reject and preserve version 0.
The subsequent mean trial is identical to forward. At its acceptance, training
loss is misclassified, validation actually improves by about `4.76e-11`, and
version/term count/best prefix advance to 1 consistently with those reported
decisions. The transaction is not misrouting the decision or mixing train and
validation. Observation guards and final zero live bytes pass in both traces.

## Reproduction and environment

The frozen [request](../../../../v1-sprints/091-acceptance-run7-request.md) and its
protocol are preserved in the manifest. The invocation ran once on Tesla T4,
15360 MiB reported GPU memory, driver 580.95.05, Python 3.12.1 and Linux/gVisor.
CUDA image 12.6.3 reports runtime API 12090 and driver API 13000. Requested capacity
was two CPUs/8192 MiB; eighteen visible CPUs are not exclusive allocation. The
private allocation pool is capped at 16 MiB, excluding context/driver/JIT memory.

Dispatch wall time was 267.184 seconds, worker time 76.001 seconds, pytest 73.52
seconds. These are bounded correctness/diagnostic timings, including extra
observation synchronization; they do not establish end-to-end competitive cost.
The known split near-tie diagnostic and original P7/E4 gaps remain separate.

Run the frozen offline analyzer for either trace:

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.normal_acceptance_trace benchmarks/v1/evidence/cuda-acceptance-091/normal/acceptance/forward.json /tmp/openboost-091-forward-analysis.json
```

The derived [forward analysis](analysis-forward.json) and
[reverse analysis](analysis-reverse.json) retain input/source hashes, exact loss
differences and their CPU environment. They execute no CUDA and are not part of
the immutable 81-artifact dispatch hash list.

## Retrospective boundary

The diagnostic question is answered: an accepted near-stationary candidate worsens
the same mathematical training objective, even though both GPU and ordinary
float64 absolute-loss comparisons report improvement. Preserve this counterexample
when designing an explicit, numerically justified comparison operation. Do not
silently widen tolerances, add a blanket epsilon, or rewrite the original tests
to declare the old run passing. Full Normal conformance, P7/E4, remaining device
families and independent author/application evidence remain open. See the
[sprint retrospective](../../../../v1-sprints/091-normal-acceptance-diagnostics.md).
