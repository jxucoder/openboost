# Local Normal acceptance counterexample

This [CPU-only study](neighbors.json) was generated from clean `ba46b3a` using
the base in run 6's passing D2 ordinary/forward/backtracking model. The original
training inputs equal the conflict fixture. Its manifest/model hashes and the
analysis/oracle source hashes are retained. No device code executes or is emulated.

| Adjacent float32 change | Full float64 NLL subtraction | 100-digit NLL difference |
| --- | ---: | ---: |
| Mean down | -4.440892098500626e-16 | +5.9048662229e-18 |
| Mean up | -4.440892098500626e-16 | +9.7345181150e-17 |
| Log scale down | +1.1102230246251565e-14 | +1.0915265462e-14 |
| Log scale up | +1.7319479184152442e-14 | +1.7506443183e-14 |

All four differences agree to the stated check at 60 and 100 decimal digits.
Both mean changes worsen the original-row objective while the float64 total-loss
subtraction reports improvement. The common Normal constant is canceled before
the high-precision row differences are averaged, and the stored binary inputs are
converted exactly. Agreement at two precisions is not an interval-arithmetic bound.

This proves a local numerical counterexample. It does **not** identify the round,
channel, raw values or losses from either failing GPU test: those were not saved
in run 6. It does not change the frozen oracle, production acceptance or failed
run verdict. [Sprint 091](../../../../v1-sprints/091-normal-acceptance-diagnostics.md)
requires actual failing-state observations before a policy decision.

Reproduce from the recorded source revision:

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.analyze_normal_neighbors /tmp/openboost-normal-neighbors-091.json
```

Python, NumPy, OS, command, clean revision and exact input/source hashes are in the
artifact. Later reruns naturally record a different revision, command or dirty
state; compare the mathematical results separately from provenance.
