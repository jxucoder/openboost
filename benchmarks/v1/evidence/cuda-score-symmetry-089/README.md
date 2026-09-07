# Scalar CUDA correction: 212 passing T4 checks

The single approved run at clean `af026efb2c4ad82cdca0228f83ba01dc064300c7`
passes **all 212 cases**, with no failures, errors, skips, duplicates or missing
cases. All 202 original cases pass, including the 14 that failed in
[run 4](../cuda-resident-078/README.md), plus ten new score diagnostics. The bounded
078-C/088 scalar correctness, residency and ownership checks now pass. This does
not establish full v1 device scope, real-data quality, performance or author benefit.

## Raw evidence and integrity

- [Manifest](manifest.json): clean revision, exact commands, all 49 snapshot file
  hashes, 27 installed module hashes, 17 pinned package versions, requested and
  observed environment, timestamps and artifact hashes.
- [Verdict](verdict.json): exact JUnit accounting and installed/snapshot checks.
- [JUnit](junit.xml): every original and added case.
- [Pytest output](pytest.log): unmodified measurements, both complete PTX outputs
  and 230 warnings. Search for `score_diagnostic=` to read the seven JSON records.
- [Protocol](../../../../v1-sprints/089-symmetry-run5.json) and
  [Sprint 089](../../../../v1-sprints/089-cuda-score-symmetry.md): scope, approval,
  correction, validation and retrospective.

Every snapshot hash matches Git at the dispatch revision. All installed module
hashes and pinned versions match. The three raw artifact hashes verify, and
recomputing the verdict reproduces the stored passing result. Raw files are
unchanged. The original manifest retains the approved dispatch protocol; only
the live protocol marks the compute allowance consumed. The earlier failed run
is retained separately and has not been rewritten.
Post-run production edits update only three module docstrings; AST comparison
excluding those docstrings matches the tested source. The hardware evidence
identifies `af026ef` regardless of later documentation commits.

## What the diagnostics establish

The weighted root has exactly swapped `(G,H)` child summaries:

| Candidate `(feature, threshold, missing_left)` | Left | Right |
| --- | --- | --- |
| `(0, 0, True)` | `(-3.222222328186035, 2)` | `(3.222221851348877, 7)` |
| `(0, 3, False)` | `(3.222221851348877, 7)` | `(-3.222222328186035, 2)` |

These are exported device measurements, with float32 bit patterns, weighted row
fields, histogram, parent and counts retained in the first JSON record. The
parent is `(-4.76837158203125e-7, 9)`. The archived and corrected scorers run on
the same resident candidate/parent buffers and context stream.

| Scorer | Gain at `(0, 0, True)` | Gain at `(0, 3, False)` | Winner |
| --- | ---: | ---: | --- |
| Archived run-4 function | 2.3793723583221436 | 2.3793725967407227 | `(0, 3, False)` |
| Corrected function | 2.3793725967407227 | 2.3793725967407227 | `(0, 0, True)` |

The old scores differ by one binary32 ULP (`0x401847a3` versus `0x401847a4`).
The corrected scores are both `0x401847a4`, so the unchanged exact-tie rule selects
the oracle's first candidate. The archived function reproduces the old selection
error on identical measured inputs; the corrected function restores equality.

The captured archived PTX contains a fused multiply-add at the child-score sum:

```text
fma.rn.f32 %f27, %f21, %f22, %f26;
```

The corrected PTX contains three explicit independently rounded multiplications,
then separate addition/subtractions:

```text
mul.rn.f32 %f23, %f21, %f22;
mul.rn.f32 %f27, %f25, %f26;
mul.rn.f32 %f31, %f29, %f30;
add.f32   %f32, %f23, %f27;
sub.f32   %f33, %f32, %f31;
sub.f32   %f36, %f33, %f9;
```

Complete corrected PTX is 26262 bytes, SHA256
`56a680d3d7905c66025feceb8fa1f63808d78733dd52016c96bfdad758bf8500`;
archived PTX is 26300 bytes, SHA256
`031d755aa8412db53ea41b04e6ff404cd1325cd9a6b0536d815a0088d28221ef`.
Both include the exact compiled signature. This confirms the contraction mechanism
for the reproduced counterexample in run 5. It does not retroactively provide
run-4 PTX or a SASS trace; the historical control is copied production code, not
an independent correctness oracle. Original-row references remain the oracle.

All six swapped-summary combinations pass across three regularization/penalty
settings and both column orders. Three adjacent-ULP choice checks pass at scales
`2^-10`, `1` and `2^10`: unequal scores remain strictly ordered and exact ties
still choose the first candidate. No tolerance or epsilon tie band was introduced.

## Accepted boundary and environment

| Module | Passing cases |
| --- | ---: |
| Score symmetry/diagnostics | 10 |
| Resident runtime/recipe | 72 |
| Resident objective/tree | 42 |
| Candidate/feasibility/route/leaf | 55 |
| Fields/aggregation | 21 |
| Execution/storage | 12 |

The original matrix now verifies weighted/missing, D2 cohort-feasibility and
validation-conflict fixtures across two rounds, depths 0/1/2 and cohort minima.
It includes gradients, named fields, exact topology/routes, leaves, raw predictions
and final losses; supplied policies; accepted/proposal/rejection/rollback ownership;
separate best/stopping; retained snapshots; and 24-round storage retention with
saved models predicting in a fresh CPU process without CUDA or training plugins.
These are small synthetic development fixtures and known designer extensions.

Observed hardware: Tesla T4, 15360 MiB, driver 580.95.05. Python 3.12.1 runs on
Linux 4.19.0 gVisor x86_64. The image tag is CUDA 12.6.3; the loaded runtime reports
12090 and driver API reports 13000. NumPy is 2.3.5, CuPy 13.6.0 and numba-cuda
0.27.0. Requested allocation is two CPU cores and 8192 MiB host memory; 18 visible
CPUs do not establish exclusive allocation. Private pool limits exclude CUDA
context, driver and JIT allocations.

Pytest reports 19.13 seconds; the worker interval is 20.693269501 seconds. These
are suite timings, not end-to-end fit/prediction benchmarks or matched-quality
cost ratios. The one invocation uses the approved 900-second function limit,
600-second test limit, 16-MiB pools and zero retries. All five allowances are
consumed. No further upload, invocation or retry is authorized by this result.

## Retrospective and next work

The public device operations now support verified bounded scalar training through
resident rounds and CPU-readable model export. Integration exposed a numerical
contract gap that primitive tests alone missed; a local score correction fixed
it without changing the public algorithm boundary or relaxing acceptance.

The next construction design is [079](../../../../v1-sprints/079-cuda-distribution-and-extension.md):
Normal K=2 ordinary/Fisher geometry, joint and ordered updates with real rejection,
and an installed D2 device extension. These should test whether fields, leaf/state
widths and transactions generalize through the same components. Design and freeze
independent fixtures before adding kernels. Original P7 Normal reproduction and
E4 cost remain separate required work. A scalar pass cannot substitute for them.

[069 author accounting/isolation](../../../../v1-sprints/069-authoring-pilot.md)
remains a parallel priority, with no independent attempt or new agent authorized.
All R/C/A/E coverage, train-many CUDA, real application quality and adoption gates
remain open. Reflect here before broader construction; any further hardware work
needs a new concrete freeze and allowance.
