# Sprint 092-A: A bounded loss-change experiment

Status: local mathematics prototype under the user's approval of Sprint 092.
No production operation, consumer migration, upload or device execution in this
slice. The seven previous device allowances remain consumed. The historical
run-7 acceptance policy and its two failures remain unchanged.

## Decision and arithmetic contract

Use an enclosure of the exact mathematical weighted Normal loss difference at
the supplied stored inputs. The enclosure, rather than the rounded full NLL or
the sign of a point estimate, determines whether a change is resolved. The
prototype is independent original-row Python code in
[`normal_comparison.py`](../tests/v1/reference/normal_comparison.py); it imports
neither the production package nor a device library. It is not a CUDA emulator.

The basic arithmetic assumption is separately rounded IEEE binary64 `+`, `-`,
`*` and `/`, round to nearest with gradual underflow. Each computed extremum is
expanded by one `nextafter` towards its respective infinity. Under this contract
the interval contains the exact operation on every value in the input intervals.
All four endpoint products/quotients are considered; division rejects an interval
containing zero. Nonfinite expanded endpoints produce an unresolved result.
Exact identities with point zero/one are allowed; equality of two uncertain
intervals never licenses cancellation. The runtime format check and rational
tests exercise these assumptions locally; they do not prove every platform or
compiler respects them. CPU production must retain the arithmetic contract.

Each scalar input is interpreted as its exact stored binary value. Residuals,
offset sums, mean/scale changes, row products, weighting, total weight, reduction
and final division all carry intervals. This avoids assuming that residual
formation or a compensated reduction eliminates other sources of error. Only
actual stored-raw equality is classified as unchanged. Different raw that happen
to have equal loss, including a change on only a zero-weight row, remain unresolved
at zero. Input/domain validation precedes either shortcut.

## Why the exponential implementation is bounded

The [Python 3.12 math documentation](https://docs.python.org/3.12/library/math.html)
describes its platform C math wrappers; it does not provide the portable rounding
guarantee needed here. CUDA 12.6's standard-function table lists one ULP for double
`exp`/`expm1`, but explicitly says the bounds come from nonexhaustive testing and
are not guaranteed. Those figures cannot establish a rigorous enclosing interval.
See [CUDA 12.6 mathematical functions](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#mathematical-functions).

Instead, this bounded experiment uses elementary arithmetic and an explicit Taylor
remainder. For an endpoint `x` in `[-64, 64]`, repeatedly halve with outward
rounding until the reduced interval lies inside `[-1/16, 1/16]`. For every value
`t` in that interval, evaluate the polynomial

```text
s = t + t^2/2! + ... + t^18/18!
```

using interval multiplication, division by exact small integers, and addition.
Starting at degree 19 the ratio of successive absolute terms is at most
`(1/16)/20 = 1/320`. Thus the entire omitted tail is bounded by
`|t|^19/19! / (1 - 1/320) < 2 * |t|^19/19!`. An outward upper bound on the
absolute eighteenth term, multiplied by an outward bound on `2*|t|/19`, bounds
the remainder. Expand `s` by that symmetric remainder before reconstruction.
This bound scales with the argument; it is not a threshold fitted to a loss.

Reconstruct `exp(x)` by squaring `1+s` after each halving. Reconstruct `expm1(x)`
by repeatedly applying `s * (s+2)`, preserving small changes without subtracting
one from a full exponential. These are separate paths: reconstructing `exp` as
the final `expm1 + 1` would lose relative information for negative arguments.
Monotonicity of both functions allows evaluation at interval endpoints. Each
reconstruction operation also expands outwards. The code special-cases exactly
zero, where both function values are known exactly.

This is a derived enclosure conditional on the stated arithmetic contract, not
a claim of formal machine-code verification. A future CUDA implementation can
use explicit directed basic operations. NVIDIA documents `__dadd_rd/ru`,
`__dmul_rd/ru` and `__ddiv_rd/ru`, including separate rounding and noncontraction
for the arithmetic intrinsics. Their lowering, actual device execution and cost
still need evidence. See [CUDA double intrinsics](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html).
Neither standard `exp` error measurements nor a silent host fallback replaces
that evidence.

## Normal difference and supported domain

The [092 design identity](092-normal-comparison-design.md) follows by writing the
new residual as `r+dm` and the new precision as `p*exp(-2*dl)`, expanding the
square, and subtracting the shared old quadratic. Offsets cancel in `dm` and
`dl`; they still occur once in the old residual and precision. The common Normal
constant cancels. The prototype propagates the intervals through that expression,
then computes `sum(w*delta) / sum(w)` with outward rounding at every operation.

The experimental support is finite aligned nonempty inputs, nonnegative weights
with positive total, and finite enclosures at all intermediate operations. Both
old and new quadratic row domains are checked on every row, including weight zero.
All evaluated exponent *intervals* must lie within `[-64,64]`; this includes both
old/new effective scales and their change. Conservative intervals at an exact
boundary can be unresolved. Unsupported exponents, arithmetic overflow, or a
weight-sum enclosure containing zero yield unresolved with a specific reason and
no fabricated estimate. Nonfinite inputs, misalignment and invalid weights raise.
The bounded comparator does not expand the existing float32 gradient/Fisher domain;
public CPU/device objective validation will remain an additional prerequisite.

The interval endpoints are authoritative. Its midpoint and outward radius provide
an estimate and uncertainty for reporting. `upper < 0` proves improvement and
`lower > 0` proves worsening. Otherwise the sign is unresolved. A patience consumer
can require `upper < -min_delta`, with a finite nonnegative, explicitly supplied
`min_delta`; the comparison operation does not own the training/best/stopping
anchor. An unresolved trial rejects in backtracking and retains its reason. A
fixed-step caller may explicitly accept a valid finite worsening proposal.

## Distinguishing evidence and limitations

The tests replay all ten actual recorded run-7 trials over both training and
validation inputs. Both mean coefficients in both update orders are worsening;
the three rounded-unchanged reverse scale trials remain unchanged. At coefficient
four the training enclosure is approximately `[5.904685e-18, 5.905035e-18]`, wholly
above zero, while the historical full-loss predicate accepted it. Its validation
comparison can correctly improve at the same time. These are CPU reanalyses of
stored GPU inputs, not new GPU results.

The exact analytic `-2^-61` change remains a resolved improvement. A scale-only
`2^-52` step at a stationary point has positive true change but an interval
containing zero, and is correctly unresolved. The original 60/100-digit oracle
does not meet its forty-relative-digit agreement target for this quadratic
cancellation: 60 digits lose about 32 digits during subtraction. Tests retain
those estimates and add 160/220 digits without modifying the historical oracle
or the binary64 policy. [Python Decimal](https://docs.python.org/3.12/library/decimal.html)
documents correctly rounded `exp`; whole-expression agreement remains supporting
numerical evidence, not the proof of the binary64 enclosure.

Additional checks cover exact-rational arithmetic including subnormals, exponential
endpoints, whole-row seeded f32/f64 inputs, weights/offsets/permutations, zero-weight
domain errors and actual storage rounding. No metric, gradient, leaf, reduction,
serialization or old verifier changes are included. The polynomial and interval
work may be expensive; this slice makes no throughput, acceptance-frequency or
end-to-end claim. The historical-to-revised cohort mapping precedes any public
operation. Public ownership, three separate consumers and actual CUDA validation
remain subsequent 092 slices.
