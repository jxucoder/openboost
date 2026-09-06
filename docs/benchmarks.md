# Benchmarks

Benchmark scripts and verified results are separate deliverables. The remote
integration brought speed, UCI quality, FormulaBoost, and survival benchmark
harnesses, but their previously transcribed tables do not have corresponding
committed raw reports with complete provenance in this checkout. Those tables
are withheld pending reproducible artifacts; they are not release gates passed.

## Available evidence

The older CPU comparison is committed at
[ngboost_comparison_20260720.json](https://github.com/jxucoder/openboost/blob/504fdd0bfc60e5d8e7518250e087fb7e4766d1b4/benchmarks/results/ngboost_comparison_20260720.json).
It covers its recorded datasets, seed, and configuration only. It cannot
support a GPU speed claim or full-suite quality claim.

The ScoringBench integration in `benchmarks/scoringbench/` keeps official
quality runs separate from OpenBoost's large-sample extension. Neither the
presence of that integration nor passing unit tests implies accepted results
on an external leaderboard.

## Reproduce candidate comparisons

These commands run experiments; execution alone does not establish a result.

```bash
uv run modal run benchmarks/bench_probabilistic.py --suite speed
uv run modal run benchmarks/bench_probabilistic.py::quality
uv run modal run benchmarks/bench_formula.py
uv run modal run benchmarks/bench_survival.py
OPENBOOST_BACKEND=cpu uv run --with ngboost python benchmarks/bench_ngboost_comparison.py
```

For NaturalBoost, compare held-out NLL, CRPS and interval coverage at matched
training budgets before discussing fit and prediction time. Record failed or
unavailable datasets rather than treating them as ties.

For FormulaBoost, measure in-domain error, extrapolation error, and parameter
recovery separately. Compare full/diagonal/plain preconditioning, a global
formula fit, and appropriate tree baselines. A known synthetic formula is a
mechanism test, not proof of real-world extrapolation.

For WeibullAFT, measure censored NLL, ranking, calibration and parameter
recovery separately. Report the censoring process and model assumptions.

## Required artifacts

Each numerical claim needs a committed raw result recording source SHA and
dirty state, dataset/version/hash, splits/seeds, package versions, OS,
CPU/RAM/threads, GPU/driver/CUDA, exact commands, actual execution path,
fallbacks, and timing/compilation policy. Report repeated runs and failures.

Measure end-to-end fit and prediction, including objective computation,
transfers, and synchronization. CPU/GPU comparisons must identify resource
differences. Do not extrapolate unmeasured timings or trade quality for speed
without stating the change.

The historical scripts write to ignored `benchmarks/results/` paths. A fresh
run must explicitly retain and review its raw files before any documentation
is updated with numbers. The foundation execution plan adds a dedicated
tracked evidence directory and stricter result handling.
