# bounded_leaves

Independent CPU/CUDA example package: `from bounded_leaves import BoundedNewton`.
Install its wheel alongside OpenBoost 1.0.0rc1; see
[the shared installation guide](../README.md) and executable `../demo.py`.

Pass `BoundedNewton(bound=.5)` to `LevelWiseBuilder(leaf_rule=...)`. The rule uses
public `NewtonLeafRule` and clips its outputs to ±bound on the context's array
backend. Bound must be finite and positive; version 0.2.0 declares CPU/CUDA capability.
Empty/zero-gradient nodes stay zero. The builder retains its original split
criterion. `tests/test_leaf.py` checks independent values and invalid bounds;
shared composition tests verify changed leaves, next gradients and persistence.
Install the `cuda` extra for GPU dependencies; the shared guide describes the
real-device installed-wheel verification.
