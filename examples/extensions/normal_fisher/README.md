# normal_fisher

Independent CPU example package: `from normal_fisher import NormalFisher,
ChannelDecay`. Install its wheel alongside OpenBoost 1.0.0rc1; see
[the shared installation guide](../README.md) and executable `../demo.py`.

`NormalFisher` uses raw `mu` and `log_sigma`, weighted NLL gradients and diagonal
expected Fisher `(exp(-2*log_sigma), 2)`, with weights applied once. Initialization
uses weighted mean and variance floored at 1e-6. `constrain` returns mu/sigma.
Invalid weights, unsupported extra targets and non-finite states fail.

`ChannelDecay(tau=1)` returns full coefficients
`base_lr * {mu: 1, log_sigma: .5} / (1 + round_idx/tau)`. This is a predetermined
schedule, not line search. `tests/test_normal.py` supplies an independent
finite-difference NLL and analytic Fisher reference. GPU support is pending.
