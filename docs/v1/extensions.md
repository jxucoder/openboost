# Public development extensions

The repository's `examples/v1_extensions/` contains two separately installable CPU
packages using only public OpenBoost interfaces. `ob-cohort-splits` supplies
independent cohort information and custom split feasibility. `ob-penalized-leaves`
replaces leaf solving with its own weighted pinball/quadratic optimizer.

These reuse histogram/routing, recipe state and model artifacts. Three-round
checks compare independent mathematical oracles and demonstrate that changed
leaves affect subsequent updates. A separate wheel verifier installs both
packages outside the source checkout, removes them after training and verifies
fresh-process core inference on their saved models.

See the example directory's README and verifier for reproduction. These are
repository-authored development probes. They establish neither independent
authoring advantage nor adoption, full E2/E6 acceptance or CUDA execution.
