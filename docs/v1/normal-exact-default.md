# Exact Normal learner policies

The device Normal recipe now selects exact stored-field scalar Newton ordering
and original-row Newton leaves when no custom learner is supplied. Its existing
`reg_lambda`, `min_child_h`, `split_penalty` and `max_depth` options configure
those operations. Custom learners keep ownership of their numerical parameters.
Other device recipes and the low-level tree builder retain their own defaults.

The installed cohort extension composes the same public operations, with exact
independent cohort minima and a separately owned one-time upload of information
columns. Each ordering call uploads its small column-index/minimum parameter
arrays; it does not upload the per-row cohort information again.

Historical work exercised exact learners, default policies, installed cohort
extensions, non-tie and parameter controls, recipe edge cases and fresh inference.
The original collection failure and its narrow correction remain in development
history. These newer archives are outside this PR; see the [checkpoint](checkpoint.md)
for the pending validation of the curated candidate. No large-workload speed or
real-application quality claim follows from this interface change.
