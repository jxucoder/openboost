# OpenBoost v1

OpenBoost is a programmable boosting foundation for researchers and agents, under
construction. Initial public CPU ownership, run-state and mapped ensemble artifact
components are available; see [CPU state usage](cpu-state.md).

[Numeric operations](numeric-ops.md) provide binning, histograms, candidate
selection, routing and scalar leaves. [Numeric tree policies](trees.md) compose these
operations and support numeric inference/persistence. The first complete
[squared-error recipe](squared.md) provides fixed/backtracking CPU boosting.
[Normal boosting](normal.md) uses the same state and scalar learners for joint
mean/log-scale updates. [Formula and sequential runs](formula-runs.md) add
full-metric structured updates and independent heterogeneous execution. CUDA
execution is not implemented yet. [Categorical support](categorical.md) now uses
explicit dictionaries and equality conditions in all three growth policies.
All R1–R9/C1–C7/A1–A13 remain required. Evaluation preparation continues alongside
the user-approved B03–B06 construction overlap. No complete quality, speed or
agent/adoption result is claimed for the new production foundation.

The old production API was retired. Historical examples require revision
`50acfc6`; current APIs are not backward compatible. See the repository's
`v1-sprints/` for construction records and `planning/` for requirements and gates.

[Binary classification](binary.md) adds typed class schemas, stable logistic
geometry and persisted probability/label output through the same foundation.
[Multiclass and vector leaves](multiclass.md) add joint softmax updates, separate
split/leaf statistics and arbitrary learner-to-model output mappings.

[Query-local ranking](ranking.md) adds pairwise/lambda CPU geometry and
fixed-step recipes with validation NDCG selection. Real A4 evaluation remains open.

[Quantile and penalized leaves](quantile.md) expose routed residuals/original
weights and compose all three CPU growth policies. Real A5 evaluation remains open.

[Poisson counts and exposure](poisson.md) add a CPU count recipe with explicit
rate/count outputs. Real A7 evaluation remains open.

[Gamma positive-target means](gamma.md) add weighted CPU mean regression.
Real A8 quality and distributional calibration remain unverified.

[Tweedie nonnegative means](tweedie.md) support fixed-power CPU fitting and
explicit annualized-loss weight semantics. Real A9 evaluation remains open.

[Frequency–severity composition](frequency-severity.md) binds matched paid-loss aggregates
and persists two-model inference with explicit output units. Real A9 evaluation remains open.

[Log-normal AFT](aft.md) adds event/right-censored CPU training and
persisted scale-aware survival outputs. Real A10 evaluation remains open.

[Multi-output squared regression](multioutput.md) supports independent/shared trees,
projected splits and persisted training-only target scaling. Real A6 evaluation remains open.

[Shared training preparation](preparation.md) reuses fitted CPU binning/codes
across independent jobs, verified at M=1/8/32.
[Independent validation stopping](stopping.md) separates outer-round patience
from model acceptance and strict best-model selection across all CPU recipes.

[Public development extensions](extensions.md) exercise installed cohort split
constraints and external penalized leaves, with core inference after plugin removal.


[Experimental CUDA storage](execution.md) now provides explicit context-owned buffers,
upload/copy/export and lifetime checks verified on a real T4. Training recipes
remain CPU-only; device fields, trees and boosting are not implemented yet.
