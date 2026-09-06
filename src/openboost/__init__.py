"""Programmable boosting foundation: initial CPU records and state transactions.

Numeric inputs, mapped ensemble artifacts and explicit run state are public.
Numeric/categorical operations support depthwise, best-first and symmetric growth.
Squared, Normal, Formula, binary and multiclass CPU recipes are public.
Vector leaves, separate split/leaf statistics and sequential runs are available.
Query-local pairwise/lambda ranking composes the same scalar tree operations.
Routed residual views support quantile and anchored penalized leaves.
Poisson counts use explicit exposure and rate/count transforms.
Gamma positive-target mean regression shares scalar Newton operations.
Fixed-power Tweedie supports nonnegative mean regression.
Frequency/severity composition preserves two-model output roles and units.
Fixed-scale AFT supports event/right censoring and persisted survival outputs.
Multi-output squared recipes support independent/shared topology and target scaling.
Explicit prepared training data can be shared across independent CPU runs.
CUDA execution is not implemented yet.
"""

from .data import ClassSchema, MixedData, NumericData, Problem
from .runtime import RunContext

__all__ = ("ClassSchema", "MixedData", "NumericData", "Problem", "RunContext")
