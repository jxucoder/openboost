"""Programmable boosting foundation: initial CPU records and state transactions.

B03 supplies numeric inputs, constant-term artifacts and explicit run state.
B04 adds composable numeric operations and depthwise tree inference.
Boosting recipes and CUDA execution are not implemented yet.
"""

from .data import NumericData, Problem
from .runtime import RunContext

__all__ = ("NumericData", "Problem", "RunContext")
