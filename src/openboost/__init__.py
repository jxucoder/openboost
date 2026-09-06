"""Programmable boosting foundation: initial CPU records and state transactions.

Numeric inputs, mapped ensemble artifacts and explicit run state are public.
B04 adds composable numeric operations and depthwise tree inference.
The scalar squared recipe is implemented; Normal and CUDA execution are not yet.
"""

from .data import NumericData, Problem
from .runtime import RunContext

__all__ = ("NumericData", "Problem", "RunContext")
