"""Programmable boosting foundation: initial CPU records and state transactions.

Numeric inputs, mapped ensemble artifacts and explicit run state are public.
B04 adds composable numeric operations and depthwise tree inference.
Squared, joint Normal and Formula CPU recipes plus sequential runs are public.
CUDA execution is not implemented yet.
"""

from .data import NumericData, Problem
from .runtime import RunContext

__all__ = ("NumericData", "Problem", "RunContext")
