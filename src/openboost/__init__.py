"""Programmable boosting foundation: initial CPU records and state transactions.

Numeric inputs, mapped ensemble artifacts and explicit run state are public.
Numeric operations support depthwise, best-first and symmetric tree growth.
Squared, joint Normal and Formula CPU recipes plus sequential runs are public.
CUDA execution is not implemented yet.
"""

from .data import NumericData, Problem
from .runtime import RunContext

__all__ = ("NumericData", "Problem", "RunContext")
