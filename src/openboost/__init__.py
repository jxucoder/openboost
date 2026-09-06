"""Programmable boosting foundation: initial CPU records and state transactions.

Numeric inputs, mapped ensemble artifacts and explicit run state are public.
Numeric/categorical operations support depthwise, best-first and symmetric growth.
Squared, Normal, Formula and binary CPU recipes plus sequential runs are public.
CUDA execution is not implemented yet.
"""

from .data import ClassSchema, MixedData, NumericData, Problem
from .runtime import RunContext

__all__ = ("ClassSchema", "MixedData", "NumericData", "Problem", "RunContext")
