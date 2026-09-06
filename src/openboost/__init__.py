"""Programmable boosting foundation: initial CPU records and state transactions.

Numeric inputs, mapped ensemble artifacts and explicit run state are public.
Numeric/categorical operations support depthwise, best-first and symmetric growth.
Squared, joint Normal and Formula CPU recipes plus sequential runs are public.
CUDA execution is not implemented yet.
"""

from .data import MixedData, NumericData, Problem
from .runtime import RunContext

__all__ = ("MixedData", "NumericData", "Problem", "RunContext")
