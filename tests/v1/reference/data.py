"""Immutable column oracles; no production layout, identity or binding machinery."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _numeric(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or np.any(np.isinf(result)):
        raise ValueError("numeric column must be one-dimensional without infinity")
    return result


@dataclass(frozen=True)
class NumericBinning:
    cuts: tuple[float, ...]

    @classmethod
    def fit(cls, values, *, bins=254):
        column = _numeric(values)
        if not len(column):
            raise ValueError("training column must not be empty")
        if not isinstance(bins, Integral) or isinstance(bins, bool) or bins < 1:
            raise ValueError("bins must be a positive integer")
        observed = sorted(float(x) for x in column if not np.isnan(x))
        if not observed:
            return cls(())
        cuts = set()
        # Direct order-statistic interpolation, deliberately no np.quantile call.
        for j in range(1, bins):
            position = (len(observed) - 1) * j / bins
            lo = int(position)
            fraction = position - lo
            hi = min(lo + 1, len(observed) - 1)
            cut = (1 - fraction) * observed[lo] + fraction * observed[hi]
            if observed[0] <= cut < observed[-1]:
                cuts.add(cut)
        return cls(tuple(sorted(cuts)))

    def transform(self, values):
        column = _numeric(values)
        missing = tuple(bool(np.isnan(x)) for x in column)
        codes = tuple(
            0 if m else sum(x > cut for cut in self.cuts)
            for x, m in zip(column, missing, strict=True)
        )
        return codes, missing


def _token(value):
    if value is None or isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    raise ValueError("tokens must be strings or integers; missing is None/NaN")


@dataclass(frozen=True)
class CategoryMap:
    values: tuple[str | int, ...]

    @classmethod
    def fit(cls, values):
        tokens = [_token(v) for v in values]
        present = [v for v in tokens if v is not None]
        if len({type(v) for v in present}) > 1:
            raise ValueError("mixed category token types are not supported")
        return cls(tuple(sorted(set(present))))

    def transform(self, values):
        tokens = [_token(v) for v in values]
        mapping = {v: i for i, v in enumerate(self.values)}
        missing = tuple(v not in mapping for v in tokens)
        return tuple(mapping.get(v, 0) for v in tokens), missing

    def route(self, values, category, *, missing_left):
        category = _token(category)
        if category not in self.values or not isinstance(missing_left, bool):
            raise ValueError("route needs a fitted category and boolean missing direction")
        codes, missing = self.transform(values)
        selected = self.values.index(category)
        return tuple(
            missing_left if m else code == selected for code, m in zip(codes, missing, strict=True)
        )
