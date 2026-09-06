"""Train-only numeric CPU quantiles and immutable feature-major bin codes."""

from dataclasses import dataclass, field

import numpy as np

from .data import NumericData, _identity


def _array(value, dtype):
    a = np.asarray(value, dtype=dtype, order="C")
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class NumericBinning:
    feature_names: tuple[str, ...]
    cuts: tuple[np.ndarray, ...]
    identity: str = field(init=False)

    def __post_init__(self):
        names = tuple(self.feature_names)
        cuts = tuple(_array(c, "<f8") for c in self.cuts)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n for n in names)
            or len(cuts) != len(names)
        ):
            raise ValueError("cuts must match unique feature names")
        if any(c.ndim != 1 or not np.isfinite(c).all() or np.any(np.diff(c) <= 0) for c in cuts):
            raise ValueError("finite strictly increasing cuts required")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "cuts", cuts)
        object.__setattr__(self, "identity", _identity("numeric-binning-v1", names, *cuts))

    @classmethod
    def fit(cls, data, *, bins=254):
        if not isinstance(data, NumericData):
            raise ValueError("numeric CPU data required")
        if type(bins) is not int or not 1 <= bins <= np.iinfo(np.int32).max:
            raise ValueError("positive int32 bin capacity required")
        cuts = []
        for column in data.values.T:
            observed = column[~np.isnan(column)]
            q = (
                np.quantile(observed, np.arange(1, bins) / bins, method="linear")
                if len(observed)
                else np.array([])
            )
            if not np.isfinite(q).all():
                raise ValueError("nonfinite interpolated cuts; rescale numeric features")
            cuts.append(
                np.unique(q[(q >= observed.min()) & (q < observed.max())]) if len(observed) else q
            )
        return cls(data.feature_names, tuple(cuts))

    def transform(self, data):
        return BinnedData(data, self)


@dataclass(frozen=True, eq=False)
class BinnedData:
    data: NumericData
    binning: NumericBinning
    codes: np.ndarray = field(init=False)
    missing: np.ndarray = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self):
        if (
            not isinstance(self.data, NumericData)
            or not isinstance(self.binning, NumericBinning)
            or self.data.feature_names != self.binning.feature_names
        ):
            raise ValueError("data and binning schema differ")
        missing = np.isnan(self.data.values.T)
        codes = np.stack(
            [
                np.where(missing[f], 0, np.searchsorted(cuts, self.data.values[:, f], side="left"))
                for f, cuts in enumerate(self.binning.cuts)
            ]
        )
        object.__setattr__(self, "codes", _array(codes, "<i4"))
        object.__setattr__(self, "missing", _array(missing, bool))
        object.__setattr__(self, "identity", _identity(self.data.identity, self.binning.identity))
