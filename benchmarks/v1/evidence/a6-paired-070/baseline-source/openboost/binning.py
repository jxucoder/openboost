"""Train-only numeric quantiles/category dictionaries and feature-major CPU codes."""

from dataclasses import dataclass, field

import numpy as np

from .data import MixedData, NumericData, _identity, category_token


def _array(value, dtype):
    a = np.asarray(value, dtype=dtype, order="C")
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class Binning:
    feature_names: tuple[str, ...]
    cuts: tuple[np.ndarray, ...]
    categories: tuple[tuple | None, ...] | None = None
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
        categories = (None,) * len(names) if self.categories is None else tuple(self.categories)
        if len(categories) != len(names):
            raise ValueError("category dictionaries must match feature schema")
        normalized = []
        for f, values in enumerate(categories):
            if values is None:
                normalized.append(None)
                continue
            if isinstance(values, (str, bytes)):
                raise ValueError("category dictionary must be a token sequence")
            tokens = tuple(category_token(v) for v in values)
            if (
                any(v is None for v in tokens)
                or len({type(v) for v in tokens}) > 1
                or len(set(tokens)) != len(tokens)
                or tokens != tuple(sorted(tokens))
                or len(cuts[f])
                or len(tokens) > np.iinfo(np.int32).max
            ):
                raise ValueError("sorted unique typed categories and no numeric cuts required")
            normalized.append(tokens)
        categories = tuple(normalized)
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "cuts", cuts)
        object.__setattr__(
            self, "identity", _identity("mixed-binning-v1", names, categories, *cuts)
        )

    @classmethod
    def fit(cls, data, *, bins=254):
        if not isinstance(data, (NumericData, MixedData)):
            raise ValueError("numeric or mixed CPU data required")
        if type(bins) is not int or not 1 <= bins <= np.iinfo(np.int32).max:
            raise ValueError("positive int32 bin capacity required")
        cuts, categories = [], []
        for kind, column in zip(data.feature_kinds, data.values.T, strict=True):
            if kind == "categorical":
                present = [category_token(v) for v in column if category_token(v) is not None]
                categories.append(tuple(sorted(set(present))))
                cuts.append(np.array([]))
                continue
            categories.append(None)
            column = np.asarray(column, dtype=float)
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
        return cls(data.feature_names, tuple(cuts), tuple(categories))

    @property
    def feature_kinds(self):
        return tuple("numeric" if c is None else "categorical" for c in self.categories)

    @property
    def bin_counts(self):
        return tuple(
            len(cut) + 1 if cat is None else max(1, len(cat))
            for cut, cat in zip(self.cuts, self.categories, strict=True)
        )

    def transform(self, data):
        return BinnedData(data, self)


@dataclass(frozen=True, eq=False)
class BinnedData:
    data: NumericData | MixedData
    binning: Binning
    codes: np.ndarray = field(init=False)
    missing: np.ndarray = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self):
        if (
            not isinstance(self.data, (NumericData, MixedData))
            or not isinstance(self.binning, Binning)
            or self.data.feature_names != self.binning.feature_names
            or self.data.feature_kinds != self.binning.feature_kinds
        ):
            raise ValueError("data and binning schema differ")
        codes, missing = [], []
        for f, column in enumerate(self.data.values.T):
            categories = self.binning.categories[f]
            if categories is None:
                column = np.asarray(column, dtype=float)
                mask = np.isnan(column)
                code = np.where(mask, 0, np.searchsorted(self.binning.cuts[f], column, side="left"))
            else:
                tokens = tuple(category_token(v) for v in column)
                mapping = {token: i for i, token in enumerate(categories)}
                mask = np.array([token not in mapping for token in tokens])
                code = np.array([mapping.get(token, 0) for token in tokens])
            codes.append(code)
            missing.append(mask)
        object.__setattr__(self, "codes", _array(codes, "<i4"))
        object.__setattr__(self, "missing", _array(missing, bool))
        object.__setattr__(self, "identity", _identity(self.data.identity, self.binning.identity))


@dataclass(frozen=True, eq=False)
class PreparedData:
    """Train-only fitted binning/codes bound to immutable feature data and config."""

    data: NumericData | MixedData
    bins: int = 254
    binned: BinnedData = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self):
        binned = Binning.fit(self.data, bins=self.bins).transform(self.data)
        object.__setattr__(self, "binned", binned)
        object.__setattr__(
            self, "identity", _identity("prepared-cpu-v1", self.bins, binned.identity)
        )


def prepare_training(data, *, bins=254, prepared=None):
    """Resolve fresh or explicitly shared preparation; never refit a supplied object."""
    if prepared is None:
        prepared = PreparedData(data, bins)
    if (
        not isinstance(prepared, PreparedData)
        or not isinstance(data, (NumericData, MixedData))
        or type(bins) is not int
        or prepared.bins != bins
        or prepared.data.identity != data.identity
    ):
        raise ValueError("prepared training data/config identity differs")
    return prepared.binned
