"""Positive-payment frequency/severity problems and explicit two-model inference."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import Model
from .data import MixedData, NumericData, Problem, _identity, _owned
from .outputs import poisson_mean, positive_mean


def paid_loss_problems(data, paid_count, total, exposure, *, weight=None):
    """Bind policy aggregates; counts must count the same positive payments as total.

    The caller performs eligibility filtering and joins. Zero counts require zero
    total; positive counts require positive total. Severity uses count-weighted
    policy averages on positive-count rows, with the same policy predictors.
    """
    if not isinstance(data, (NumericData, MixedData)):
        raise ValueError("policy feature data required")
    count, amount, e = (_owned(v, ndim=1) for v in (paid_count, total, exposure))
    if (
        count.shape != amount.shape
        or count.shape != e.shape
        or len(count) != len(data.row_ids)
        or np.any(count < 0)
        or np.any(count != np.floor(count))
        or np.any(amount < 0)
        or np.any(e <= 0)
        or np.any((count == 0) != (amount == 0))
    ):
        raise ValueError("aligned paid counts/totals and positive exposure required")
    frequency = Problem(
        data, count[:, None], data.row_ids, weight=weight, structure={"exposure": e[:, None]}
    )
    keep = count > 0
    if not np.any(keep & (frequency.weight > 0)):
        raise ValueError("positive-weight paid policies required for severity")
    subset = (
        MixedData(data.values[keep], data.row_ids[keep], data.feature_names, data.feature_kinds)
        if isinstance(data, MixedData)
        else NumericData(data.values[keep], data.row_ids[keep], data.feature_names)
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        severity = Problem(
            subset,
            (amount[keep] / count[keep])[:, None],
            subset.row_ids,
            weight=frequency.weight[keep] * count[keep],
        )
    if np.any(severity.target <= 0):
        raise ValueError("positive paid severity underflows float64")
    return frequency, severity


@dataclass(frozen=True, eq=False)
class FrequencySeverity:
    """Declared positive-payment log-rate and log-severity raw models.

    This declaration cannot prove the models' training targets; use matched
    paid-loss problems. No output is a full aggregate distribution.
    """

    frequency: Model
    severity: Model

    def __post_init__(self):
        if any(
            not isinstance(m, Model) or len(m.base) != 1 or m.classes is not None
            for m in (self.frequency, self.severity)
        ):
            raise ValueError("two scalar regression models required")

    def predict(
        self,
        frequency_data,
        severity_data,
        exposure,
        *,
        frequency_offset=None,
        severity_offset=None,
    ):
        if not np.array_equal(frequency_data.row_ids, severity_data.row_ids):
            raise ValueError("frequency/severity policy row IDs must align in order")
        frequency = poisson_mean(
            self.frequency.predict(frequency_data, offset=frequency_offset), exposure
        )
        severity = positive_mean(self.severity.predict(severity_data, offset=severity_offset))
        with np.errstate(over="raise", invalid="raise"):
            annualized = frequency["rate"] * severity
            period = frequency["count_mean"] * severity
        if np.any(annualized <= 0) or np.any(period <= 0):
            raise ValueError("composed positive means underflow float64")
        return dict(
            paid_count_rate=frequency["rate"],
            paid_count_mean=frequency["count_mean"],
            severity_mean=severity,
            annualized_mean=_owned(annualized, ndim=1),
            period_mean=_owned(period, ndim=1),
        )

    def record(self):
        return dict(
            format="openboost-frequency-severity-v1",
            frequency=self.frequency.record(),
            severity=self.severity.record(),
        )

    @property
    def identity(self):
        return _identity(self.record())

    def save(self, path):
        Path(path).write_text(json.dumps(self.record(), allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate composition artifact field")
                result[key] = value
            return result

        record = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
        if (
            not isinstance(record, dict)
            or set(record) != {"format", "frequency", "severity"}
            or record["format"] != "openboost-frequency-severity-v1"
        ):
            raise ValueError("unsupported composition artifact")
        return cls(Model.from_record(record["frequency"]), Model.from_record(record["severity"]))
