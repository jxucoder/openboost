"""Produce development reference traces before isolated extension execution."""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
from tests.v1.reference.coupled import formula, formula_base, normal, normal_base, step


def main(destination):
    values = [[0], [1], [2], [3], [4], [None]]
    # Six bins preserve the five distinct numeric values in this tiny fixture.
    bins = np.array([[0], [1], [2], [3], [4], [np.nan]])
    y = np.array([0.5, 1, 2, 3, 5, 8])
    weight = np.array([1, 0, 2, 1, 3, 1])
    structure = np.linspace(0.2, 2, 6)
    records = []
    for family, mode in (("normal", "natural"), ("normal", "ordinary"), ("formula", "full")):
        for order in ((0, 1), (1, 0)):
            base = (
                normal_base(y, minimum_scale=1e-6, weight=weight)
                if family == "normal"
                else formula_base(y, weight=weight)
            )
            raw = np.broadcast_to(base, (6, 2)).copy()
            objective = normal if family == "normal" else partial(formula, x=structure)
            trace = []
            for _ in range(3):
                for channel in order:
                    update = step(
                        bins,
                        raw,
                        y,
                        objective,
                        weight=weight,
                        mode="ordinary" if mode == "ordinary" else "full",
                        damping=0.1 if family == "formula" else 0.0,
                        channels=(channel,),
                        rates=tuple(0.1 * 0.5**j for j in range(6)),
                    )
                    trace.append(
                        {
                            "channel": channel,
                            "raw_before": update.raw_before,
                            "raw_after": update.raw_after,
                            "accepted": update.accepted,
                            "coefficients": [t[0] for t in update.trials],
                        }
                    )
                    raw = np.array(update.raw_after)
            records.append(dict(family=family, mode=mode, order=order, trace=trace))
    result = dict(
        values=values,
        target=y.tolist(),
        weight=weight.tolist(),
        structure=structure.tolist(),
        records=records,
    )
    Path(destination).write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
