"""092 original-row trajectories with separate training/best comparisons.

The frozen 090 geometry, exhaustive row partitioner and prediction functions are
reused unchanged. Only the loop's comparison policy and best anchor differ. This
is a float64 mathematical reference, not CUDA emulation or production code.
"""

import numpy as np

from .device_normal import base, direction, fixture, geometry
from .device_rounds import predict, tree
from .normal_comparison import compare


def trial(raw, delta, target, offset, weight, *, rate=0.1, fixed=False):
    """Actual stored-input comparison and six halving trials; metrics stay truthful."""
    geometry(raw, target, offset, weight)
    attempts = []
    for j in range(1 if fixed else 6):
        coefficient = rate * 0.5**j
        try:
            with np.errstate(over="raise", invalid="raise"):
                candidate = raw + coefficient * delta
            value = geometry(candidate, target, offset, weight)[0]
        except (ValueError, OverflowError, FloatingPointError):
            attempts.append((coefficient, None, "invalid"))
            if fixed:
                raise
            continue
        change = compare(raw, candidate, target, offset, weight)
        accepted = fixed or change.improves()
        attempts.append((coefficient, value, "accepted" if accepted else "rejected"))
        if accepted:
            return candidate, tuple(attempts)
    return raw, tuple(attempts)


def rounds(
    case,
    *,
    mode="natural",
    damping=0.0,
    update="joint",
    depth=2,
    minimum=None,
    count=3,
    rate=0.1,
    fixed=False,
):
    """Record complete tiny trajectories; best selection is separate from acceptance."""
    groups = {"joint": ((0, 1),), "forward": ((0,), (1,)), "reverse": ((1,), (0,))}[update]
    f = fixture(case)
    initial = base(f["target"], f["offset"], f["weight"])
    raw = np.broadcast_to(initial, f["offset"].shape).copy()
    validation = np.broadcast_to(initial, f["validation_offset"].shape).copy()

    def validation_loss(values):
        return geometry(
            values, f["validation_target"], f["validation_offset"], f["validation_weight"]
        )[0]

    best_score, best_terms, version, nterms = validation_loss(validation), 0, 0, 0
    best_raw = validation.copy()
    steps = []
    for round_index in range(count):
        for channels in groups:
            loss, gradient, fisher = geometry(raw, f["target"], f["offset"], f["weight"])
            z = direction(gradient, fisher, mode=mode, damping=damping)
            learners, fields = [], []
            delta, validation_delta = np.zeros_like(raw), np.zeros_like(validation)
            for k in channels:
                values = np.column_stack((-f["weight"] * z[:, k], f["weight"], f["information"]))
                nodes = tree(f["x"], values, depth, minimum)
                fields.append(values)
                learners.append(nodes)
                delta[:, k] = predict(nodes, f["x"])
                validation_delta[:, k] = predict(nodes, f["validation_x"])
            after, attempts = trial(
                raw, delta, f["target"], f["offset"], f["weight"], rate=rate, fixed=fixed
            )
            accepted = after is not raw
            if accepted:
                validation = validation + attempts[-1][0] * validation_delta
                version, nterms = version + 1, nterms + len(channels)
                value = validation_loss(validation)
                change = compare(
                    best_raw,
                    validation,
                    f["validation_target"],
                    f["validation_offset"],
                    f["validation_weight"],
                )
                if change.improves():
                    best_score, best_terms = value, nterms
                    best_raw = validation.copy()
            steps.append(
                dict(
                    round=round_index,
                    channels=channels,
                    gradient=gradient,
                    fisher=fisher,
                    direction=z,
                    fields=fields,
                    nodes=learners,
                    before=raw.copy(),
                    raw=after.copy(),
                    loss_before=loss,
                    loss=geometry(after, f["target"], f["offset"], f["weight"])[0],
                    validation_raw=validation.copy(),
                    validation_score=validation_loss(validation),
                    accepted=accepted,
                    attempts=attempts,
                    version=version,
                    nterms=nterms,
                    best_terms=best_terms,
                    best_score=best_score,
                    best_raw=best_raw.copy(),
                )
            )
            raw = after
    return initial, steps
