"""090 original-row Normal math and transactions, without production imports.

Small float64 semantic oracle, not a CUDA emulator or an author evaluation.
Tree candidates are exhaustively partitioned from original rows, not histograms.
"""

import math

import numpy as np

from .device_rounds import fixture as scalar_fixture
from .device_rounds import predict, tree


def fixture(case):
    if case not in ("weighted", "weighted_ties", "d2", "conflict"):
        raise ValueError("unknown Normal fixture")
    f = scalar_fixture("weighted" if case == "weighted_ties" else case)
    if case == "weighted":
        # Keep row 4 at zero weight, but give the finite extreme row positive
        # mass so mirrored missing routes have a distinguishable training score.
        f["weight"][0] = 0.5
        f["x"] = f["x"][:, :1]
        f["validation_x"] = f["validation_x"][:, :1]
    n = len(f["target"])
    f["offset"] = np.column_stack((f["offset"], (np.arange(n) % 3 - 1) / 8))
    f["validation_offset"] = np.column_stack((f["validation_offset"], (np.arange(n) % 2 - 0.5) / 4))
    return f


def _positive_exp(value):
    result = math.exp(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Normal scale or precision outside positive finite support")
    return result


def base(target, offset, weight, *, minimum_scale=1e-6):
    if not math.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("positive finite initial scale floor required")
    mass = math.fsum(weight)
    relative = [
        float(w) * _positive_exp(-2 * float(off[1])) / mass
        for w, off in zip(weight, offset, strict=True)
    ]
    centered = [float(y) - float(off[0]) for y, off in zip(target, offset, strict=True)]
    mean = math.fsum(q * y for q, y in zip(relative, centered, strict=True)) / math.fsum(relative)
    variance = math.fsum(q * (y - mean) ** 2 for q, y in zip(relative, centered, strict=True))
    result = np.array([mean, math.log(max(math.sqrt(variance), minimum_scale))])
    geometry(np.broadcast_to(result, offset.shape), target, offset, weight)
    return result


def geometry(raw, target, offset, weight):
    gradients, fisher, losses = [], [], []
    for i in range(len(target)):
        mean = float(raw[i, 0]) + float(offset[i, 0])
        ell = float(raw[i, 1]) + float(offset[i, 1])
        _positive_exp(ell)
        precision = _positive_exp(-2 * ell)
        residual = mean - float(target[i])
        square = residual * residual * precision
        loss = ell + square / 2 + math.log(2 * math.pi) / 2
        g = (residual * precision, 1 - square)
        if not all(math.isfinite(v) for v in (loss, *g)):
            raise ValueError("nonfinite Normal geometry")
        gradients.append(g)
        fisher.append((precision, 2.0))
        losses.append(float(weight[i]) * loss)
    return math.fsum(losses) / math.fsum(weight), np.array(gradients), np.array(fisher)


def direction(gradient, fisher, *, mode="natural", damping=0.0):
    if mode not in ("ordinary", "natural") or not math.isfinite(damping) or damping < 0:
        raise ValueError("valid direction mode and nonnegative finite damping required")
    if mode == "ordinary" and damping != 0:
        raise ValueError("ordinary direction does not use damping")
    result = np.empty_like(gradient)
    for i in range(len(gradient)):
        for k in range(2):
            denominator = 1 if mode == "ordinary" else float(fisher[i, k]) + damping
            result[i, k] = -float(gradient[i, k]) / denominator
    return result


def trial(raw, delta, target, offset, weight, *, rate=0.1, fixed=False):
    """Actual Normal loss, strict improvement and six halving trials; no mocks."""
    before = geometry(raw, target, offset, weight)[0]
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
        accepted = fixed or value < before
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
                if value < best_score:
                    best_score, best_terms = value, nterms
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
                )
            )
            raw = after
    return initial, steps
