"""110 independent high-precision softmax oracle; no production or CUDA imports."""

from decimal import Decimal, localcontext

import numpy as np

RTOL, ATOL = 1e-4, 1e-5
GEOMETRY_RTOL, GEOMETRY_ATOL = 1e-6, 1e-8
METRIC_SCALE = 1e-3

# Every row must remain supported regardless of its training weight.
DOMAIN_CASES = (
    ("uniform", (0, 0, 0), 0, True),
    ("positive_tail", (80, 0, 0), 0, True),
    ("wrong_class_tail", (80, 0, 0), 1, True),
    ("tied_maxima", (40, 40, 0), 1, True),
    ("shifted_tail", (880, 800, 800), 0, True),
    ("tail_extinction", (120, 0, 0), 0, False),
    ("one_extinct_class", (0, 0, -120), 0, False),
)


def geometry(raw, target, offset, weight, *, stored=True):
    """Direct Decimal exp/sum/log and exact Hessian, independent of stable kernels."""
    dtype = np.float32 if stored else np.float64
    with np.errstate(over="raise", invalid="raise"):
        try:
            raw, offset, weight = (np.asarray(a, dtype=dtype) for a in (raw, offset, weight))
        except FloatingPointError as error:
            raise ValueError("finite stored inputs required") from error
    codes = np.asarray(target)
    if (
        raw.ndim != 2
        or raw.shape[1] < 2
        or offset.shape != raw.shape
        or codes.shape != (len(raw),)
        or weight.shape != (len(raw),)
        or any(not np.isfinite(a).all() for a in (raw, offset, weight, codes))
        or np.any(codes != np.floor(codes))
        or np.any(codes < 0)
        or np.any(codes >= raw.shape[1])
        or np.any(weight < 0)
        or weight.sum() <= 0
    ):
        raise ValueError("aligned multiclass inputs required")
    probabilities, gradients, bounds, hessians, losses = [], [], [], [], []
    with localcontext() as ctx:
        ctx.prec = 180
        for r, code in enumerate(codes.astype(int)):
            values = [
                Decimal(float(a)) + Decimal(float(b))
                for a, b in zip(raw[r], offset[r], strict=True)
            ]
            exponentials = [(v - max(values)).exp() for v in values]
            total = sum(exponentials)
            p = [v / total for v in exponentials]
            probabilities.append([float(v) for v in p])
            gradients.append([float(v - int(j == code)) for j, v in enumerate(p)])
            bounds.append([float(2 * v * (1 - v)) for v in p])
            hessians.append(
                [[float(v * (int(i == j) - q)) for j, q in enumerate(p)] for i, v in enumerate(p)]
            )
            losses.append(total.ln() + max(values) - values[code])
        loss = float(
            sum(Decimal(float(w)) * value for w, value in zip(weight, losses, strict=True))
            / sum(Decimal(float(w)) for w in weight)
        )
    with np.errstate(over="ignore", under="ignore"):
        g, h = np.asarray(gradients, dtype=dtype), np.asarray(bounds, dtype=dtype)
    if not np.isfinite(g).all() or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("multiclass geometry outside stored support")
    return loss, g, h, np.asarray(probabilities), np.asarray(hessians)


def base(target, width):
    if type(width) is not int or width < 2 or set(target) != set(range(width)):
        raise ValueError("every declared class required")
    return np.zeros(width, np.float32)


def rounds(train, validation, *, depth, count=2, rate=0.25):
    """K original-row scalar trees from one stored snapshot, then one joint update."""
    from .device_rounds import predict, tree

    initial = base(train["target"], train["offset"].shape[1])
    raw = np.broadcast_to(initial, train["offset"].shape).copy()
    val = np.broadcast_to(initial, validation["offset"].shape).copy()
    steps = []
    for _ in range(count):
        _, g, h, _, _ = geometry(raw, train["target"], train["offset"], train["weight"])
        fields = [
            np.column_stack((g[:, j], h[:, j])) * np.asarray(train["weight"], np.float32)[:, None]
            for j in range(len(initial))
        ]
        trees = [tree(train["x"], field, depth) for field in fields]
        before = raw.copy()
        for j, nodes in enumerate(trees):
            for node in nodes:
                node["value"] = float(np.float32(node["value"]))
            for data, values in ((train, raw), (validation, val)):
                prediction = predict(nodes, data["x"]).astype(np.float32)
                values[:, j] = values[:, j] + np.float32(rate) * prediction
        loss = geometry(raw, train["target"], train["offset"], train["weight"])[0]
        score = geometry(val, validation["target"], validation["offset"], validation["weight"])[0]
        steps.append(
            dict(
                gradient=g,
                bound=h,
                fields=fields,
                trees=trees,
                before=before,
                raw=raw.copy(),
                validation_raw=val.copy(),
                loss=loss,
                score=score,
            )
        )
    return initial, steps
