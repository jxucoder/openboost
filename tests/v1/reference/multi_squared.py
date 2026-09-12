"""118 rational stored-input mathematics and original-row recipe oracle."""

from fractions import Fraction as F

import numpy as np

from . import device_vector as vector


def q(value):
    return F(float(value))


def stored(value):
    """Nearest-even binary32 from an exact rational, including midpoint controls."""
    guess = np.float32(float(value))
    if not np.isfinite(guess):
        raise ValueError("outside finite binary32")
    neighbors = (
        np.nextafter(guess, np.float32(-np.inf)),
        guess,
        np.nextafter(guess, np.float32(np.inf)),
    )
    return min(
        (x for x in neighbors if np.isfinite(x)),
        key=lambda x: (abs(q(x) - value), int(x.view(np.uint32)) % 2),
    )


def loss(raw, target, offset, weight):
    total = F()
    for r in range(len(raw)):
        for k in range(raw.shape[1]):
            residual = q(raw[r, k]) + q(offset[r, k]) - q(target[r, k])
            total += q(weight[r]) * residual**2 / 2
    return total / sum(map(q, weight))


def change(before, after, target, offset, weight):
    total = F()
    for r in range(len(before)):
        for k in range(before.shape[1]):
            delta = q(after[r, k]) - q(before[r, k])
            residual = q(before[r, k]) + q(offset[r, k]) - q(target[r, k])
            total += q(weight[r]) * delta * (residual + delta / 2)
    return total / sum(map(q, weight))


def mapped(raw, prediction, mapping, coefficient):
    output = np.empty_like(raw, dtype=np.float32)
    for r in range(len(raw)):
        for k in range(raw.shape[1]):
            total = np.float32(0)
            for channel in range(prediction.shape[1]):
                product = stored(q(prediction[r, channel]) * q(mapping[channel, k]))
                total = stored(q(total) + q(product))
            output[r, k] = stored(q(raw[r, k]) + q(stored(q(coefficient) * q(total))))
    return output


def cases():
    result = []
    for width in (1, 2, 4):
        for kind in (
            "improve",
            "worsen",
            "stationary",
            "tiny",
            "identity",
            "zero_weight",
            "cancel",
            "offset",
        ):
            old = np.zeros((2, width), np.float32)
            new, offset = old.copy(), old.copy()
            target, weight = np.ones_like(old), np.array([1, 2], np.float32)
            if kind == "improve":
                new[:] = 0.5
            elif kind == "worsen":
                new[:] = -0.5
            elif kind in ("stationary", "tiny"):
                target[:] = 0
                new[-1, -1] = 1e-20 if kind == "tiny" else 0.25
            elif kind == "zero_weight":
                weight[-1] = 0
                new[-1] = 100
            elif kind == "cancel":
                new[:] = 2  # identical squared residuals with distinct stored raw
            elif kind == "offset":
                offset[:] = 0.25
                new[:] = 0.5
            result.append(dict(id=f"k{width}-{kind}", arrays=(old, new, target, offset, weight)))
    # A large unchanged channel hides a tiny strictly worse second channel in reporting.
    old = np.zeros((1, 2), np.float32)
    for small in (1e-10, 1e-20, 1e-30):
        result.append(
            dict(
                id=f"reporting-tie-{small}",
                arrays=(
                    old.copy(),
                    np.array([[0, small]], np.float32),
                    np.array([[1e10, 0]], np.float32),
                    old.copy(),
                    np.ones(1, np.float32),
                ),
            )
        )
    return tuple(result)


CASES = cases()


def fixture(width=2, conflict=False):
    source = vector.fixture(width)
    offset = np.tile(np.arange(8, dtype=np.float32)[:, None] / 8, (1, width))
    target = -source["g"].astype(np.float32) + offset
    return dict(
        x=source["x"],
        target=target,
        offset=offset,
        weight=source["weight"].astype(np.float32),
        validation_x=source["x"][::-1].copy(),
        validation_offset=offset[::-1] / 2,
        validation_target=(source["g"] if conflict else -source["g"])[::-1].astype(np.float32)
        + offset[::-1] / 2,
        validation_weight=np.ones(8, np.float32),
    )


def rounds(width=2, mode="shared", depth=1, step="fixed", count=2, conflict=False, rate=0.5):
    f = fixture(width, conflict)
    mass = sum(map(q, f["weight"]))
    base = np.array(
        [
            stored(
                sum(
                    q(w) * (q(y[k]) - q(o[k]))
                    for w, y, o in zip(f["weight"], f["target"], f["offset"], strict=True)
                )
                / mass
            )
            for k in range(width)
        ]
    )
    raw = np.tile(base, (8, 1))
    validation, best = raw.copy(), raw.copy()
    best_prefix, history = 0, []
    for i in range(count):
        g = np.array(
            [
                [
                    stored(q(raw[r, k]) + q(f["offset"][r, k]) - q(f["target"][r, k]))
                    for k in range(width)
                ]
                for r in range(8)
            ]
        )
        h = np.ones_like(g)
        terms = []
        for channel in range(width) if mode == "independent" else (None,):
            leaf_g, leaf_h = (
                (g[:, channel : channel + 1], h[:, channel : channel + 1])
                if channel is not None
                else (g, h)
            )
            split_g, split_h = (g[:, :1], h[:, :1]) if mode == "projected" else (leaf_g, leaf_h)
            source = dict(
                x=f["x"], weight=f["weight"], g=leaf_g, h=leaf_h, split_g=split_g, split_h=split_h
            )
            nodes = vector.tree(source, depth)
            mapping = np.eye(width, dtype=np.float32)
            if channel is not None:
                mapping = mapping[channel : channel + 1]
            terms.append((nodes, mapping))
        trials = []
        for attempt in range(1 if step == "fixed" else 6):
            coefficient = np.float32(rate * 0.5**attempt)
            candidate, candidate_validation = raw.copy(), validation.copy()
            for nodes, mapping in terms:
                candidate = mapped(
                    candidate,
                    vector.predict(nodes, f["x"]).astype(np.float32),
                    mapping,
                    coefficient,
                )
                candidate_validation = mapped(
                    candidate_validation,
                    vector.predict(nodes, f["validation_x"]).astype(np.float32),
                    mapping,
                    coefficient,
                )
            delta = change(raw, candidate, f["target"], f["offset"], f["weight"])
            accepted = step == "fixed" or delta < 0
            trials.append((float(coefficient), accepted, delta))
            if accepted:
                raw, validation = candidate, candidate_validation
                if (
                    change(
                        best,
                        validation,
                        f["validation_target"],
                        f["validation_offset"],
                        f["validation_weight"],
                    )
                    < 0
                ):
                    best, best_prefix = validation.copy(), (i + 1) * len(terms)
                break
        history.append(
            dict(
                gradient=g,
                terms=terms,
                trials=trials,
                raw=raw.copy(),
                validation=validation.copy(),
                best=best.copy(),
                best_prefix=best_prefix,
            )
        )
    return base, history
