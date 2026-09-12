"""Independent sequential outcomes from original-row squared trees and stop math."""

import numpy as np

from .device_rounds import fixture as source_fixture
from .device_rounds import loss, predict, tree


def case(index):
    source = source_fixture("weighted")
    scale = 2 ** ((index // 4) % 2)
    for target, offset in (("target", "offset"), ("validation_target", "validation_offset")):
        source[target] = (source[target] - source[offset]) * scale + source[offset]
    source["weight"] = source["weight"] * 2 ** ((index // 8) % 2)
    mode = index % 4
    options = dict(rounds=(2, 4, 3, 2)[mode], learning_rate=(.5, .5, 0., 256.)[mode],
                   max_depth=1, step="backtracking" if mode == 3 else "fixed",
                   max_trials=1, patience=(None, 1, 2, None)[mode],
                   min_delta=100. if mode == 1 else 0.)
    return source, options


def run(index):
    f, options = case(index)
    base = sum(f["weight"] * (f["target"] - f["offset"])) / sum(f["weight"])
    raw, valid = (np.full(len(f[k]), base) for k in ("target", "validation_target"))

    def training(value):
        return loss(f["target"], f["offset"], f["weight"], value)

    def validation(value):
        return loss(f["validation_target"], f["validation_offset"], f["validation_weight"], value)

    best, anchor, best_prefix, stale, version = validation(valid), validation(valid), 0, 0, 0
    best_raw, best_valid, steps = raw.copy(), valid.copy(), []
    for _ in range(options["rounds"]):
        fields = np.column_stack(((raw + f["offset"] - f["target"]) * f["weight"], f["weight"]))
        nodes = tree(f["x"], fields, options["max_depth"])
        candidate = raw + options["learning_rate"] * predict(nodes, f["x"])
        accepted = options["step"] == "fixed" or training(candidate) < training(raw)
        if accepted:
            raw = candidate
            valid = valid + options["learning_rate"] * predict(nodes, f["validation_x"])
            version += 1
        score = validation(valid)
        if score < best:
            best, best_prefix = score, version
            best_raw, best_valid = raw.copy(), valid.copy()
        if anchor - score > options["min_delta"]:
            anchor, stale = score, 0
        else:
            stale += 1
        steps.append(dict(accepted=accepted, loss=training(raw), score=score, nodes=nodes))
        if options["patience"] is not None and stale >= options["patience"]:
            break
    reason = "patience" if options["patience"] is not None and stale >= options["patience"] else "budget"
    return dict(base=base, raw=raw, validation_raw=valid, best_raw=best_raw,
                best_validation_raw=best_valid, best_score=best, best_prefix=best_prefix,
                version=version, completed=len(steps), stale=stale, reason=reason, steps=steps)
