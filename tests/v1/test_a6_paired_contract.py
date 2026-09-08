"""A paired cost result requires exact complete artifacts from both fits."""

from copy import deepcopy

import pytest
from benchmarks.v1 import a6_resource_preflight as preflight


def result():
    return dict(
        passed=True,
        replay="exact",
        artifacts={
            name: name.encode()
            for name in (
                "fit/model.bin",
                "fit/predictions.npz",
                "fit/training.json",
                "replay.npz",
            )
        },
    )


def test_complete_pair():
    a = result()
    assert preflight.pair_matches([a, deepcopy(a)])
    assert not preflight.pair_matches([a])
    assert not preflight.pair_matches([a, a, a])


@pytest.mark.parametrize("name", result()["artifacts"])
@pytest.mark.parametrize("change", ["different", "missing", "empty"])
def test_reject_changed_or_absent_evidence(name, change):
    a, b = result(), result()
    if change == "missing":
        del b["artifacts"][name]
    elif change == "empty":
        a["artifacts"][name] = b["artifacts"][name] = b""
    else:
        b["artifacts"][name] += b"changed"
    assert not preflight.pair_matches([a, b])


@pytest.mark.parametrize("field,value", [("passed", False), ("replay", "mismatch")])
def test_reject_failed_execution(field, value):
    a, b = result(), result()
    b[field] = value
    assert not preflight.pair_matches([a, b])


@pytest.mark.parametrize("fails", [False, True])
def test_pair_dispatch_and_failure_retention(monkeypatch, fails):
    calls = []

    def probe(spec, *, package_root):
        calls.append(package_root)
        value = result()
        value["passed"] = not fails
        return value

    monkeypatch.setattr(preflight, "probe", probe)
    pair = preflight.paired_probe({"id": "shared"})
    assert pair["passed"] is not fails
    assert calls == (["/baseline"] if fails else ["/baseline", None])
    assert pair["artifacts"]["baseline/fit/model.bin"] == b"fit/model.bin"
    assert pair["variants"][0]["variant"] == "baseline"
    if not fails:
        assert pair["artifacts"]["current/fit/model.bin"] == b"fit/model.bin"


def test_conflicting_modes_do_not_create_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="separate experiments"):
        preflight.main(output, tmp_path, profile=True, paired=True)
    assert not output.exists()
