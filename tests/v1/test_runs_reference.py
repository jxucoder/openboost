"""Data identity and run isolation checked with actual tiny tree fits."""

import numpy as np
import pytest

from .reference.runs import (
    RunSpec,
    bind_identity,
    data_identity,
    derive_seed,
    run_many,
    select_best,
)


def test_identity_is_typed_content_not_object_or_shape():
    arguments = dict(
        row_ids=[10, 20],
        values=[[0], [1]],
        schema=["x"],
        transformer={"bins": 2, "cuts": [0.5], "categories": []},
    )
    identity = data_identity(**arguments)
    assert identity == data_identity(**arguments)
    for change in (
        dict(values=[[0], [2]]),
        dict(row_ids=[20, 10]),
        dict(schema=["other"]),
        dict(transformer={"bins": 3, "cuts": [0.5], "categories": []}),
    ):
        assert data_identity(**(arguments | change)) != identity
    assert data_identity(**(arguments | {"values": [["1"], ["2"]]})) != data_identity(
        **(arguments | {"values": [[1], [2]]})
    )
    assert data_identity(
        **(arguments | {"transformer": {"categories": ["a", "b"]}})
    ) != data_identity(**(arguments | {"transformer": {"categories": ["b", "a"]}}))
    values = np.array([[0.0], [1.0]])
    first = data_identity(**(arguments | {"values": values}))
    values[0] = 99
    assert data_identity(**(arguments | {"values": values})) != first


def test_binding_rejects_misaligned_roles_and_includes_all_fields():
    prepared = data_identity([10, 20], [[0], [1]], ["x"], {"cuts": [0.5]})
    target = ([10, 20], [[1], [2]])
    first = bind_identity(prepared, [10, 20], target=target, weight=([10, 20], [1, 1]))
    assert bind_identity(prepared, [10, 20], target=target, weight=([10, 20], [1, 2])) != first
    assert bind_identity(prepared, [10, 20], target=target, offset=([10, 20], [[0], [1]])) != first
    with pytest.raises(ValueError):
        bind_identity(prepared, [10, 20], target=([20, 10], [[2], [1]]))
    with pytest.raises(ValueError):
        data_identity([10, 10], [[0], [1]], ["x"], {})


def fixtures():
    bins = np.array([[0], [1], [2], [3]])
    y = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    specs = [
        RunSpec("scalar", 7, 5, 0.2, 2, 3),
        RunSpec("vector", 7, 3, 0.1, 3, 3),
        RunSpec("flat", 7, 8, 0.0, 1, 4),
    ]
    problems = {
        "scalar": (y, y),
        "vector": (np.column_stack((y, 2 * y)), np.column_stack((y, 2 * y))),
        "flat": (y, y),
    }
    return bins, specs, problems


def test_independent_sequential_reordered_and_regrouped_runs():
    bins, specs, problems = fixtures()
    together = run_many(specs, bins, bins, problems)
    reverse = run_many(list(reversed(specs)), bins, bins, problems)
    separate = {s.run_id: run_many([s], bins, bins, problems)[s.run_id] for s in specs}
    grouped = run_many(specs[::2], bins, bins, problems) | run_many(
        specs[1::2], bins, bins, problems
    )
    assert together == reverse == separate == grouped
    assert together["flat"].status == "early_stopped"
    assert together["flat"].rounds == 1 and together["flat"].best_round == 0
    assert together["scalar"].rounds == 5 and together["vector"].rounds == 3
    assert len(together["vector"].best_raw[0]) == 2


def test_failure_isolation_retry_and_best_snapshot_reconstruction():
    bins, specs, problems = fixtures()
    failed = RunSpec("broken", 7, 5, 0.2, 2, 3, fail_round=2)
    problems["broken"] = problems["scalar"]
    baseline = run_many(specs, bins, bins, problems)
    result = run_many([failed, *specs], bins, bins, problems)
    assert result["broken"].status == "failed" and "round 2" in result["broken"].error
    assert {k: result[k] for k in baseline} == baseline
    retry = RunSpec("broken", 7, 5, 0.2, 2, 3)
    successful = run_many([retry], bins, bins, problems)["broken"]
    assert successful.samples[:1] == result["broken"].samples
    for record in baseline.values():
        raw = np.tile(record.base, (len(bins), 1))
        for term in record.best_terms:
            for channel, tree in enumerate(term):
                raw[:, channel] += record.learning_rate * tree.predict(bins)
        np.testing.assert_allclose(raw, record.best_raw)


def test_validation_selection_restores_base_when_training_hurts_validation():
    bins = [[0], [1]]
    problems = {"a": ([[-1], [1]], [[1], [-1]]), "b": ([[-1], [1]], [[1], [-1]])}
    records = run_many(
        [RunSpec("b", 0, 5, 0.3, 2, 2), RunSpec("a", 0, 5, 0.1, 1, 2)], bins, bins, problems
    )
    for record in records.values():
        assert record.best_round == 0 and record.best_terms == ()
        np.testing.assert_array_equal(record.best_raw, [[0], [0]])
    assert select_best(records).run_id == "a"  # deterministic ID tie, not final train loss


def test_rng_key_is_structured_and_independent_of_global_state():
    key = derive_seed(7, "run-α", 2, "tree", "rows")
    np.random.seed(987)
    assert derive_seed(7, "run-α", 2, "tree", "rows") == key
    assert (
        len(
            {
                key,
                derive_seed(7, "run-β", 2, "tree", "rows"),
                derive_seed(7, "run-α", 3, "tree", "rows"),
                derive_seed(7, "run-α", 2, "tree", "columns"),
            }
        )
        == 4
    )
    assert derive_seed(1, "ab", 2, "c", "d") != derive_seed(1, "a", 2, "bc", "d")


def test_duplicate_run_ids_and_no_successful_selection_rejected():
    bins, specs, problems = fixtures()
    with pytest.raises(ValueError):
        run_many([specs[0], specs[0]], bins, bins, problems)
    failed = run_many([RunSpec("missing", 0, 1, 0.1, 1, 4)], bins, bins, problems)
    assert failed["missing"].status == "failed"
    with pytest.raises(ValueError):
        select_best(failed)


def test_fixed_rng_derivation_fixture():
    assert derive_seed(7, "run-α", 2, "tree", "rows") == 6175955064790668999


def test_selection_rejects_incomparable_targets_and_changed_data_identity():
    bins, specs, problems = fixtures()
    records = run_many(specs, bins, bins, problems)
    with pytest.raises(ValueError):
        select_best(records)
    original = records["scalar"].problem_id
    changed = bins.copy()
    changed[-1] = 10
    assert run_many([specs[0]], changed, bins, problems)["scalar"].problem_id != original


@pytest.mark.parametrize(
    "budget,patience,count,fail",
    [(-1, 1, 4, None), (1, 0, 4, None), (1, 1, 0, None), (1, 1, 5, None), (1, 1, 4, 2)],
)
def test_invalid_run_config_is_reported_without_losing_other_runs(budget, patience, count, fail):
    bins, specs, problems = fixtures()
    problems["bad"] = problems["scalar"]
    bad = RunSpec("bad", 0, budget, 0.1, patience, count, fail_round=fail)
    records = run_many([bad, specs[0]], bins, bins, problems)
    assert records["bad"].status == "failed"
    assert records["scalar"].status == "completed"


def test_zero_budget_returns_base_and_input_mutation_cannot_change_snapshot():
    bins, specs, problems = fixtures()
    result = run_many([RunSpec("scalar", 7, 0, 0.1, 1, 4)], bins, bins, problems)["scalar"]
    assert result.rounds == result.best_round == 0 and result.status == "completed"
    assert result.best_terms == ()
    saved = result.best_raw
    problems["scalar"][0][:] = 100
    bins[:] = 0
    assert result.best_raw == saved
    np.testing.assert_array_equal(result.best_raw, np.zeros((4, 1)))


def test_same_seed_distinct_run_ids_have_distinct_sampling_streams():
    x = np.arange(20)[:, None]
    y = x.astype(float)
    specs = [RunSpec(name, 5, 2, 0.1, 5, 5) for name in ("a", "b")]
    records = run_many(specs, x, x, {name: (y, y) for name in ("a", "b")})
    assert records["a"].samples != records["b"].samples


def test_identity_missing_payloads_and_metadata_order_are_canonical():
    first = data_identity([0, 1], [[float("nan")], [1.0]], ["x"], {"cuts": [0.5], "version": 1})
    assert first == data_identity(
        [0, 1], np.array([[np.nan], [1.0]]), ["x"], {"version": 1, "cuts": [0.5]}
    )
    assert first != data_identity([0, 1], [[None], [1.0]], ["x"], {"version": 1, "cuts": [0.5]})
    with pytest.raises(ValueError):
        data_identity([0], [[object()]], ["x"], {})


def test_binding_cannot_change_prepared_row_order_even_if_fields_agree():
    prepared = data_identity([10, 20], [[0], [1]], ["x"], {})
    with pytest.raises(ValueError):
        bind_identity(prepared, [20, 10], target=([20, 10], [[2], [1]]))


@pytest.mark.parametrize("count", [1, 8, 32])
def test_required_run_counts_keep_individual_results(count):
    x = [[0], [1]]
    specs = [RunSpec(f"run-{i:02d}", 3, 2, 0.1, 3, 2) for i in range(count)]
    problems = {s.run_id: ([[-1], [1]], [[-1], [1]]) for s in specs}
    records = run_many(specs, x, x, problems)
    assert len(records) == count and all(r.rounds == 2 for r in records.values())
    assert select_best(records).run_id == "run-00"
