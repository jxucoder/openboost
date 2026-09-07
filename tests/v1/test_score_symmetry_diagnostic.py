"""Mathematical counterexample checks, without running or simulating CUDA kernels."""

import numpy as np
from benchmarks.v1.cuda_score_symmetry import arithmetic_report


def test_swapped_summaries_and_one_rounding_can_break_the_declared_tie():
    report = arithmetic_report()
    left, right = report["candidates"]
    assert left["oracle_gain"] == right["oracle_gain"]
    assert left["key"] < right["key"]
    np.testing.assert_array_equal(
        left["reconstructed_float32_summaries"], right["reconstructed_float32_summaries"][::-1]
    )
    assert (
        left["child_score_bits"]["separate_products"]
        == right["child_score_bits"]["separate_products"]
    )
    assert (
        left["child_score_values"]["left_product_fused"]
        < right["child_score_values"]["left_product_fused"]
    )
    assert (
        left["child_score_values"]["right_product_fused"]
        > right["child_score_values"]["right_product_fused"]
    )
    difference = (
        right["child_score_values"]["left_product_fused"]
        - left["child_score_values"]["left_product_fused"]
    )
    assert difference == np.spacing(np.float32(left["child_score_values"]["left_product_fused"]))
