import pytest

from rag_permission.evaluation import mean_reciprocal_rank, precision_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_exact_match():
    assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0


def test_recall_at_k_partial_match():
    assert recall_at_k(["a", "x", "b"], ["a", "b"], 2) == 0.5


def test_recall_at_k_no_match():
    assert recall_at_k(["x", "y"], ["a"], 2) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a"], [], 1) == 0.0


def test_recall_at_k_zero_k_boundary():
    assert recall_at_k(["a"], ["a"], 0) == 0.0


def test_recall_at_k_negative_k_boundary():
    assert recall_at_k(["a"], ["a"], -1) == 0.0


def test_recall_at_k_order_insensitive():
    assert recall_at_k(["b", "a"], ["a", "b"], 2) == recall_at_k(["a", "b"], ["a", "b"], 2)


def test_precision_at_k_exact_match():
    assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_precision_at_k_uses_min_k_and_len_denominator():
    assert precision_at_k(["a", "x"], ["a"], 3) == 0.5


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], ["a"], 3) == 0.0


def test_precision_at_k_k_larger_than_retrieved():
    assert precision_at_k(["a"], ["a"], 10) == 1.0


def test_precision_at_k_deduplicates_matches():
    assert precision_at_k(["a", "a", "x"], ["a"], 3) == 0.5


def test_precision_at_k_empty_relevant_boundary():
    assert precision_at_k(["a"], [], 1) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b"], ["a"], 2) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], ["a"], 3) == pytest.approx(1 / 3)


def test_reciprocal_rank_miss():
    assert reciprocal_rank(["x", "y"], ["a"], 2) == 0.0


def test_mean_reciprocal_rank_is_order_sensitive():
    value = mean_reciprocal_rank(
        [["b", "a"], ["a", "b"]], [["a"], ["a"]], 2
    )
    assert value == pytest.approx((0.5 + 1.0) / 2)
