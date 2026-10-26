"""Threshold boundaries and zero-coverage review behavior."""

import pytest

from banking_query_triage.policy import ReviewPolicy


def test_threshold_equality_routes():
    policy = ReviewPolicy(0.7)
    assert policy.actions([[0.7, 0.3]]).tolist() == ["route"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        ReviewPolicy(threshold)
