"""Known answers for notebook classification metrics."""

import numpy as np
import pytest

from notebooks.utils.evaluation import ClassificationEvaluator, ReviewPolicy


def test_metrics_match_hand_calculation_and_accept_additional_extractors():
    evaluator = ClassificationEvaluator(
        ["a", "b"], extra_metrics={"example_metric": lambda y, p: 7.0}
    )
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    metrics = evaluator.from_probabilities(["a", "b"], probabilities)
    assert metrics["accuracy"] == metrics["macro_f1"] == metrics["roc_auc_macro"] == 1
    assert metrics["log_loss"] == pytest.approx((-np.log(0.9) - np.log(0.8)) / 2)
    assert metrics["brier"] == pytest.approx(0.05)
    assert metrics["example_metric"] == 7


def test_unknown_truth_is_rejected_instead_of_mapped_to_a_known_category():
    with pytest.raises(ValueError, match="unseen"):
        ClassificationEvaluator(["a", "b"]).from_probabilities(
            ["unknown"], [[0.9, 0.1]]
        )


def test_zero_coverage_review_policy_has_no_accepted_accuracy():
    assert (
        ReviewPolicy(0.7).evaluate(["a"], [[0.5, 0.5]], ["a", "b"])["accepted_accuracy"]
        is None
    )
