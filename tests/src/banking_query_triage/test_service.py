"""One inference pass, ordered labels, routing boundaries and stable fitted state."""

from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from banking_query_triage.artifacts import load_bundle
from banking_query_triage.inference import TrainedModel
from banking_query_triage.service import TriageService


class ProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0

    def predict_proba(self, messages):
        self.calls += 1
        return np.array(self.probabilities)


def test_one_probability_pass_preserves_order_and_threshold_equality():
    pipeline = ProbabilityModel([[0.3, 0.7], [0.6, 0.4]])
    model = TrainedModel(pipeline, np.array(["card", "cash"]))
    service = TriageService(
        SimpleNamespace(model=model, manifest=SimpleNamespace(review_threshold=0.7))
    )
    predictions = service.predict(["first", "second"])
    assert pipeline.calls == 1
    assert [(p.category, p.confidence, p.action) for p in predictions] == [
        ("cash", 0.7, "route"),
        ("card", 0.6, "review"),
    ]


def test_empty_batch_is_rejected_before_inference():
    pipeline = ProbabilityModel([])
    model = TrainedModel(pipeline, np.array(["card", "cash"]))
    service = TriageService(
        SimpleNamespace(model=model, manifest=SimpleNamespace(review_threshold=0.7))
    )
    with pytest.raises(ValueError, match="At least one"):
        service.predict([])
    assert pipeline.calls == 0


@pytest.mark.integration
def test_repeated_unknown_and_regular_messages_do_not_change_fitted_state(bundle_dir):
    bundle = load_bundle(bundle_dir)
    before = joblib.hash(bundle.model)
    service = TriageService(bundle)
    first = service.predict(["unknownvocabulary123", "cash withdrawal"])
    assert service.predict(["unknownvocabulary123", "cash withdrawal"]) == first
    assert all(0 <= prediction.confidence <= 1 for prediction in first)
    assert joblib.hash(bundle.model) == before
