"""Small real fitted bundle shared by package component tests."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from banking_query_triage.inference import TrainedModel
from banking_query_triage.preprocessing import clean_messages
from banking_query_triage.settings import Settings
from notebooks.utils.serving import save_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle_dir(tmp_path):
    pipeline = Pipeline(
        [
            ("preprocessing", FunctionTransformer(clean_messages)),
            ("features", TfidfVectorizer()),
            ("model", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(
        ["card bill", "card arrival", "cash bill", "cash withdrawal"], [0, 0, 1, 1]
    )
    directory = tmp_path / "model"
    save_bundle(
        directory,
        TrainedModel(pipeline, np.array(["card", "cash"])),
        model_version="fixture_v1",
        candidate="fixture",
        threshold=0.7,
        provenance={"operation": "test_fixture"},
    )
    return directory


@pytest.fixture
def settings(bundle_dir):
    return Settings(model_dir=bundle_dir, max_messages=3, max_message_length=64)


@pytest.fixture
def project_root():
    return ROOT
