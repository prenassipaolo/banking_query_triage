"""HTTP validation, startup lifecycle, and parity with direct service calls."""

from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from banking_query_triage.api import create_app
from banking_query_triage.artifacts import load_bundle
from banking_query_triage.service import TriageService
from banking_query_triage.settings import Settings

pytestmark = pytest.mark.integration


def test_http_predictions_match_service_and_model_metadata(settings):
    messages = ["card arrival", "cash withdrawal"]
    service = TriageService(load_bundle(settings.model_dir))
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {
            "status": "ready",
            "model_version": "fixture_v1",
        }
        response = client.post("/predict", json={"messages": messages})
        assert response.status_code == 200
        assert response.json() == {
            "model_version": "fixture_v1",
            "predictions": [asdict(p) for p in service.predict(messages)],
        }
        info = client.get("/model").json()
        assert info["labels"] == ["card", "cash"]
        assert info["review_threshold"] == 0.7
        assert info["max_messages"] == 3
        assert info["max_message_length"] == 64
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
        request_schema = schema["components"]["schemas"]["PredictionRequest"]
        assert request_schema["properties"]["messages"]["maxItems"] == 3
        assert request_schema["properties"]["messages"]["items"]["maxLength"] == 64


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"messages": []},
        {"messages": "card"},
        {"messages": [None]},
        {"messages": [42]},
        {"messages": [True]},
        {"messages": [""]},
        {"messages": [" \n\t"]},
        {"messages": ["\u00a0"]},
        {"messages": ["x" * 65]},
        {"messages": ["card"] * 4},
        {"messages": ["card"], "threshold": 0.1},
    ],
)
def test_invalid_requests_return_422(settings, payload):
    with TestClient(create_app(settings)) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_model_is_loaded_once_during_startup_and_released_after_shutdown(
    settings, monkeypatch
):
    from banking_query_triage import api

    loads = []
    original = api.load_bundle

    def record(directory):
        loads.append(directory)
        return original(directory)

    monkeypatch.setattr(api, "load_bundle", record)
    app = create_app(settings)
    assert loads == []
    with TestClient(app) as client:
        client.post("/predict", json={"messages": ["card"]})
        client.post("/predict", json={"messages": ["cash"]})
        assert loads == [settings.model_dir]
    assert client.get("/ready").status_code == 503


def test_missing_artifact_fails_startup(tmp_path):
    app = create_app(Settings(model_dir=tmp_path / "missing"))
    with pytest.raises(FileNotFoundError), TestClient(app):
        pytest.fail("Application must not start without its serving model")


def test_unstarted_app_has_health_but_is_not_ready(settings):
    client = TestClient(create_app(settings))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/model").status_code == 503
    assert client.post("/predict", json={"messages": ["card"]}).status_code == 503
