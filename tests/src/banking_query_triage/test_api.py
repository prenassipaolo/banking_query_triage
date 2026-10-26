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
            "model": "svm",
            "model_version": "fixture_v1",
            "mocked": False,
            "predictions": [asdict(p) for p in service.predict(messages)],
        }
        info = client.get("/model").json()
        assert info["labels"] == ["card", "cash"]
        assert info["review_threshold"] == 0.7
        assert info["max_messages"] == 3
        assert info["max_message_length"] == 64
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        schema = client.get("/openapi.json").json()
        request_schema = schema["components"]["schemas"]["PredictionRequest"]
        assert request_schema["properties"]["messages"]["maxItems"] == 3
        assert request_schema["properties"]["messages"]["items"]["maxLength"] == 64
        for example in request_schema["examples"]:
            result = client.post("/predict", json=example)
            assert result.status_code == 200
            assert len(result.json()["predictions"]) == len(example["messages"])


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
        {"messages": ["card"], "model": "unknown"},
        {"messages": ["card"], "model": None},
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


def test_openai_auto_mock_and_backend_discovery(settings, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    messages = ["card arrival", "cash withdrawal"]
    with TestClient(create_app(settings)) as client:
        models = {model["model"]: model for model in client.get("/models").json()}
        assert set(models) == {"svm", "openai", "local_llm"}
        assert models["svm"]["mocked"] is False
        assert models["openai"]["mocked"] is True
        assert models["openai"]["confidence_kind"] == "svm_mock_probability"
        assert models["local_llm"]["available"] is False
        assert client.get("/model?model=openai").json() == models["openai"]
        assert client.get("/ready?model=openai").status_code == 200
        assert client.get("/ready?model=local_llm").status_code == 503
        svm = client.post("/predict", json={"messages": messages}).json()
        mock = client.post(
            "/predict", json={"messages": messages, "model": "openai"}
        ).json()
        assert mock["model_version"] == "mock:gpt-6-luna"
        assert mock["model"] == "openai"
        assert mock["mocked"] is True
        assert mock["predictions"] == [
            {**p, "action": "review"} for p in svm["predictions"]
        ]
        assert (
            client.post(
                "/predict", json={"messages": messages, "model": "local_llm"}
            ).status_code
            == 503
        )
    assert client.get("/models").status_code == 503


def test_live_without_key_does_not_mock(settings, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = settings.model_copy(
        update={"openai": settings.openai.model_copy(update={"mode": "live"})}
    )
    with TestClient(create_app(settings)) as client:
        info = client.get("/model?model=openai").json()
        assert info["mocked"] is False
        assert info["available"] is False
        assert (
            client.post(
                "/predict", json={"messages": ["card"], "model": "openai"}
            ).status_code
            == 503
        )


def test_explicit_mock_never_constructs_a_provider_client(settings, monkeypatch):
    from banking_query_triage import api

    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    settings = settings.model_copy(
        update={"openai": settings.openai.model_copy(update={"mode": "mock"})}
    )

    def forbidden(**kwargs):
        pytest.fail("Explicit mock must never construct a provider client")

    monkeypatch.setattr(api, "OpenAI", forbidden)
    with TestClient(create_app(settings)) as client:
        assert (
            client.post(
                "/predict", json={"messages": ["card"], "model": "openai"}
            ).json()["mocked"]
            is True
        )


@pytest.mark.parametrize("provider", ["openai", "local_llm"])
def test_provider_failure_is_502_and_svm_stays_available(
    settings, monkeypatch, provider
):
    import httpx2
    from openai import OpenAI

    from banking_query_triage import api

    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    settings = settings.model_copy(
        update={"local_llm": settings.local_llm.model_copy(update={"enabled": True})}
    )
    clients = []
    requests = []

    def fail(request):
        requests.append(request)
        return httpx2.Response(
            429, json={"error": {"message": "private-provider-body"}}
        )

    def factory(**kwargs):
        client = OpenAI(
            **kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(fail))
        )
        clients.append(client)
        return client

    monkeypatch.setattr(api, "OpenAI", factory)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/predict", json={"messages": ["card"], "model": provider}
        )
        assert response.status_code == 502
        assert "private-provider-body" not in response.text
        assert "fake-test-key" not in response.text
        assert len(requests) == 1  # No automatic retries or silent mock fallback.
        assert client.get(f"/model?model={provider}").json()["mocked"] is False
        assert client.post("/predict", json={"messages": ["card"]}).status_code == 200
    assert all(client.is_closed() for client in clients)
