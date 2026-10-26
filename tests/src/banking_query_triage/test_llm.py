"""Exercise actual SDK serialization and parsing with offline HTTP transports."""

import json
from types import SimpleNamespace

import httpx2
import pytest
from openai import OpenAI

from banking_query_triage.llm import BackendFailure, LLMService


def provider_response(provider, output, completed=True):
    if provider == "local_llm":
        return {
            "id": "chat-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop" if completed else "length",
                    "message": {"role": "assistant", "content": output},
                }
            ],
        }
    return {
        "id": "resp-test",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": "completed" if completed else "incomplete",
        "output": [
            {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": output, "annotations": []}],
            }
        ],
    }


@pytest.mark.parametrize("provider", ["openai", "local_llm"])
def test_wire_contract_allowed_labels_order_and_uncalibrated_review(provider):
    requests = []
    predictions = [
        {"message_index": 1, "category": "cash", "confidence": 0.95},
        {"message_index": 0, "category": "card", "confidence": 1.0},
    ]

    def respond(request):
        requests.append(request)
        return httpx2.Response(
            200,
            json=provider_response(provider, json.dumps({"predictions": predictions})),
        )

    with OpenAI(
        api_key="fake-test-key",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
        service = LLMService(
            provider,
            "test-model",
            SimpleNamespace(info=SimpleNamespace(labels=["card", "cash"])),
            client,
            512,
        )
        result = service.predict(["where is my card?", "cash withdrawal"])
    assert [(p.category, p.confidence, p.action) for p in result] == [
        ("card", 1.0, "review"),
        ("cash", 0.95, "review"),
    ]
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["model"] == "test-model"
    if provider == "openai":
        assert requests[0].url.path == "/v1/responses"
        assert body["store"] is False
        assert body["max_output_tokens"] == 512
        schema = body["text"]["format"]["schema"]
        assert schema["properties"]["predictions"]["minItems"] == 2
        assert schema["properties"]["predictions"]["maxItems"] == 2
        assert (
            schema["$defs"]["LLMClassification"]["properties"]["message_index"][
                "maximum"
            ]
            == 1
        )
        assert schema["$defs"]["LLMClassification"]["properties"]["category"][
            "enum"
        ] == ["card", "cash"]
        content = json.loads(body["input"])
    else:
        assert requests[0].url.path == "/v1/chat/completions"
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"]["$defs"][
            "LLMClassification"
        ]["properties"]["category"]["enum"] == ["card", "cash"]
        assert body["max_tokens"] == 512
        content = json.loads(body["messages"][1]["content"])
    assert content == [
        {"message_index": 0, "text": "where is my card?"},
        {"message_index": 1, "text": "cash withdrawal"},
    ]


@pytest.mark.parametrize("provider", ["openai", "local_llm"])
@pytest.mark.parametrize(
    "output, completed",
    [
        ("not json", True),
        ('{"predictions": []}', True),
        (
            '{"predictions": [{"message_index": 0, "category": "unknown", "confidence": 0.5}]}',
            True,
        ),
        (
            '{"predictions": [{"message_index": 0, "category": "card", "confidence": 1.1}]}',
            True,
        ),
        (
            '{"predictions": [{"message_index": 0, "category": "card", "confidence": NaN}]}',
            True,
        ),
        (
            '{"predictions": [{"message_index": true, "category": "card", "confidence": 0.5}]}',
            True,
        ),
        (
            '{"predictions": [{"message_index": 1, "category": "card", "confidence": 0.5}]}',
            True,
        ),
        (
            '{"predictions": [{"message_index": 0, "category": "card", "confidence": 0.5}, {"message_index": 0, "category": "card", "confidence": 0.5}]}',
            True,
        ),
        (
            '{"predictions": [{"message_index": 0, "category": "card", "confidence": 0.5}]}',
            False,
        ),
        ("", True),  # Refusal or no usable text.
    ],
)
def test_rejects_invalid_refused_or_truncated_output(provider, output, completed):
    with OpenAI(
        api_key="fake-test-key",
        http_client=httpx2.Client(
            transport=httpx2.MockTransport(
                lambda request: httpx2.Response(
                    200, json=provider_response(provider, output, completed)
                )
            )
        ),
    ) as client:
        service = LLMService(
            provider,
            "test-model",
            SimpleNamespace(info=SimpleNamespace(labels=["card", "cash"])),
            client,
            512,
        )
        with pytest.raises(BackendFailure):
            service.predict(["card"])


@pytest.mark.parametrize("provider", ["openai", "local_llm"])
def test_connection_failures_are_backend_failures(provider):
    def fail(request):
        raise httpx2.ConnectError("private-connection-details", request=request)

    with OpenAI(
        api_key="fake-test-key",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(fail)),
    ) as client:
        service = LLMService(
            provider,
            "test-model",
            SimpleNamespace(info=SimpleNamespace(labels=["card", "cash"])),
            client,
            512,
        )
        with pytest.raises(BackendFailure, match="LLM request failed"):
            service.predict(["card"])


@pytest.mark.parametrize(
    "provider, envelope",
    [
        ("openai", []),
        ("openai", {"status": "completed", "output": None}),
        (
            "openai",
            {"status": "completed", "output": [{"type": "message", "content": None}]},
        ),
        ("local_llm", []),
        ("local_llm", {"choices": [None]}),
        ("local_llm", {"choices": [{"finish_reason": "stop", "message": None}]}),
    ],
)
def test_malformed_provider_envelopes_are_backend_failures(provider, envelope):
    with OpenAI(
        api_key="fake",
        http_client=httpx2.Client(
            transport=httpx2.MockTransport(
                lambda request: httpx2.Response(200, json=envelope)
            )
        ),
    ) as client:
        service = LLMService(
            provider,
            "test",
            SimpleNamespace(info=SimpleNamespace(labels=["card", "cash"])),
            client,
            512,
        )
        with pytest.raises(BackendFailure):
            service.predict(["card"])


def test_local_context_budget_rejects_before_contacting_provider():
    from banking_query_triage.llm import BackendInputTooLarge

    def forbidden(request):
        pytest.fail("Oversized local prompt must not reach the provider")

    with OpenAI(
        api_key="fake",
        http_client=httpx2.Client(transport=httpx2.MockTransport(forbidden)),
    ) as client:
        service = LLMService(
            "local_llm",
            "test",
            SimpleNamespace(info=SimpleNamespace(labels=["card"])),
            client,
            512,
            max_input_bytes=20,
        )
        with pytest.raises(BackendInputTooLarge):
            service.predict(["card"])
