# Using the banking query API

**Goal:** start the local API and send banking queries from Swagger, Postman,
Bruno, curl or the example Python client.

## Start the server

Run from the repository root:

```bash
uv sync --frozen
make api
```

Keep this terminal running. The default address is `http://127.0.0.1:8000`.
Stop the server with **Ctrl+C**.

## Copy and paste into Postman or Bruno

Create a request with these settings:

| Setting | Value |
| --- | --- |
| Method | `POST` |
| URL | `http://127.0.0.1:8000/predict` |
| Header | `Content-Type: application/json` |
| Body | JSON; choose **raw → JSON** in Postman or **JSON** in Bruno |

Paste this into the request body and click **Send**:

```json
{
  "messages": [
    "My card has not arrived yet",
    "I do not recognise this cash withdrawal",
    "I have a problem"
  ]
}
```

For a single query, keep a list with one message:

```json
{
  "messages": ["My card has not arrived yet"]
}
```

The same example payload is saved in [examples/predict.json](../examples/predict.json).

## Select the model

Add `model` to any prediction request; omitting it selects `svm`:

```json
{
  "model": "openai",
  "messages": ["My card has not arrived yet"]
}
```

| `model` | Behavior |
| --- | --- |
| `svm` | Loaded TF-IDF/SVM artifact; calibrated confidence and existing route/review threshold |
| `openai` | OpenAI Responses API using the model configured in `[openai]`, initially `gpt-6-luna` |
| `local_llm` | Optional local OpenAI-compatible chat server; disabled until configured |

Use `GET /models` to inspect all backends, `GET /model?model=openai` for one,
or `GET /ready?model=openai` to check configuration readiness. These endpoints
do not probe upstream availability. Unknown model names return HTTP 422.

The default OpenAI `mode = "auto"` uses `OPENAI_API_KEY` when present at startup.
Without a key it reuses the SVM as a deterministic mock: responses include
`"mocked": true` and `"model_version": "mock:gpt-6-luna"`. This tests API wiring,
not LLM quality. `mode = "mock"` always stays offline; `mode = "live"` returns
HTTP 503 if the key is missing. A failed live request returns HTTP 502 and never
falls back to mock.

To use a real OpenAI call, set `OPENAI_API_KEY` in the server environment and
restart `make api`. Keep the key out of TOML and request bodies. Change
`[openai].model` in [serving.toml](../config/serving.toml) to another model that
supports Responses, structured outputs and `reasoning.effort = "none"`.
[Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)
and [structured output documentation](https://developers.openai.com/api/docs/guides/structured-outputs)
describe the provider interface.

LLM confidence is a self-reported estimate, not a calibrated probability. All
LLM and mock predictions return `action: "review"` and `review_threshold: null`.
`confidence_kind` in model metadata distinguishes `self_reported`,
`svm_mock_probability` and the SVM's `calibrated_probability`.

For a local model, start an OpenAI-compatible server with JSON-mode support.
For example, with [Ollama](https://docs.ollama.com/api/openai-compatibility):

```bash
ollama serve
# In another terminal:
ollama pull qwen2.5:0.5b
```

Set `[local_llm].enabled = true` in the serving TOML and restart the API. The
defaults use `http://127.0.0.1:11434/v1` and `qwen2.5:0.5b`; change both for
another server/model. `BQT_LOCAL_LLM_API_KEY` optionally supplies a separate local
server credential. The OpenAI key is never sent to the local server. No model
download or provider call happens at API startup. Local model quality has not
been benchmarked.

```bash
make predict
uv run python examples/call_api.py --model openai
uv run python examples/call_api.py --model local_llm
```

## Understand the response

The single-query request above returns this response with the supplied model
(confidence shown rounded here):

```json
{
  "model": "svm",
  "model_version": "tfidf_svm_v1",
  "mocked": false,
  "predictions": [
    {
      "category": "card_arrival",
      "confidence": 0.96827,
      "action": "route"
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `model` | Selected backend: `svm`, `openai` or `local_llm` |
| `model_version` | Artifact version or provider model identifier; `mock:` prefix for mock mode |
| `mocked` | Whether a mock supplied the predictions |
| `category` | Predicted banking intent; `/model` lists the allowed categories |
| `confidence` | Category confidence, between 0 and 1; semantics depend on the backend |
| `action` | SVM: `route` when confidence meets its threshold; otherwise `review`. LLM/mock: always `review` |

Results follow the order of the input messages. The supplied model uses a
threshold of **0.7**, including equality. Actions are simulated: the API returns a
decision without forwarding messages or creating a review task. Confidence does
not guarantee correctness, and this model has no separate out-of-domain detector.

## Try it in Swagger

1. Start the API, then open [Swagger UI](http://127.0.0.1:8000/docs).
2. Expand **POST /predict** under **Predictions**.
3. Click **Try it out**, paste either JSON payload above, and click **Execute**.
4. Read the response status and JSON under **Server response**.

[ReDoc](http://127.0.0.1:8000/redoc) provides a reading view. The
[OpenAPI schema](http://127.0.0.1:8000/openapi.json) can also be imported into API
clients that support OpenAPI URL import. These local links work while the server
is running.

## Use curl or Python

In a second terminal, from the repository root:

```bash
curl --request POST http://127.0.0.1:8000/predict \
  --header 'Content-Type: application/json' \
  --data-binary @examples/predict.json
```

Or use the supplied Python client through Make:

```bash
make predict
uv run python examples/call_api.py --payload path/to/my_payload.json
```

The client prints the JSON response or an HTTP/connection error. This is a manual
smoke check: it does not assert the predicted category or response structure.
`make test` runs the automated tests. Make targets are fixed shortcuts; use the
underlying CLI directly when supplying options.

## Endpoints and request limits

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/predict` | Predict one or more messages |
| `GET` | `/health` | Check whether the application responds |
| `GET` | `/ready` | Check whether the model is loaded; includes its version |
| `GET` | `/model` | Inspect the version, categories, review threshold and input limits |
| `GET` | `/models` | List backend configuration, availability, confidence semantics and mock status |

With the default [serving configuration](../config/serving.toml):

- Send **1–32 messages** per request, with at most **4,096 characters per message**.
- Each message must be a nonempty string; whitespace-only messages are rejected.
- The body must contain `messages` as a JSON list; `model` is optional. Other fields are rejected.

`GET /model` reports the actual limits of the running server. The model loads
once at startup; requests run inference without training.

## Change the port or configuration

```bash
uv run banking-query-triage --port 8001
```

Use `http://127.0.0.1:8001/predict` in your client, or run:

```bash
uv run python examples/call_api.py --url http://127.0.0.1:8001/predict
```

Swagger is then at `http://127.0.0.1:8001/docs`.

```bash
uv run banking-query-triage --config path/to/serving.toml
```

The TOML file controls the artifact directory, request limits and provider settings. Relative artifact
paths resolve against the TOML file. The review threshold belongs to the model
manifest and cannot be changed in a prediction request. Restart the server after
changing its configuration.

Keep model settings in `config/serving.toml`, credentials in environment variables,
and example request bodies in `examples/predict.json`. Host and port are server CLI
options; they do not need another configuration file for occasional overrides.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Connection refused | Start `make api`, wait for startup to complete, and check the port |
| HTTP 422 | Check JSON syntax, the `messages` list, nonblank strings and request limits; response `detail` identifies validation errors |
| HTTP 503 | The model is not ready, local LLM is disabled, or live OpenAI has no key; inspect `/models` and configuration |
| HTTP 502 | Provider failed, timed out, refused or returned invalid/truncated output; check the provider and configured token limit |
| Server fails at startup | Check the configured model directory and manifest; use `uv sync --frozen` to reproduce the dependency environment |
| Port already in use | Stop the process using it or select another port |

[Repository overview](../README.md) · [Code map](repository_guide.md)
