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

## Understand the response

The single-query request above returns this response with the supplied model
(confidence shown rounded here):

```json
{
  "model_version": "tfidf_svm_v1",
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
| `model_version` | Identifier of the loaded model artifact |
| `category` | Predicted banking intent; `/model` lists the allowed categories |
| `confidence` | Model probability for the predicted category, between 0 and 1 |
| `action` | `route` when confidence meets the review threshold; otherwise `review` |

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
make predict PAYLOAD=path/to/my_payload.json
```

The client prints the JSON response or an HTTP/connection error.

## Endpoints and request limits

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/predict` | Predict one or more messages |
| `GET` | `/health` | Check whether the application responds |
| `GET` | `/ready` | Check whether the model is loaded; includes its version |
| `GET` | `/model` | Inspect the version, categories, review threshold and input limits |

With the default [serving configuration](../config/serving.toml):

- Send **1–32 messages** per request, with at most **4,096 characters per message**.
- Each message must be a nonempty string; whitespace-only messages are rejected.
- The body must contain `messages` as a JSON list. Additional fields are rejected.

`GET /model` reports the actual limits of the running server. The model loads
once at startup; requests run inference without training.

## Change the port or configuration

```bash
make api PORT=8001
```

Use `http://127.0.0.1:8001/predict` in your client, or run `make predict PORT=8001`.
Swagger is then at `http://127.0.0.1:8001/docs`.

```bash
make api CONFIG=path/to/serving.toml
```

The TOML file controls the artifact directory and request limits. Relative artifact
paths resolve against the TOML file. The review threshold belongs to the model
manifest and cannot be changed in a prediction request. Restart the server after
changing its configuration.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Connection refused | Start `make api`, wait for startup to complete, and check the port |
| HTTP 422 | Check JSON syntax, the `messages` list, nonblank strings and request limits; response `detail` identifies validation errors |
| HTTP 503 | The model is not ready; inspect the server output |
| Server fails at startup | Check the configured model directory and manifest; use `uv sync --frozen` to reproduce the dependency environment |
| Port already in use | Stop the process using it or select another port |

[Repository overview](../README.md) · [Code map](repository_guide.md)
