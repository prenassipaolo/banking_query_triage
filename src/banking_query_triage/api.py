"""FastAPI application and local server entry point."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .artifacts import load_bundle
from .service import Prediction, TriageService
from .settings import Settings


class PredictionResponse(BaseModel):
    """Ordered predictions tagged with the loaded model version."""

    model_version: str = Field(description="Identifier of the loaded model artifact.")
    predictions: list[Prediction] = Field(
        description="One result per input message, in input order."
    )


class ModelInfo(BaseModel):
    """Public serving metadata without local paths or training data."""

    model_version: str
    candidate: str
    labels: list[str]
    review_threshold: float
    max_messages: int
    max_message_length: int


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an application that loads its model once during startup.

    Args:
        settings: Explicit settings, or settings loaded from BQT_CONFIG/default TOML.

    Returns:
        Configured API application; construction does not load the fitted model.
    """
    settings = settings or Settings.load()
    service: TriageService | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal service
        service = TriageService(load_bundle(settings.model_dir))
        try:
            yield
        finally:
            service = None

    app = FastAPI(
        title="Banking query triage",
        version="0.1.0",
        description=(
            "Classify banking queries using the already-trained model. "
            "Expand **POST /predict**, click **Try it out**, then **Execute** to send "
            "the example JSON request. Results follow input order.\n\n"
            f"Accepts **1–{settings.max_messages} messages** per request, with at most "
            f"**{settings.max_message_length} characters** per message. "
            "Messages must be nonblank strings.\n\n"
            "Each result includes a category, its probability and a simulated "
            "`route` or `review` decision. No message is forwarded and no review task "
            "is created. Check **GET /model** for the actual review threshold and "
            "category list. Prediction requests never train the model."
        ),
        openapi_tags=[
            {
                "name": "Predictions",
                "description": "Send banking queries for classification.",
            },
            {
                "name": "Model",
                "description": "Inspect the loaded model and request limits.",
            },
            {
                "name": "Health",
                "description": "Check application health and model readiness.",
            },
        ],
        lifespan=lifespan,
    )

    class PredictionRequest(BaseModel):
        """A bounded batch of banking queries."""

        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={
                "examples": [
                    {
                        "messages": [
                            "My card has not arrived yet"[: settings.max_message_length]
                        ]
                    }
                ]
            },
        )
        messages: list[
            Annotated[
                str,
                StringConstraints(
                    strict=True, min_length=1, max_length=settings.max_message_length
                ),
            ]
        ] = Field(
            min_length=1,
            max_length=settings.max_messages,
            strict=True,
            description="Banking queries to classify. Use a JSON list even for one query.",
        )

        @field_validator("messages")
        @classmethod
        def reject_blank_messages(cls, messages: list[str]) -> list[str]:
            if any(not message.strip() for message in messages):
                raise ValueError("Messages must be nonempty strings")
            return messages

    def get_service() -> TriageService:
        if service is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return service

    @app.get("/health", tags=["Health"], summary="Check application health")
    def health() -> dict[str, Literal["ok"]]:
        """Return `ok` when the application responds."""
        return {"status": "ok"}

    @app.get("/ready", tags=["Health"], summary="Check model readiness")
    def ready() -> dict[str, str]:
        """Return the loaded model version, or HTTP 503 when no model is ready."""
        loaded = get_service()
        return {
            "status": "ready",
            "model_version": loaded.bundle.manifest.model_version,
        }

    @app.get("/model", tags=["Model"], summary="Inspect the loaded model")
    def model_info() -> ModelInfo:
        """List allowed categories, the review threshold and configured request limits."""
        manifest = get_service().bundle.manifest
        return ModelInfo(
            model_version=manifest.model_version,
            candidate=manifest.candidate,
            labels=manifest.labels,
            review_threshold=manifest.review_threshold,
            max_messages=settings.max_messages,
            max_message_length=settings.max_message_length,
        )

    @app.post("/predict", tags=["Predictions"], summary="Classify banking queries")
    def predict(payload: PredictionRequest) -> PredictionResponse:
        """Return one category, confidence and route/review decision per message.

        `confidence` is the probability of the predicted category. `route` means it
        meets the loaded model's threshold, including equality; otherwise the
        action is `review`. Decisions are simulated and do not guarantee correctness.

        HTTP 422 indicates an invalid body: missing messages, blank/non-string
        messages, excessive lengths/counts or extra fields.
        """
        loaded = get_service()
        return PredictionResponse(
            model_version=loaded.bundle.manifest.model_version,
            predictions=loaded.predict(payload.messages),
        )

    return app


def main() -> None:
    """Run the local API with explicit configuration, host and port options."""
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(Settings.load(args.config)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
