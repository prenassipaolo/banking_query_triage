"""FastAPI application and local server entry point."""

import argparse
import os
from contextlib import ExitStack, asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .artifacts import load_bundle
from .llm import BackendFailure, BackendUnavailable, LLMService
from .service import Prediction, TriageService
from .settings import Settings


class PredictionResponse(BaseModel):
    """Ordered predictions tagged with the loaded model version."""

    model: str
    model_version: str = Field(
        description="Artifact version or provider model identifier."
    )
    mocked: bool
    predictions: list[Prediction] = Field(
        description="One result per input message, in input order."
    )


class ModelInfo(BaseModel):
    """Public serving metadata without local paths or training data."""

    model: str
    model_version: str
    candidate: str
    labels: list[str]
    review_threshold: float | None
    confidence_kind: str
    mocked: bool
    available: bool
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
    services: dict[str, TriageService | LLMService] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with ExitStack() as stack:
            svm = TriageService(load_bundle(settings.model_dir))
            key = os.getenv("OPENAI_API_KEY", "").strip()
            mocked = settings.openai.mode == "mock" or (
                settings.openai.mode == "auto" and not key
            )
            openai_client = (
                stack.enter_context(
                    OpenAI(
                        api_key=key,
                        base_url="https://api.openai.com/v1",
                        timeout=settings.openai.timeout_seconds,
                        max_retries=0,
                    )
                )
                if key and not mocked
                else None
            )
            local_client = (
                stack.enter_context(
                    OpenAI(
                        api_key=os.getenv("BQT_LOCAL_LLM_API_KEY") or "local",
                        base_url=str(settings.local_llm.base_url),
                        timeout=settings.local_llm.timeout_seconds,
                        max_retries=0,
                    )
                )
                if settings.local_llm.enabled
                else None
            )
            services.update(
                svm=svm,
                openai=LLMService(
                    "openai",
                    settings.openai.model,
                    svm,
                    openai_client,
                    settings.openai.max_output_tokens,
                    mocked=mocked,
                ),
                local_llm=LLMService(
                    "local_llm",
                    settings.local_llm.model,
                    svm,
                    local_client,
                    settings.local_llm.max_output_tokens,
                ),
            )
            try:
                yield
            finally:
                services.clear()

    app = FastAPI(
        title="Banking query triage",
        version="0.1.0",
        description=(
            "Classify banking queries using `svm` (default), `openai` or `local_llm`. "
            "Expand **POST /predict**, click **Try it out**, then **Execute** to send "
            "the example JSON request. Results follow input order.\n\n"
            f"Accepts **1–{settings.max_messages} messages** per request, with at most "
            f"**{settings.max_message_length} characters** per message. "
            "Messages must be nonblank strings.\n\n"
            "Each result includes a category, confidence and a simulated "
            "`route` or `review` decision. These decisions do not forward messages to "
            "banking teams or create review tasks. Live LLM requests send messages to "
            "the selected provider. Check **GET /model** for the review threshold and "
            "category list. LLM results always require review because their confidence "
            "is uncalibrated. Without an API key, OpenAI auto mode uses an explicitly "
            "marked SVM mock. GET /models lists configured backends. "
            "Prediction requests never train the model."
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
        model: Literal["svm", "openai", "local_llm"] = Field(
            default="svm", description="Backend to use for this request."
        )

        @field_validator("messages")
        @classmethod
        def reject_blank_messages(cls, messages: list[str]) -> list[str]:
            if any(not message.strip() for message in messages):
                raise ValueError("Messages must be nonempty strings")
            return messages

    def get_service(
        model: Literal["svm", "openai", "local_llm"] = "svm",
    ) -> TriageService | LLMService:
        if not services:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return services[model]

    def describe(service: TriageService | LLMService) -> ModelInfo:
        return ModelInfo(
            **asdict(service.info),
            max_messages=settings.max_messages,
            max_message_length=settings.max_message_length,
        )

    @app.get("/health", tags=["Health"], summary="Check application health")
    def health() -> dict[str, Literal["ok"]]:
        """Return `ok` when the application responds."""
        return {"status": "ok"}

    @app.get("/ready", tags=["Health"], summary="Check model readiness")
    def ready(model: Literal["svm", "openai", "local_llm"] = "svm") -> dict[str, str]:
        """Return the loaded model version, or HTTP 503 when no model is ready."""
        loaded = get_service(model)
        if not loaded.info.available:
            raise HTTPException(
                status_code=503, detail="Selected backend is not configured"
            )
        return {
            "status": "ready",
            "model_version": loaded.info.model_version,
        }

    @app.get("/models", tags=["Model"], summary="List configured backends")
    def list_models() -> list[ModelInfo]:
        """List backend identities, availability and live/mock status without network calls."""
        get_service()
        return [describe(service) for service in services.values()]

    @app.get("/model", tags=["Model"], summary="Inspect the loaded model")
    def model_info(model: Literal["svm", "openai", "local_llm"] = "svm") -> ModelInfo:
        """List allowed categories, the review threshold and configured request limits."""
        return describe(get_service(model))

    @app.post("/predict", tags=["Predictions"], summary="Classify banking queries")
    def predict(payload: PredictionRequest) -> PredictionResponse:
        """Return one category, confidence and route/review decision per message.

        SVM confidence is a calibrated category probability. SVM results route when
        they meet the artifact threshold, including equality. LLM confidence is a
        self-reported estimate; LLM and mock results always require review.

        HTTP 422 indicates an invalid body: missing messages, blank/non-string
        messages, excessive lengths/counts, unknown models or extra fields. HTTP 503
        means the backend is unconfigured. HTTP 502 means a provider failed or
        returned invalid output. Provider failures never silently fall back to mock.
        """
        loaded = get_service(payload.model)
        try:
            predictions = loaded.predict(payload.messages)
        except BackendUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except BackendFailure as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return PredictionResponse(
            model=loaded.info.model,
            model_version=loaded.info.model_version,
            mocked=loaded.info.mocked,
            predictions=predictions,
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
