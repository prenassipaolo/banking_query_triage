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

    model_version: str
    predictions: list[Prediction]


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

    app = FastAPI(title="Banking query triage", version="0.1.0", lifespan=lifespan)

    class PredictionRequest(BaseModel):
        """A bounded batch of banking queries."""

        model_config = ConfigDict(extra="forbid")
        messages: list[
            Annotated[
                str,
                StringConstraints(
                    strict=True, min_length=1, max_length=settings.max_message_length
                ),
            ]
        ] = Field(min_length=1, max_length=settings.max_messages, strict=True)

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

    @app.get("/health")
    def health() -> dict[str, Literal["ok"]]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        loaded = get_service()
        return {
            "status": "ready",
            "model_version": loaded.bundle.manifest.model_version,
        }

    @app.get("/model")
    def model_info() -> ModelInfo:
        manifest = get_service().bundle.manifest
        return ModelInfo(
            model_version=manifest.model_version,
            candidate=manifest.candidate,
            labels=manifest.labels,
            review_threshold=manifest.review_threshold,
            max_messages=settings.max_messages,
            max_message_length=settings.max_message_length,
        )

    @app.post("/predict")
    def predict(payload: PredictionRequest) -> PredictionResponse:
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
