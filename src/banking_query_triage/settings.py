"""Small serving configuration with paths relative to its TOML file."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class OpenAISettings(BaseModel):
    """OpenAI model and explicit live/mock selection; keys come from the environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(default="gpt-6-luna", min_length=1, pattern=r"\S")
    mode: Literal["auto", "mock", "live"] = "auto"
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    max_output_tokens: int = Field(default=4096, ge=1, strict=True)


class LocalLLMSettings(BaseModel):
    """Optional OpenAI-compatible local chat server with JSON-mode support."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=False, strict=True)
    model: str = Field(default="qwen2.5:0.5b", min_length=1, pattern=r"\S")
    base_url: HttpUrl = HttpUrl("http://127.0.0.1:11434/v1")
    timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    max_output_tokens: int = Field(default=4096, ge=1, strict=True)


class Settings(BaseModel):
    """Validated artifact location and prediction request limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_dir: Path
    max_messages: int = Field(default=32, ge=1, strict=True)
    max_message_length: int = Field(default=4096, ge=1, strict=True)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    local_llm: LocalLLMSettings = Field(default_factory=LocalLLMSettings)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        """Load TOML; BQT_CONFIG selects the file when no path is supplied.

        Args:
            path: Optional explicit serving configuration file.

        Returns:
            Settings with an absolute artifact directory.
        """
        config_path = (
            path or Path(os.getenv("BQT_CONFIG", "config/serving.toml"))
        ).resolve()
        settings = cls.model_validate(tomllib.loads(config_path.read_text()))
        return settings.model_copy(
            update={"model_dir": (config_path.parent / settings.model_dir).resolve()}
        )
