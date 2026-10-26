"""Small serving configuration with paths relative to its TOML file."""

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Validated artifact location and prediction request limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_dir: Path
    max_messages: int = Field(default=32, ge=1, strict=True)
    max_message_length: int = Field(default=4096, ge=1, strict=True)

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
