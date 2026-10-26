"""Serving paths and limits are explicit and independent of the caller's directory."""

import pytest
from pydantic import ValidationError

from banking_query_triage.settings import Settings


def test_config_resolves_artifacts_relative_to_file(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "serving.toml"
    config_path.write_text('model_dir = "../models/version1"\nmax_messages = 5\n')
    monkeypatch.chdir("/tmp")
    monkeypatch.setenv("BQT_CONFIG", str(config_path))
    settings = Settings.load()
    assert settings.model_dir == tmp_path / "models/version1"
    assert settings.max_messages == 5


@pytest.mark.parametrize(
    "extra",
    [
        {"max_messages": 0},
        {"max_message_length": -1},
        {"max_messages": True},
        {"unknown": 1},
    ],
)
def test_rejects_invalid_configuration(tmp_path, extra):
    with pytest.raises(ValidationError):
        Settings(model_dir=tmp_path, **extra)
