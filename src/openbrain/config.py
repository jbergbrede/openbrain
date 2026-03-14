from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"


class SearchConfig(BaseModel):
    similarity_threshold: float = 0.4
    max_results: int = 10
    adaptive_weights: bool = True
    score_spread_threshold: float = 0.05
    rrf_k: int = 60


class ConnectionConfig(BaseModel):
    similarity_threshold: float = 0.75
    max_connections: int = 5


class Settings(BaseSettings):
    # Secrets (env vars only)
    openai_api_key: str = ""
    google_api_key: str = ""
    postgres_dsn: str = "postgresql://openbrain:openbrain@localhost:5432/openbrain"
    slack_bot_token: str = ""
    slack_app_token: str = ""

    # Non-secret settings (can also come from YAML)
    embedding: EmbeddingConfig = EmbeddingConfig()
    search: SearchConfig = SearchConfig()
    connection_finding: ConnectionConfig = ConnectionConfig()

    model_config = {
        "env_prefix": "",
        "env_nested_delimiter": "__",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def load_config(config_path: str | Path | None = None) -> Settings:
    yaml_data: dict[str, Any] = {}

    if config_path is None:
        config_path = os.environ.get("OPENBRAIN_CONFIG", "config.yaml")

    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}

    return Settings(**yaml_data)
