from .base import EmbeddingProvider
from ..config import EmbeddingConfig


def get_embedder(config: EmbeddingConfig, secrets: dict) -> EmbeddingProvider:
    provider = config.provider.lower()

    if provider == "openai":
        from .openai import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(
            api_key=secrets.get("openai_api_key", ""),
            model=config.model,
        )
    elif provider == "google":
        from .google import GoogleEmbeddingProvider
        return GoogleEmbeddingProvider(
            api_key=secrets.get("google_api_key", ""),
            model=config.model,
        )
    elif provider == "ollama":
        from .ollama import OllamaEmbeddingProvider
        return OllamaEmbeddingProvider(model=config.model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


__all__ = ["EmbeddingProvider", "get_embedder"]
