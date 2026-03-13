from google import genai

from .base import EmbeddingProvider


class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def embed(self, text: str) -> list[float]:
        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
        )
        return response.embeddings[0].values
