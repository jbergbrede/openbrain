from __future__ import annotations

import logging

import aiohttp

log = logging.getLogger(__name__)


async def fetch_paperless_document(base_url: str, api_token: str, doc_id: int) -> dict:
    """Fetch a document's OCR text and metadata from the Paperless-ngx API.

    Returns a dict with keys: content, title, tags (list of str), correspondent (str|None).
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Token {api_token}"}

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(f"{base}/api/documents/{doc_id}/") as resp:
            resp.raise_for_status()
            doc = await resp.json()

        tag_ids: list[int] = doc.get("tags", [])
        tag_names = await _resolve_tags(session, base, tag_ids)

        correspondent_id = doc.get("correspondent")
        correspondent_name: str | None = None
        if correspondent_id:
            correspondent_name = await _resolve_correspondent(session, base, correspondent_id)

    return {
        "content": doc.get("content", ""),
        "title": doc.get("title", ""),
        "tags": tag_names,
        "correspondent": correspondent_name,
    }


async def _resolve_tags(session: aiohttp.ClientSession, base: str, tag_ids: list[int]) -> list[str]:
    if not tag_ids:
        return []
    id_param = ",".join(str(i) for i in tag_ids)
    async with session.get(f"{base}/api/tags/", params={"id__in": id_param}) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return [t["name"] for t in data.get("results", [])]


async def _resolve_correspondent(session: aiohttp.ClientSession, base: str, correspondent_id: int) -> str | None:
    async with session.get(f"{base}/api/correspondents/{correspondent_id}/") as resp:
        if resp.status == 404:
            return None
        resp.raise_for_status()
        data = await resp.json()
    return data.get("name")
