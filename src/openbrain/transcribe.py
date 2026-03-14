from __future__ import annotations

import io

import aiohttp
from openai import AsyncOpenAI

_AUDIO_MIME_PREFIXES = ("audio/",)
_AUDIO_VIDEO_MIMES = ("video/mp4", "video/webm")


def is_audio_file(file_info: dict) -> bool:
    mime = file_info.get("mimetype", "")
    return mime.startswith(_AUDIO_MIME_PREFIXES) or mime in _AUDIO_VIDEO_MIMES


async def transcribe_slack_file(
    bot_token: str,
    file_info: dict,
    openai_api_key: str,
) -> tuple[str, float]:
    """Download a Slack audio file and transcribe it via Whisper.

    Returns (transcript_text, duration_seconds).
    Raises on download failure, unsupported format, or API error.
    """
    url = file_info.get("url_private_download") or file_info.get("url_private")
    filename = file_info.get("name", "audio")

    headers = {"Authorization": f"Bearer {bot_token}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            audio_bytes = await resp.read()

    client = AsyncOpenAI(api_key=openai_api_key)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    result = await client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",
    )

    transcript = result.text or ""
    duration = float(result.duration) if result.duration is not None else 0.0
    return transcript, duration
