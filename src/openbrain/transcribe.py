from __future__ import annotations

import io
import logging
import os

import aiohttp
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

_AUDIO_MIME_PREFIXES = ("audio/",)
_AUDIO_VIDEO_MIMES = ("video/mp4", "video/webm")

# Whisper-supported extensions; maps MIME base type → extension
_MIME_TO_EXT: dict[str, str] = {
    "audio/flac": "flac",
    "audio/m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mpeg": "mpeg",
    "audio/mp4": "mp4",
    "audio/mpga": "mpga",
    "audio/ogg": "ogg",
    "audio/oga": "oga",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "video/mp4": "mp4",
    "video/webm": "webm",
}


def is_audio_file(file_info: dict) -> bool:
    mime = file_info.get("mimetype", "").split(";")[0].strip()
    return mime.startswith(_AUDIO_MIME_PREFIXES) or mime in _AUDIO_VIDEO_MIMES


def _whisper_filename(file_info: dict) -> str:
    """Return a filename with a Whisper-recognised extension."""
    name = file_info.get("name", "audio")
    _, ext = os.path.splitext(name)
    if ext.lstrip(".").lower() in _MIME_TO_EXT.values():
        return name  # already has a good extension

    # Derive extension from MIME type (strip codec params, e.g. audio/webm;codecs=opus)
    mime_base = file_info.get("mimetype", "").split(";")[0].strip()
    derived = _MIME_TO_EXT.get(mime_base)
    if derived:
        return f"{os.path.splitext(name)[0] or 'audio'}.{derived}"

    # Fall back to Slack's filetype field (e.g. "webm", "mp4")
    filetype = file_info.get("filetype", "")
    if filetype in _MIME_TO_EXT.values():
        return f"{os.path.splitext(name)[0] or 'audio'}.{filetype}"

    return name  # let Whisper reject if truly unsupported


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
    filename = _whisper_filename(file_info)
    log.info(f"Downloading file: url={url} filename={filename}")

    headers = {"Authorization": f"Bearer {bot_token}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            audio_bytes = await resp.read()
            log.info(
                f"Downloaded {len(audio_bytes)} bytes, "
                f"status={resp.status}, content-type={resp.content_type}, "
                f"first_bytes={audio_bytes[:8].hex()}"
            )

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
    log.info(f"Transcription complete: duration={duration:.1f}s transcript_len={len(transcript)}")
    return transcript, duration
