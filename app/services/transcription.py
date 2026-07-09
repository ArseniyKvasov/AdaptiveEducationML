from __future__ import annotations

import asyncio
import httpx
import logging
import time
from pathlib import Path
from typing import Any, Optional
from groq import AsyncGroq
from app.schemas import TranscribeChunkJobRequest

logger = logging.getLogger(__name__)

_MAX_RETRY_DELAY_SECONDS = 20 * 60
_MAX_PROCESSING_BUDGET_SECONDS = 24 * 60 * 60


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()
    except Exception as exc:
        logger.error(f"Error unlinking path {path}: {exc}")


class TranscriptionService:
    def __init__(
        self,
        transcriber_type: str,
        local_whisper_url: str,
        local_whisper_model: str,
        groq_client: Optional[AsyncGroq],
        transcription_prompt: str = "Убедись, что язык транскрибации соответствует языку на записи",
    ) -> None:
        self.transcriber_type = transcriber_type.lower()
        self.local_whisper_url = local_whisper_url
        self.local_whisper_model = local_whisper_model
        self.groq_client = groq_client
        self.transcription_prompt = transcription_prompt
        self.transcription_models = ["whisper-large-v3", "whisper-large-v3-turbo"]

    async def transcribe_with_retry(self, audio_path: Path) -> Any:
        last_error: Optional[Exception] = None
        delay_seconds = 1.0
        deadline = time.monotonic() + _MAX_PROCESSING_BUDGET_SECONDS
        attempt = 0
        while True:
            try:
                if self.transcriber_type == "local":
                    files = {
                        "file": (audio_path.name, audio_path.read_bytes(), "audio/wav")
                    }
                    data = {
                        "model": self.local_whisper_model,
                        "prompt": self.transcription_prompt,
                        "response_format": "verbose_json",
                        "temperature": "0.0"
                    }
                    async with httpx.AsyncClient(timeout=300) as client:
                        resp = await client.post(
                            f"{self.local_whisper_url.rstrip('/')}/audio/transcriptions",
                            data=data,
                            files=files
                        )
                        resp.raise_for_status()
                        return resp.json()
                else:
                    if not self.groq_client:
                        raise RuntimeError("Groq client is not initialized (GROQ_API_KEY missing).")
                    model = self.transcription_models[attempt % len(self.transcription_models)]
                    return await self.groq_client.audio.transcriptions.create(
                        model=model,
                        file=audio_path,
                        prompt=self.transcription_prompt,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                        temperature=0.0,
                    )
            except Exception as exc:
                last_error = exc
                attempt += 1
                logger.warning(f"Transcription attempt {attempt} failed: {exc}")
                remaining_budget = deadline - time.monotonic()
                if remaining_budget <= 0:
                    raise RuntimeError(
                        f"Не удалось выполнить транскрибацию в пределах {_MAX_PROCESSING_BUDGET_SECONDS // 3600} часов: {exc}"
                    ) from exc
                sleep_time = min(delay_seconds, remaining_budget, _MAX_RETRY_DELAY_SECONDS)
                logger.info(f"Retrying transcription in {sleep_time:.1f}s... (attempt {attempt})")
                await asyncio.sleep(sleep_time)
                delay_seconds = min(delay_seconds * 2, float(_MAX_RETRY_DELAY_SECONDS))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Не удалось выполнить транскрибацию.")

    async def transcribe_chunk(self, payload: TranscribeChunkJobRequest, semaphore: asyncio.Semaphore) -> dict[str, Any]:
        async with semaphore:
            audio_path = Path(payload.file_path)
            try:
                transcription = await self.transcribe_with_retry(audio_path)
            finally:
                await asyncio.to_thread(safe_unlink, audio_path)

        transcript: list[dict[str, Any]] = []

        def parse_segment(seg: Any) -> Optional[dict[str, Any]]:
            if isinstance(seg, dict):
                start_sec = seg.get("start") or 0
                text_val = (seg.get("text") or "").strip()
                if not text_val:
                    return None
                item = {"start_ms": int(float(start_sec) * 1000), "text": text_val}
                if "speaker" in seg and seg["speaker"] is not None:
                    item["speaker"] = str(seg["speaker"]).strip()
                return item
            else:
                start_sec = getattr(seg, "start", 0) or 0
                text_val = (getattr(seg, "text", "") or "").strip()
                if not text_val:
                    return None
                item = {"start_ms": int(float(start_sec) * 1000), "text": text_val}
                if hasattr(seg, "speaker") and getattr(seg, "speaker") is not None:
                    item["speaker"] = str(getattr(seg, "speaker")).strip()
                elif hasattr(seg, "get") and seg.get("speaker") is not None:
                    item["speaker"] = str(seg.get("speaker")).strip()
                return item

        if isinstance(transcription, dict):
            segments = transcription.get("segments") or []
            for segment in segments:
                item_dict = parse_segment(segment)
                if item_dict:
                    transcript.append(item_dict)
            if not transcript:
                text = (transcription.get("text") or "").strip()
                if text:
                    item_dict = {"start_ms": payload.start_ms, "text": text}
                    if "speaker" in transcription and transcription["speaker"] is not None:
                        item_dict["speaker"] = str(transcription["speaker"]).strip()
                    transcript.append(item_dict)
                else:
                    raise RuntimeError(
                        "Пустая транскрибация: в аудио не распознано речи или файл пришел в неподдерживаемом формате."
                    )
        else:
            segments = getattr(transcription, "segments", None) or []
            for segment in segments:
                item_dict = parse_segment(segment)
                if item_dict:
                    transcript.append(item_dict)
            if not transcript:
                text = (getattr(transcription, "text", "") or "").strip()
                if text:
                    item_dict = {"start_ms": payload.start_ms, "text": text}
                    if hasattr(transcription, "speaker") and getattr(transcription, "speaker") is not None:
                        item_dict["speaker"] = str(getattr(transcription, "speaker")).strip()
                    elif isinstance(transcription, dict) and "speaker" in transcription and transcription["speaker"] is not None:
                        item_dict["speaker"] = str(transcription["speaker"]).strip()
                    transcript.append(item_dict)
                else:
                    raise RuntimeError(
                        "Пустая транскрибация: в аудио не распознано речи или файл пришел в неподдерживаемом формате."
                    )

        return {
            "chunk_id": payload.chunk_id,
            "start_ms": payload.start_ms,
            "end_ms": payload.end_ms,
            "transcript": transcript,
        }
