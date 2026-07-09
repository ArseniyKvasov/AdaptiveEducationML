from __future__ import annotations

import asyncio
import httpx
import logging
import time
from functools import partial
from typing import Any, Callable, Optional
from google import genai
from app.utils.json_utils import extract_json

logger = logging.getLogger(__name__)

_MAX_RETRY_DELAY_SECONDS = 20 * 60
_MAX_PROCESSING_BUDGET_SECONDS = 24 * 60 * 60


def parse_model_list(raw_models: Optional[str], *, fallback: list[str]) -> list[str]:
    if not raw_models:
        return fallback
    models = [item.strip() for item in raw_models.split(",") if item.strip()]
    return models or fallback


class GenerationService:
    _ANALYSIS_TEMPERATURE = 0.0
    _ANALYSIS_TOP_P = 1.0
    _ANALYSIS_TOP_K = 1

    def __init__(
        self,
        llm_type: str,
        local_llm_url: str,
        local_llm_model: str,
        google_client: Optional[genai.Client],
        raw_google_models: Optional[str] = None,
        timeout_seconds: int = 600,
    ) -> None:
        self.llm_type = llm_type.lower()
        self.local_llm_url = local_llm_url
        self.local_llm_model = local_llm_model
        self.google_client = google_client
        self.timeout_seconds = timeout_seconds

        self.google_generation_models = parse_model_list(
            raw_google_models,
            fallback=[
                "gemini-3.5-flash",
                "gemini-3.1-flash-lite",
                "gemini-2.0-flash",
                "gemini-2.5-flash",
                "gemma-4-26b-a4b-it",
            ],
        )

    def _analysis_generation_config(
        self,
        *,
        json_response: bool = False,
        max_output_tokens: Optional[int] = None,
        disable_thinking: bool = False,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": self._ANALYSIS_TEMPERATURE,
            "top_p": self._ANALYSIS_TOP_P,
            "top_k": self._ANALYSIS_TOP_K,
        }
        if json_response:
            config["response_mime_type"] = "application/json"
        if max_output_tokens is not None:
            config["max_output_tokens"] = max_output_tokens
        if disable_thinking:
            # Newer Gemini models spend part of max_output_tokens on internal
            # "thinking" before producing visible text, which can leave a
            # tiny token budget with no visible output at all. Disabling it
            # keeps small-budget calls (e.g. health checks) actually
            # producing text.
            config["thinking_config"] = {"thinking_budget": 0}
        return config

    async def chat_json(
        self,
        prompt: str,
        *,
        request_type: str = "unknown",
        is_retry: bool = False,
    ) -> Any:
        if self.llm_type == "local":
            logger.info(f"[Ollama] {self.local_llm_model} ({request_type})")
            try:
                headers = {
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.local_llm_model,
                    "messages": [
                        {"role": "system", "content": "Отвечай строго по контракту."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                }
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    resp = await client.post(
                        f"{self.local_llm_url.rstrip('/')}/chat/completions",
                        json=payload,
                        headers=headers
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"] or "{}"
                    return extract_json(content)
            except Exception as exc:
                logger.error(f"Ollama chat failed: {exc}")
                raise RuntimeError(f"Локальный ИИ (Ollama) вернул ошибку: {exc}") from exc
        else:
            if self.google_client is None:
                raise RuntimeError("GEMINI_API_KEY не передан. Добавьте его в .env.")

            model = random.choice(self.google_generation_models)
            logger.info(f"[Gemini] {model} ({request_type})")
            try:
                from google.genai import types
                
                # Use explicit types for a 'completely right' request
                contents = [
                    types.Content(
                        role='user',
                        parts=[types.Part.from_text(text=prompt)]
                    )
                ]
                
                config = types.GenerateContentConfig(
                    **self._analysis_generation_config(json_response=True),
                    system_instruction=types.Part.from_text(text="Отвечай строго по контракту."),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    http_options=types.HttpOptions(
                        timeout=self.timeout_seconds * 1000,
                        client_args={'http2': False, 'timeout': self.timeout_seconds},
                        async_client_args={'http2': False, 'timeout': self.timeout_seconds}
                    )
                )
                
                response = await self.google_client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                
                # Correct way to get text from response in newer SDK
                response_text = response.text or "{}"
                preview = response_text[:100].replace('\n', ' ')
                logger.info(f"[Gemini Response] {preview}...")
                return extract_json(response_text)
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    f"Время ожидания генерации ({self.timeout_seconds}с) истекло. Попробуйте еще раз."
                ) from exc
            except Exception as exc:
                logger.error(f"Gemini chat failed for {request_type} (prompt len: {len(prompt)}): {exc}")
                raise exc

    async def chat_json_with_validation(
        self,
        prompt: str,
        validator: Callable[[Any], None],
        *,
        request_type: str = "unknown",
        max_retries: Optional[int] = None,
    ) -> Any:
        last_error: Optional[Exception] = None
        delay_seconds = 1.0
        deadline = time.monotonic() + _MAX_PROCESSING_BUDGET_SECONDS
        attempt = 0
        while True:
            try:
                data = await self.chat_json(
                    prompt,
                    request_type=request_type,
                    is_retry=attempt > 0,
                )
                validator(data)
                return data
            except Exception as exc:
                last_error = exc
                attempt += 1
                logger.warning(f"Generation attempt {attempt} failed for {request_type}: {exc}")
                remaining_budget = deadline - time.monotonic()
                if remaining_budget <= 0:
                    raise RuntimeError(
                        f"Не удалось получить корректный JSON ответ в пределах {_MAX_PROCESSING_BUDGET_SECONDS // 3600} часов: {exc}"
                    ) from exc
                sleep_time = min(delay_seconds, remaining_budget, _MAX_RETRY_DELAY_SECONDS)
                logger.info(f"Retrying in {sleep_time:.1f}s... (attempt {attempt})")
                await asyncio.sleep(sleep_time)
                delay_seconds = min(delay_seconds * 2, float(_MAX_RETRY_DELAY_SECONDS))
        if last_error is not None:
            raise RuntimeError(
                f"Не удалось получить корректный JSON ответ в пределах {_MAX_PROCESSING_BUDGET_SECONDS // 3600} часов: {last_error}"
            ) from last_error
        return {}
