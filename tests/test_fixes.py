from __future__ import annotations

import asyncio
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError

from app.main import app, reset_groq_service, lifespan, _execute_task, validation_exception_handler
from app.groq_service import GenerationBackend, GroqService
from app.services.generation import GenerationService
from app.services.transcription import TranscriptionService
from app.prompts import CHUNK_ANALYZE_PROMPT, TEACHER_ANALYSIS_AGGREGATE_PROMPT, TEACHER_AGGREGATE_PART_STRUCTURE_PROMPT, TEACHER_AGGREGATE_PART_ENGAGEMENT_PROMPT, TEACHER_AGGREGATE_PART_FLAGS_PROMPT
from app.schemas import TeacherAnalysisAggregateRequest, TeacherAnalysisAggregateResponse, ChunkAnalyzeRequest, ChunkAnalyzeResponse
from app.task_queue import TaskQueueManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """FastAPI TestClient with overridden lifespan."""
    with TestClient(app) as c:
        yield c


@pytest.mark.asyncio
async def test_debug_queue_reports_queue_and_running_counts(client, monkeypatch):
    """/debug/queue должен показывать queued/running по каждому пулу."""
    monkeypatch.setenv("API_KEY", "test-key")

    started = 0
    all_started = asyncio.Event()

    async def executor(task_type, payload):
        nonlocal started
        started += 1
        if started >= 4:
            all_started.set()
        await asyncio.Event().wait()

    manager = TaskQueueManager(executor=executor)
    await manager.start()

    try:
        for idx in range(5):
            await manager.submit(
                "chunk-analyze",
                {
                    "chunk_id": idx,
                    "start_time": "00:00",
                    "end_time": "00:15",
                    "transcript": [{"start_ms": 0, "text": "Тест"}],
                },
            )

        await asyncio.wait_for(all_started.wait(), timeout=1.0)

        with patch("app.main.get_task_queue", return_value=manager):
            response = client.get("/debug/queue", headers={"X-API-Key": "test-key"})

        assert response.status_code == 200
        body = response.json()
        assert body["analysis"]["running"] == 4
        assert body["analysis"]["queued"] == 1
        assert body["analysis"]["workers"] == 4
        assert body["transcribe"]["queued"] == 0
        assert body["transcribe"]["running"] == 0
    finally:
        await manager.stop()


def test_analysis_generation_configs_are_deterministic():
    """Аналитические конфиги Groq и Gemini должны быть детерминированными."""
    assert GroqService._analysis_groq_kwargs() == {
        "temperature": 0.0,
        "top_p": 1.0,
    }

    assert GroqService._analysis_generation_config(json_response=True, max_output_tokens=1) == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "response_mime_type": "application/json",
        "max_output_tokens": 1,
    }


def test_groq_service_init_checks(monkeypatch):
    """Проверяет корректность инициализации GroqService с различными настройками в .env."""
    # 1. Groq transcriber without key raises error
    monkeypatch.setenv("TRANSCRIBER_TYPE", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Не передан GROQ_API_KEY"):
        GroqService()

    # 2. Gemini LLM without key raises error
    monkeypatch.setenv("TRANSCRIBER_TYPE", "local")
    monkeypatch.setenv("LLM_TYPE", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Не передан GEMINI_API_KEY"):
        GroqService()

    # 3. Local + Local doesn't require keys
    monkeypatch.setenv("TRANSCRIBER_TYPE", "local")
    monkeypatch.setenv("LLM_TYPE", "local")
    monkeypatch.setenv("LOCAL_WHISPER_URL", "http://local-whisper/v1")
    monkeypatch.setenv("LOCAL_WHISPER_MODEL", "whisper-custom")
    monkeypatch.setenv("LOCAL_LLM_URL", "http://local-ollama/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "gemma-custom")

    service = GroqService()
    assert service.transcriber_type == "local"
    assert service.llm_type == "local"
    assert service.local_whisper_url == "http://local-whisper/v1"
    assert service.local_whisper_model == "whisper-custom"
    assert service.local_llm_url == "http://local-ollama/v1"
    assert service.local_llm_model == "gemma-custom"


def test_teacher_analysis_aggregate_allows_groups_without_title():
    """В агрегате title должен быть опциональным для некоторых групп."""
    data = {
        "lesson_format": {
            "format": "лекция",
            "comment": "Преобладает объяснение.",
        },
        "audience_engagement": {
            "questions_to_students": {
                "comment": "Много проверочных вопросов.",
                "fragments": [
                    {
                        "start_ms": 0,
                        "end_ms": 1000,
                        "question_type": "checking_understanding",
                        "text": "Поняли?",
                    }
                ],
            },
            "student_answers": {
                "comment": "Есть короткие ответы студентов.",
                "fragments": [
                    {"start_ms": 1000, "end_ms": 2000, "text": "Да"}
                ],
            },
        },
        "lesson_structure": {
            "step_by_step_explanation": {
                "timeline": [
                    {"start_ms": 0, "title": "Введение", "description": "Старт урока."}
                ]
            },
            "goals_and_summary": {
                "intro": {"present": True, "start_ms": 0, "comment": "Есть ввод."},
                "summary": {"present": False, "start_ms": None, "comment": ""},
            },
        },
        "material_explanation": {
            "examples_and_analogies": {
                "comment": "Есть образное объяснение.",
                "fragments": [
                    {
                        "start_ms": 2000,
                        "end_ms": 3000,
                        "type": "analogy",
                        "text": "Объяснение через аналогию.",
                    }
                ],
            }
        },
        "teacher_recommendation": {
            "title": "Рекомендация преподавателю",
            "comment": "Продолжайте задавать короткие проверочные вопросы.",
        },
        "flags": {
            "profanity": {"present": False, "fragments": []},
            "overly_familiar_tone": {"present": False, "fragments": []},
        },
    }

    GroqService._validate_teacher_analysis_aggregate(data)


def test_teacher_analysis_aggregate_response_allows_missing_group_titles():
    """Схема aggregate-response должна принимать группы без title, как в prompt."""
    aggregate = TeacherAnalysisAggregateResponse.model_validate(
        {
            "lesson_format": {"format": "лекция", "comment": "Преобладает объяснение."},
            "audience_engagement": {
                "questions_to_students": {
                    "comment": "Много проверочных вопросов.",
                    "fragments": [
                        {
                            "start_ms": 0,
                            "end_ms": 1000,
                            "question_type": "checking_understanding",
                            "text": "Поняли?",
                        }
                    ],
                },
                "student_answers": {
                    "comment": "Есть короткие ответы студентов.",
                    "fragments": [
                        {"start_ms": 1000, "end_ms": 2000, "text": "Да"}
                    ],
                },
            },
            "lesson_structure": {
                "step_by_step_explanation": {"timeline": []},
                "goals_and_summary": {
                    "intro": {"present": False, "start_ms": None, "comment": ""},
                    "summary": {"present": False, "start_ms": None, "comment": ""},
                },
            },
            "material_explanation": {
                "examples_and_analogies": {
                    "comment": "Есть образное объяснение.",
                    "fragments": [
                        {
                            "start_ms": 2000,
                            "end_ms": 3000,
                            "type": "analogy",
                            "text": "Объяснение через аналогию.",
                        }
                    ],
                }
            },
            "teacher_recommendation": {
                "title": "Рекомендация преподавателю",
                "comment": "Продолжайте задавать короткие проверочные вопросы.",
            },
            "flags": {
                "profanity": {"present": False, "fragments": []},
                "overly_familiar_tone": {"present": False, "fragments": []},
            },
        }
    )

    assert aggregate.audience_engagement.questions_to_students.title is None
    assert aggregate.audience_engagement.student_answers.title is None
    assert aggregate.material_explanation.examples_and_analogies.title is None


def test_teacher_analysis_accepts_russian_fragment_types():
    """Русские алиасы нормализуются, а произвольные question_type не ломают схему."""
    data = {
        "chunk_id": 1,
        "start_time": "00:00",
        "end_time": "00:15",
        "teacher_questions": [
            {
                "start_ms": 100,
                "end_ms": 200,
                "question_type": "проверка понимания",
                "text": "Поняли?",
            },
            {
                "start_ms": 210,
                "end_ms": 300,
                "question_type": "",
                "text": "Еще вопрос?",
            },
            {
                "start_ms": 310,
                "end_ms": 400,
                "question_type": "какой-то нестандартный тип",
                "text": "И еще вопрос?",
            }
        ],
        "student_answers": [],
        "examples_and_analogies": [
            {
                "start_ms": 300,
                "end_ms": 400,
                "type": "аналогия",
                "text": "Объяснение через аналогию.",
            }
        ],
        "lesson_events": [],
        "goals_and_summary": {
            "intro": {"present": False, "start_ms": None, "comment": ""},
            "summary": {"present": False, "start_ms": None, "comment": ""},
        },
        "flags": {
            "profanity": [],
            "overly_familiar_tone": [],
        },
    }

    GroqService._validate_teacher_analysis(data)

    questions = GroqService._extract_teacher_question_items(data["teacher_questions"])
    examples = GroqService._extract_example_span_items(data["examples_and_analogies"])

    assert questions[0]["question_type"] == "checking_understanding"
    assert questions[1]["question_type"] == ""
    assert questions[2]["question_type"] == "какой-то нестандартный тип"
    assert examples[0]["type"] == "analogy"


def test_teacher_analysis_validation_requires_chunk_id_and_timing_metadata():
    """Ответ модели должен содержать chunk_id, start_time и end_time."""
    with pytest.raises(ValueError, match="Отсутствует обязательное поле: chunk_id"):
        GroqService._validate_teacher_analysis(
            {
                "start_time": "00:00",
                "end_time": "00:15",
                "teacher_questions": [],
                "student_answers": [],
                "examples_and_analogies": [],
                "lesson_events": [],
                "goals_and_summary": {
                    "intro": {"present": False, "start_ms": None, "comment": ""},
                    "summary": {"present": False, "start_ms": None, "comment": ""},
                },
                "flags": {
                    "profanity": [],
                    "overly_familiar_tone": [],
                },
            }
        )

    with pytest.raises(ValueError, match="Отсутствует обязательное поле: start_time"):
        GroqService._validate_teacher_analysis(
            {
                "chunk_id": 1,
                "end_time": "00:15",
                "teacher_questions": [],
                "student_answers": [],
                "examples_and_analogies": [],
                "lesson_events": [],
                "goals_and_summary": {
                    "intro": {"present": False, "start_ms": None, "comment": ""},
                    "summary": {"present": False, "start_ms": None, "comment": ""},
                },
                "flags": {
                    "profanity": [],
                    "overly_familiar_tone": [],
                },
            }
        )


@pytest.mark.asyncio
async def test_chunk_analyze_builds_prompt_from_prompt_template():
    """chunk-analyze должен отправлять в модель prompt из prompts.py и данные чанка."""
    service = GroqService.__new__(GroqService)
    service._concurrency_limit = asyncio.Semaphore(1)
    captured = {}

    async def fake_chat(prompt, **kwargs):
        captured["prompt"] = prompt
        return {
            "chunk_id": 1,
            "start_time": "00:00",
            "end_time": "00:15",
            "key_points": ["Основная мысль"],
            "teacher_questions": [],
            "student_answers": [],
            "examples_and_analogies": [],
            "lesson_events": [],
            "goals_and_summary": {
                "intro": {"present": False, "start_ms": None, "comment": ""},
                "summary": {"present": False, "start_ms": None, "comment": ""},
            },
            "flags": {
                "profanity": [],
                "overly_familiar_tone": [],
            },
        }

    service._chat_json_with_validation = AsyncMock(side_effect=fake_chat)

    await GroqService.chunk_analyze(
        service,
        ChunkAnalyzeRequest.model_validate(
            {
                "chunk_id": 1,
                "start_time": "00:00",
                "end_time": "00:15",
                "transcript": [
                    {"start_ms": 0, "text": "Здравствуйте, сегодня разберём производные."},
                    {"start_ms": 5000, "text": "Поняли, почему это работает?"},
                    {"start_ms": 8000, "text": "Давайте посмотрим на пример."},
                ],
            }
        ),
    )

    assert CHUNK_ANALYZE_PROMPT in captured["prompt"]
    assert '"chunk_id": 1' in captured["prompt"]
    assert '"start_ms": 0' in captured["prompt"]


@pytest.mark.asyncio
async def test_teacher_analysis_aggregate_builds_prompt_from_prompt_template():
    """teacher-analysis-aggregate должен отправлять в модель prompt из prompts.py и chunk_analyses."""
    service = GroqService.__new__(GroqService)
    service._concurrency_limit = asyncio.Semaphore(1)
    captured = {}

    async def fake_chat(prompt, **kwargs):
        captured["prompt"] = prompt
        return {
            "lesson_format": {"format": "лекция", "comment": "Преобладает объяснение."},
            "audience_engagement": {
                "questions_to_students": {"comment": "", "fragments": []},
                "student_answers": {"comment": "", "fragments": []},
            },
            "lesson_structure": {
                "step_by_step_explanation": {"timeline": []},
                "goals_and_summary": {
                    "intro": {"present": False, "start_ms": None, "comment": ""},
                    "summary": {"present": False, "start_ms": None, "comment": ""},
                },
            },
            "material_explanation": {
                "examples_and_analogies": {"comment": "", "fragments": []}
            },
            "teacher_recommendation": {"title": "Рекомендация", "comment": "..."},
            "flags": {
                "profanity": {"present": False, "fragments": []},
                "overly_familiar_tone": {"present": False, "fragments": []},
            },
        }

    service._chat_json_with_validation = AsyncMock(side_effect=fake_chat)

    await GroqService.teacher_analysis_aggregate(
        service,
        TeacherAnalysisAggregateRequest.model_validate(
            {
                "chunk_analyses": [
                    {
                        "chunk_id": 1,
                        "start_time": "00:00",
                        "end_time": "00:15",
                        "key_points": ["Основная мысль"],
                        "teacher_questions": [],
                        "student_answers": [],
                        "examples_and_analogies": [],
                        "lesson_events": [],
                        "goals_and_summary": {
                            "intro": {"present": False, "start_ms": None, "comment": ""},
                            "summary": {"present": False, "start_ms": None, "comment": ""},
                        },
                        "flags": {
                            "profanity": [],
                            "overly_familiar_tone": [],
                        },
                    }
                ]
            }
        ),
    )

    assert (TEACHER_AGGREGATE_PART_STRUCTURE_PROMPT in captured["prompt"] or TEACHER_AGGREGATE_PART_ENGAGEMENT_PROMPT in captured["prompt"] or TEACHER_AGGREGATE_PART_FLAGS_PROMPT in captured["prompt"])
    assert '"chunk_analyses"' in captured["prompt"]


@pytest.mark.asyncio
async def test_teacher_analysis_aggregate_limits_payload_to_thirty_chunks():
    """В teacher-analysis-aggregate в модель должно уходить не больше 30 чанков."""
    service = GroqService.__new__(GroqService)
    service._concurrency_limit = asyncio.Semaphore(1)
    captured = {}

    async def fake_chat(prompt, **kwargs):
        captured["prompt"] = prompt
        return {
            "lesson_format": {"format": "лекция", "comment": "Преобладает объяснение."},
            "audience_engagement": {
                "questions_to_students": {"comment": "", "fragments": []},
                "student_answers": {"comment": "", "fragments": []},
            },
            "lesson_structure": {
                "step_by_step_explanation": {"timeline": []},
                "goals_and_summary": {
                    "intro": {"present": False, "start_ms": None, "comment": ""},
                    "summary": {"present": False, "start_ms": None, "comment": ""},
                },
            },
            "material_explanation": {
                "examples_and_analogies": {"comment": "", "fragments": []}
            },
            "teacher_recommendation": {"title": "Рекомендация", "comment": "..."},
            "flags": {
                "profanity": {"present": False, "fragments": []},
                "overly_familiar_tone": {"present": False, "fragments": []},
            },
        }

    service._chat_json_with_validation = AsyncMock(side_effect=fake_chat)

    chunk_analyses = []
    for idx in range(31):
        chunk_analyses.append(
            {
                "chunk_id": idx + 1,
                "start_time": "00:00",
                "end_time": "00:15",
                "key_points": ["Основная мысль"],
                "teacher_questions": [],
                "student_answers": [],
                "examples_and_analogies": [],
                "lesson_events": [],
                "goals_and_summary": {
                    "intro": {"present": False, "start_ms": None, "comment": ""},
                    "summary": {"present": False, "start_ms": None, "comment": ""},
                },
                "flags": {
                    "profanity": [],
                    "overly_familiar_tone": [],
                },
            }
        )

    await GroqService.teacher_analysis_aggregate(
        service,
        TeacherAnalysisAggregateRequest.model_validate({"chunk_analyses": chunk_analyses}),
    )

    payload_start = captured["prompt"].index('json {"chunk_analyses": ')
    payload = captured["prompt"][payload_start:]
    assert '"chunk_analyses"' in payload
    assert '"chunk_id": 1' in payload
    assert '"chunk_id": 30' in payload
    assert '"chunk_id": 31' not in payload


@pytest.mark.asyncio
async def test_chat_json_gemini_routing():
    service = GroqService.__new__(GroqService)
    service.llm_type = 'gemini'
    service.google_generation_models = ['gemma-4-26b-a4b-it']
    
    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    
    mock_response = MagicMock()
    mock_response.text = '{"hello": "world"}'
    
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    service._google_client = mock_client
    
    service._analysis_generation_config = MagicMock(return_value={})
    service._extract_json = staticmethod(GroqService._extract_json)
    
    res = await service._chat_json('hello prompt')
    assert res == {'hello': 'world'}


@pytest.mark.asyncio
async def test_chat_json_gemini_timeout():
    service = GroqService.__new__(GroqService)
    service.llm_type = 'gemini'
    service.google_generation_models = ['gemma-4-26b-a4b-it']
    
    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=asyncio.TimeoutError())
    service._google_client = mock_client
    
    service._analysis_generation_config = MagicMock(return_value={})
    
    with pytest.raises(RuntimeError, match='Время ожидания генерации'):
        await service._chat_json('hello prompt')


@pytest.mark.asyncio
async def test_chat_json_local_ollama_routing():
    service = GroqService.__new__(GroqService)
    service.llm_type = "local"
    service.local_llm_url = "http://localhost:11434/v1"
    service.local_llm_model = "gemma"
    service._extract_json = staticmethod(GroqService._extract_json)
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value={
        "choices": [
            {
                "message": {
                    "content": '{"local": "ollama"}'
                }
            }
        ]
    })
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await service._chat_json("hello prompt")
        assert res == {"local": "ollama"}
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/v1/chat/completions"
        assert kwargs["json"]["model"] == "gemma"


@pytest.mark.asyncio
async def test_transcribe_local_whisper_routing(tmp_path):
    service = GroqService.__new__(GroqService)
    service.transcriber_type = "local"
    service.local_whisper_url = "http://localhost:8080/v1"
    service.local_whisper_model = "whisper-large-v3"
    service._TRANSCRIPTION_PROMPT = "Убедись, что язык транскрибации соответствует языку на записи"
    
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"dummy audio data")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value={"text": "hello whisper"})
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await service._transcribe_with_retry(audio_path)
        assert res == {"text": "hello whisper"}
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:8080/v1/audio/transcriptions"
        assert kwargs["data"]["model"] == "whisper-large-v3"


@pytest.mark.asyncio
async def test_transcribe_groq_routing():
    from pathlib import Path
    service = GroqService.__new__(GroqService)
    service.transcriber_type = "groq"
    service.transcription_models = ["whisper-large-v3"]
    service._TRANSCRIPTION_PROMPT = "Убедись, что язык транскрибации соответствует языку на записи"
    
    service._groq_client = MagicMock()
    service._groq_client.audio.transcriptions.create = AsyncMock(return_value="mock_verbose_json")
    
    res = await service._transcribe_with_retry(Path("dummy_audio.wav"))
    assert res == "mock_verbose_json"
    service._groq_client.audio.transcriptions.create.assert_awaited_once_with(
        model="whisper-large-v3",
        file=Path("dummy_audio.wav"),
        prompt=service._TRANSCRIPTION_PROMPT,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
        temperature=0.0
    )


@pytest.mark.asyncio
async def test_transcribe_chunk_speaker_parsing():
    from app.schemas import TranscribeChunkJobRequest
    service = GroqService.__new__(GroqService)
    service._concurrency_limit = asyncio.Semaphore(1)
    service._safe_unlink = MagicMock()
    
    # 1. With speaker
    service._transcribe_with_retry = AsyncMock(return_value={
        "segments": [
            {"start": 1.2, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 3.5, "text": "Hi there", "speaker": "SPEAKER_01"},
            {"start": 5.0, "text": "   "}
        ]
    })
    
    payload = TranscribeChunkJobRequest(
        file_path="dummy.wav",
        file_name="dummy.wav",
        mime_type="audio/wav",
        chunk_id=1,
        start_ms=1000,
        end_ms=6000
    )
    
    res = await service.transcribe_chunk(payload)
    assert len(res["transcript"]) == 2
    assert res["transcript"][0]["start_ms"] == 1200
    assert res["transcript"][0]["text"] == "Hello"
    assert res["transcript"][0]["speaker"] == "SPEAKER_00"
    assert res["transcript"][1]["start_ms"] == 3500
    assert res["transcript"][1]["text"] == "Hi there"
    assert res["transcript"][1]["speaker"] == "SPEAKER_01"

    # 2. Without speaker
    service._transcribe_with_retry = AsyncMock(return_value={
        "segments": [
            {"start": 1.2, "text": "Hello"}
        ]
    })
    res2 = await service.transcribe_chunk(payload)
    assert len(res2["transcript"]) == 1
    assert res2["transcript"][0]["start_ms"] == 1200
    assert res2["transcript"][0]["text"] == "Hello"
    assert "speaker" not in res2["transcript"][0]


@pytest.mark.asyncio
async def test_transcription_retries_beyond_old_limit_and_caps_delay(tmp_path):
    service = TranscriptionService(
        transcriber_type="groq",
        local_whisper_url="http://localhost:8080/v1",
        local_whisper_model="whisper-large-v3",
        groq_client=MagicMock(),
    )
    service.groq_client.audio = MagicMock()
    service.groq_client.audio.transcriptions = MagicMock()
    service.groq_client.audio.transcriptions.create = AsyncMock(
        side_effect=[RuntimeError("boom")] * 12 + ["ok"],
    )

    sleep_mock = AsyncMock()
    with patch("app.services.transcription.time.monotonic", return_value=0.0), patch(
        "app.services.transcription.asyncio.sleep",
        sleep_mock,
    ):
        result = await service.transcribe_with_retry(tmp_path / "audio.wav")

    assert result == "ok"
    assert service.groq_client.audio.transcriptions.create.await_count == 13
    sleep_values = [call.args[0] for call in sleep_mock.await_args_list]
    assert sleep_values[:4] == [1.0, 2.0, 4.0, 8.0]
    assert sleep_values[-1] == 1200


@pytest.mark.asyncio
async def test_generation_retries_beyond_old_limit_and_caps_delay():
    service = GenerationService(
        llm_type="local",
        local_llm_url="http://localhost:11434/v1",
        local_llm_model="gemma",
        google_client=None,
    )
    service.chat_json = AsyncMock(side_effect=[RuntimeError("boom")] * 12 + [{"ok": True}])

    sleep_mock = AsyncMock()

    def validator(data: Any) -> None:
        assert data["ok"] is True

    with patch("app.services.generation.time.monotonic", return_value=0.0), patch(
        "app.services.generation.asyncio.sleep",
        sleep_mock,
    ):
        result = await service.chat_json_with_validation(
            "prompt",
            validator,
            request_type="unit-test",
            max_retries=0,
        )

    assert result == {"ok": True}
    assert service.chat_json.await_count == 13
    sleep_values = [call.args[0] for call in sleep_mock.await_args_list]
    assert sleep_values[:4] == [1.0, 2.0, 4.0, 8.0]
    assert sleep_values[-1] == 1200


# ---------------------------------------------------------------------------
# 1. GroqService recreation after APIError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_groq_service_reset_on_api_error():
    """При APIError из Groq сервис должен сбрасываться для пересоздания."""
    reset_groq_service()
    assert app.state.groq_service is None

    mock_service = MagicMock(); mock_service.close = AsyncMock()
    app.state.groq_service = mock_service

    reset_groq_service()
    assert app.state.groq_service is None


@pytest.mark.asyncio
async def test_groq_service_created_lazily():
    """GroqService создаётся лениво при первом обращении."""
    reset_groq_service()

    with patch("app.main.GroqService") as MockSvc:
        instance = MagicMock()
        instance.health_check = AsyncMock(return_value=True)
        MockSvc.return_value = instance

        from app.main import get_groq_service
        svc = get_groq_service()
        assert svc is instance
        MockSvc.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Health check includes groq_ok
# ---------------------------------------------------------------------------

def test_health_check_without_groq(client):
    """Health возвращает groq_ok=False когда сервис не инициализирован."""
    reset_groq_service()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["groq_ok"] is False
    assert data["service"] == "ml"


def test_health_check_with_groq_ok(client):
    """Health возвращает groq_ok=True когда Groq доступен."""
    mock_service = MagicMock(); mock_service.close = AsyncMock()
    mock_service.health_check = AsyncMock(return_value=True)
    mock_service.google_health_check = AsyncMock(return_value=False)
    app.state.groq_service = mock_service

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["groq_ok"] is True
    assert data["google_ok"] is False


def test_health_check_with_google_ok(client):
    """Health возвращает google_ok=True когда Google AI доступен."""
    mock_service = MagicMock(); mock_service.close = AsyncMock()
    mock_service.health_check = AsyncMock(return_value=False)
    mock_service.google_health_check = AsyncMock(return_value=True)
    app.state.groq_service = mock_service

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["groq_ok"] is False
    assert data["google_ok"] is True


@pytest.mark.asyncio
async def test_validation_error_response_includes_field_details():
    """422 должен возвращать конкретный путь и причину ошибки."""
    exc = RequestValidationError(
        [
            {
                "loc": ("body", "chunk_analyses"),
                "msg": "Field required",
                "type": "missing",
            }
        ]
    )

    response = await validation_exception_handler(MagicMock(), exc)
    data = response.body.decode()

    assert response.status_code == 422
    assert "chunk_analyses" in data
    assert "Field required" in data


# ---------------------------------------------------------------------------
# 3. Worker loop: task_done only after successful get
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_loop_task_done_balance():
    """task_done вызывается ровно столько раз, сколько get."""
    executed = []

    async def executor(task_type, payload):
        executed.append(payload)
        return "ok"

    manager = TaskQueueManager(executor=executor)
    await manager.start()

    # submit добавляет задачу в очередь и в _jobs
    job = await manager.submit("transcribe-chunk", {"test": 1})

    # Даём воркеру время обработать задачу
    await asyncio.sleep(0.3)

    await manager.stop()

    assert len(executed) == 1
    assert executed[0] == {"test": 1}
    # task_done должен быть вызван ровно 1 раз
    assert manager._queues["transcribe"]._unfinished_tasks == 0


@pytest.mark.asyncio
async def test_worker_loop_cancelled_during_get():
    """При CancelledError во время get() task_done НЕ вызывается."""
    q = asyncio.Queue(maxsize=10)
    manager = TaskQueueManager(executor=AsyncMock())
    manager._queues = {"transcribe": q, "analysis": asyncio.Queue()}

    worker = asyncio.create_task(manager._worker_loop("transcribe", 0))
    # Даём воркеру зайти в get()
    await asyncio.sleep(0.05)
    worker.cancel()

    with pytest.raises(asyncio.CancelledError):
        await worker

    # task_done не должен был вызваться — unfinished_tasks == 0
    assert q._unfinished_tasks == 0


# ---------------------------------------------------------------------------
# 4. Semaphore safety: CancelledError inside acquire
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semaphore_released_on_cancel():
    """Semaphore освобождается при CancelledError."""
    sem = asyncio.Semaphore(1)

    async def hold():
        async with sem:
            await asyncio.sleep(10)

    task = asyncio.create_task(hold())
    await asyncio.sleep(0.05)  # даём захватить семафор
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Семафор должен быть свободен
    assert sem.locked() is False
    assert sem._value == 1


# ---------------------------------------------------------------------------
# 5. TaskQueueManager submit / get / purge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_and_get():
    """Задача корректно добавляется и извлекается."""
    manager = TaskQueueManager(executor=AsyncMock())
    await manager.start()

    job = await manager.submit("chunk-analyze", {"chunk_id": 1})
    assert job.status == "queued"

    fetched = await manager.get(job.job_id)
    assert fetched is not None
    assert fetched.job_id == job.job_id

    await manager.stop()


@pytest.mark.asyncio
async def test_teacher_analysis_dispatch():
    """Новый тип задачи уходит в chunk_analyze сервиса."""
    mock_service = MagicMock(); mock_service.close = AsyncMock()
    mock_service.chunk_analyze = AsyncMock(return_value={"ok": True})

    with patch("app.main.get_groq_service", return_value=mock_service):
        result = await _execute_task(
            "chunk-analyze",
            {
                "chunk_id": 1,
                "start_time": "00:00",
                "end_time": "00:15",
                "transcript": [{"start_ms": 0, "text": "Тест"}],
            },
        )

    mock_service.chunk_analyze.assert_awaited_once()
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_teacher_analysis_aggregate_dispatch():
    """Агрегатор чанков уходит в teacher_analysis_aggregate сервиса."""
    mock_service = MagicMock(); mock_service.close = AsyncMock()
    mock_service.teacher_analysis_aggregate = AsyncMock(return_value={"lesson_format": {}})

    with patch("app.main.get_groq_service", return_value=mock_service):
        result = await _execute_task(
            "teacher-analysis-aggregate",
            {
                "chunk_analyses": [
                    {
                        "chunk_id": 1,
                        "start_time": "00:00",
                        "end_time": "00:15",
                        "key_points": ["Основная мысль"],
                        "teacher_questions": [],
                        "student_answers": [],
                        "examples_and_analogies": [],
                        "lesson_events": [],
                        "goals_and_summary": {
                            "intro": {"present": False, "start_ms": None, "comment": ""},
                            "summary": {"present": False, "start_ms": None, "comment": ""},
                        },
                        "flags": {
                            "profanity": [],
                            "overly_familiar_tone": [],
                        },
                    }
                ]
            },
        )

    mock_service.teacher_analysis_aggregate.assert_awaited_once()
    assert result == {"lesson_format": {}}


@pytest.mark.asyncio
async def test_purge_expired():
    """Завершённые задачи старше retention удаляются."""
    manager = TaskQueueManager(
        executor=AsyncMock(),
        completed_retention_seconds=0,
    )
    await manager.start()

    job = await manager.submit("chunk-analyze", {"chunk_id": 1})
    job.mark_succeeded("result")

    await manager.purge_expired()
    assert await manager.get(job.job_id) is None

    await manager.stop()


def test_example_fragments_include_type():
    """Фрагменты примеров и аналогий принимают англоязычный type."""
    chunk_analysis = ChunkAnalyzeResponse.model_validate(
        {
            "chunk_id": 1,
            "start_time": "00:00",
            "end_time": "00:15",
            "key_points": ["Основная мысль"],
            "teacher_questions": [
                {
                    "start_ms": 4200,
                    "end_ms": 9800,
                    "question_type": "checking_understanding",
                    "text": "Поняли, почему это работает?",
                }
            ],
            "student_answers": [],
            "examples_and_analogies": [
                {
                    "start_ms": 355000,
                    "end_ms": 363000,
                    "type": "analogy",
                    "text": "Преподаватель сравнивает идею с ситуацией, когда сначала строят фундамент.",
                }
            ],
            "lesson_events": [],
            "goals_and_summary": {
                "intro": {"present": False, "start_ms": None, "comment": ""},
                "summary": {"present": False, "start_ms": None, "comment": ""},
            },
            "flags": {
                "profanity": [],
                "overly_familiar_tone": [],
            },
        }
    )

    example_fragment = chunk_analysis.examples_and_analogies[0].model_dump()
    question_fragment = chunk_analysis.teacher_questions[0].model_dump()

    aggregate = TeacherAnalysisAggregateResponse.model_validate(
        {
            "lesson_format": {"format": "лекция", "comment": "..." },
            "audience_engagement": {
                "questions_to_students": {"title": "Q", "comment": "", "fragments": [question_fragment]},
                "student_answers": {"title": "A", "comment": "", "fragments": []},
            },
            "lesson_structure": {
                "step_by_step_explanation": {"timeline": []},
                "goals_and_summary": {
                    "intro": {"present": False, "start_ms": None, "comment": ""},
                    "summary": {"present": False, "start_ms": None, "comment": ""},
                },
            },
            "material_explanation": {
                "examples_and_analogies": {
                    "title": "Преподаватель активно использует примеры и аналогии",
                    "comment": "",
                    "fragments": [example_fragment],
                }
            },
            "teacher_recommendation": {"title": "Рекомендация", "comment": "..."},
            "flags": {
                "profanity": {"present": False, "fragments": []},
                "overly_familiar_tone": {"present": False, "fragments": []},
            },
        }
    )

    assert aggregate.material_explanation.examples_and_analogies.fragments[0].type == "analogy"
    assert aggregate.audience_engagement.questions_to_students.fragments[0].question_type == "checking_understanding"


def test_prompt_rules_cover_uncertainty_and_paraphrasing():
    """Промпты фиксируют двухшаговый процесс и правила уверенности."""
    assert "FIND" in CHUNK_ANALYZE_PROMPT
    assert "FORMULATE" in CHUNK_ANALYZE_PROMPT
    assert "student_answers" in CHUNK_ANALYZE_PROMPT
    assert "question_type" in CHUNK_ANALYZE_PROMPT
    assert "key_points" in CHUNK_ANALYZE_PROMPT
    assert "точно понятно" in CHUNK_ANALYZE_PROMPT
    assert "перефразируй" in CHUNK_ANALYZE_PROMPT
    assert "end_ms" in CHUNK_ANALYZE_PROMPT
    assert "self-check" in CHUNK_ANALYZE_PROMPT.lower()

    assert "FIND" in TEACHER_ANALYSIS_AGGREGATE_PROMPT
    assert "FORMULATE" in TEACHER_ANALYSIS_AGGREGATE_PROMPT
    assert "teacher_recommendation" in TEACHER_ANALYSIS_AGGREGATE_PROMPT
    assert "только из уже подтверждённых наблюдений" in TEACHER_ANALYSIS_AGGREGATE_PROMPT
    assert "перефразируй" in TEACHER_ANALYSIS_AGGREGATE_PROMPT
    assert "end_ms" in TEACHER_ANALYSIS_AGGREGATE_PROMPT
    assert "question_type" in TEACHER_ANALYSIS_AGGREGATE_PROMPT
