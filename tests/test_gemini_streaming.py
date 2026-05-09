"""Tests for Gemini async streaming with buffered tool calls (llm_router._stream_gemini)."""
from __future__ import annotations

import re
import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import types
from google.api_core.exceptions import ResourceExhausted

from core.brain.llm_router import Brain
from core.brain.intent_router import IntentDecision
from tools.executor import ToolResult
from tools.registry import EXECUTOR


def _response_with_function_call(name: str, args: dict | None) -> types.GenerateContentResponse:
    fc = types.FunctionCall(name=name, args=args)
    part = types.Part(function_call=fc)
    content = types.Content(role="model", parts=[part])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def _response_with_text(text: str) -> types.GenerateContentResponse:
    part = types.Part(text=text)
    content = types.Content(role="model", parts=[part])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class GeminiStreamingToolCallTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._cfg_patcher = patch("core.brain.llm_router.get_config")
        self.mock_get_config = self._cfg_patcher.start()
        cfg = MagicMock()
        cfg.models.primary_llm = "gemini-2.0-flash"
        cfg.gemini_api_key = "test-key"
        cfg.groq_api_key = "test-key"
        cfg.history.max_tokens = 1800
        self.mock_get_config.return_value = cfg

    def tearDown(self) -> None:
        self._cfg_patcher.stop()

    async def test_partial_function_call_empty_args_does_not_execute(self) -> None:
        mock_client = MagicMock()
        exec_mock = AsyncMock(
            return_value=ToolResult(
                success=True,
                data="ok",
                error=None,
                tool_name="get_weather",
                duration_ms=1.0,
                timestamp=datetime.now(UTC),
            )
        )

        async def gcs_round1(*_a, **_kw):
            async def agen():
                yield _response_with_function_call("get_weather", {})

            return agen()

        mock_client.aio.models.generate_content_stream = gcs_round1

        def custom_gemini_init(self) -> None:
            self.gemini_available = True
            self.gemini_client = mock_client
            self._genai_types = types
            self.gemini_model_name = "gemini-2.0-flash"

        with patch.object(Brain, "_init_gemini", custom_gemini_init):
            with patch.object(Brain, "_init_groq", lambda s: setattr(s, "groq_available", False)):
                with patch.object(Brain, "_init_ollama", lambda s: setattr(s, "ollama_available", False)):
                    with patch.object(EXECUTOR, "execute", exec_mock):
                        brain = Brain()
                        out = [c async for c in brain._stream_gemini("weather?")]

        exec_mock.assert_not_awaited()
        self.assertEqual(out, [])

    async def test_complete_function_call_executes_once_and_streams_followup(self) -> None:
        mock_client = MagicMock()
        exec_mock = AsyncMock(
            return_value=ToolResult(
                success=True,
                data="72F and sunny",
                error=None,
                tool_name="get_weather",
                duration_ms=1.0,
                timestamp=datetime.now(UTC),
            )
        )

        call_count = [0]

        async def gcs(*_a, **_kw):
            if call_count[0] == 0:
                call_count[0] += 1

                async def round1():
                    yield _response_with_function_call("get_weather", {"city": "Boston"})

                return round1()
            call_count[0] += 1

            async def round2():
                yield _response_with_text("Right away sir. It is ")
                yield _response_with_text("Right away sir. It is 72F and sunny in Boston.")

            return round2()

        mock_client.aio.models.generate_content_stream = gcs

        def custom_gemini_init(self) -> None:
            self.gemini_available = True
            self.gemini_client = mock_client
            self._genai_types = types
            self.gemini_model_name = "gemini-2.0-flash"

        with patch.object(Brain, "_init_gemini", custom_gemini_init):
            with patch.object(Brain, "_init_groq", lambda s: setattr(s, "groq_available", False)):
                with patch.object(Brain, "_init_ollama", lambda s: setattr(s, "ollama_available", False)):
                    with patch.object(EXECUTOR, "execute", exec_mock):
                        brain = Brain()
                        chunks = [c async for c in brain._stream_gemini("weather in Boston?")]

        exec_mock.assert_awaited_once()
        args, kwargs = exec_mock.await_args
        self.assertEqual(args[0], "get_weather")
        self.assertEqual(args[1], {"city": "Boston"})

        joined = "".join(chunks)
        self.assertIn("72F", joined)
        self.assertIn("Boston", joined)
        self.assertTrue(call_count[0] >= 2)

    async def test_partial_then_complete_only_one_execution(self) -> None:
        """Incomplete {} chunk then complete args — executor runs once after full args."""
        mock_client = MagicMock()
        exec_mock = AsyncMock(
            return_value=ToolResult(
                success=True,
                data="ok",
                error=None,
                tool_name="get_weather",
                duration_ms=1.0,
                timestamp=datetime.now(UTC),
            )
        )

        call_count = [0]

        async def gcs(*_a, **_kw):
            if call_count[0] == 0:
                call_count[0] += 1

                async def round1():
                    yield _response_with_function_call("get_weather", {})
                    yield _response_with_function_call("get_weather", {"city": "NYC"})

                return round1()
            call_count[0] += 1

            async def round2():
                yield _response_with_text("Understood.")

            return round2()

        mock_client.aio.models.generate_content_stream = gcs

        def custom_gemini_init(self) -> None:
            self.gemini_available = True
            self.gemini_client = mock_client
            self._genai_types = types
            self.gemini_model_name = "gemini-2.0-flash"

        with patch.object(Brain, "_init_gemini", custom_gemini_init):
            with patch.object(Brain, "_init_groq", lambda s: setattr(s, "groq_available", False)):
                with patch.object(Brain, "_init_ollama", lambda s: setattr(s, "ollama_available", False)):
                    with patch.object(EXECUTOR, "execute", exec_mock):
                        brain = Brain()
                        _ = [c async for c in brain._stream_gemini("weather?")]

        exec_mock.assert_awaited_once()
        self.assertEqual(exec_mock.await_args.args[1], {"city": "NYC"})

    async def test_gemini_resource_exhausted_falls_back_to_groq_with_fallback_note_and_excerpt_cap(self) -> None:
        indexed_context = (
            "PERSONAL FILE CONTEXT (answer naturally as if you read these files yourself; cite by number when relevant):\n\n"
            "[1] office-reporting-system.md  (src)\n"
            + ("X" * 2000)
            + "\n\n"
            "[2] other.md  (src)\n"
            + ("Y" * 2000)
        )

        captured: dict[str, str] = {}

        async def fake_stream_groq_with_tools(self, prompt: str, query_hint: str = "", allow_tools: bool = True):
            captured["prompt"] = prompt
            yield "OK"

        async def raise_gemini_stream(*_a, **_kw):
            raise ResourceExhausted("Quota exceeded")

        def custom_gemini_init(self) -> None:
            self.gemini_available = True
            self.gemini_client = MagicMock()
            self.gemini_client.aio.models.generate_content_stream = raise_gemini_stream
            self._genai_types = types
            self.gemini_model_name = "gemini-2.0-flash"

        def custom_groq_init(self) -> None:
            self.groq_available = True
            self.groq_client = MagicMock()
            self.groq_tools = []

        def custom_ollama_init(self) -> None:
            self.ollama_available = False

        with patch.object(Brain, "_init_gemini", custom_gemini_init):
            with patch.object(Brain, "_init_groq", custom_groq_init):
                with patch.object(Brain, "_init_ollama", custom_ollama_init):
                    with patch.object(Brain, "_stream_groq_with_tools", fake_stream_groq_with_tools):
                        brain = Brain()
                        brain.intent_router.detect_intent = lambda _t: IntentDecision(action="none")

                        _chunks = [c async for c in brain.process_command_stream("hello", indexed_context=indexed_context)]

        self.assertTrue(captured, "Expected groq fallback to be invoked and prompt captured")
        prompt = captured["prompt"]

        self.assertIn("[Note: fallback provider — keep response concise]\n", prompt)

        # Ensure excerpts are capped <= 800 chars (specifically the X-filled excerpt).
        lines = prompt.splitlines()
        x_lines = [l for l in lines if "X" in l]
        self.assertTrue(x_lines, "Expected the truncated indexed-context excerpt to include X characters")
        self.assertTrue(all(len(l) <= 800 for l in x_lines))


if __name__ == "__main__":
    unittest.main()
