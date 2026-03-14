"""Tests for OCLlamaServer covering all endpoints.

Uses pytest-httpx for HTTP mocking — no live server required.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import httpx
import pytest

from OCLlamaServer import (
    APIError,
    AsyncOCLlamaClient,
    AuthenticationError,
    BadRequestError,
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    OCLlamaClient,
    TimeoutError as ClientTimeoutError,
    UnavailableError,
)
from OCLlamaServer import _parsing as P


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> OCLlamaClient:
    return OCLlamaClient("http://test-server:8080", api_key="test-key")


@pytest.fixture
def async_client() -> AsyncOCLlamaClient:
    return AsyncOCLlamaClient("http://test-server:8080", api_key="test-key")


def mock_response(httpx_mock, method: str, path: str, json_data: Any, status: int = 200):
    """Helper to register a mock response."""
    url = f"http://test-server:8080{path}"
    httpx_mock.add_response(
        method=method,
        url=url,
        json=json_data,
        status_code=status,
    )


def _chat_stream_lines(content: str) -> list[str]:
    payload = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": None,
        }],
    }
    return [f"data: {json.dumps(payload)}", "", "data: [DONE]", ""]


class _FakeSyncStreamResponse:
    def __init__(self, lines: list[str], state: dict[str, Any]) -> None:
        self._lines = lines
        self._state = state
        self.closed = False

    def iter_lines(self):
        self._state["iter_started"] = True
        self._state["closed_when_iter_started"] = self.closed
        if self.closed:
            raise httpx.StreamClosed("stream closed before iteration started")

        for line in self._lines:
            if self.closed:
                raise httpx.StreamClosed("stream closed during iteration")
            yield line

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True
        self._state["response_close_count"] += 1


class _FakeAsyncStreamResponse:
    def __init__(self, lines: list[str], state: dict[str, Any]) -> None:
        self._lines = lines
        self._state = state
        self.closed = False

    async def aiter_lines(self):
        self._state["iter_started"] = True
        self._state["closed_when_iter_started"] = self.closed
        if self.closed:
            raise httpx.StreamClosed("stream closed before iteration started")

        for line in self._lines:
            if self.closed:
                raise httpx.StreamClosed("stream closed during iteration")
            yield line

    async def aclose(self) -> None:
        if self.closed:
            return

        self.closed = True
        self._state["response_close_count"] += 1


class _FakeSyncHTTPClient:
    def __init__(self, response: _FakeSyncStreamResponse, state: dict[str, Any]) -> None:
        self._response = response
        self._state = state

    def stream(self, method: str, url: str, *, json: Any = None):
        @contextlib.contextmanager
        def stream_ctx():
            self._state["stream_call"] = {"method": method, "url": url, "json": json}
            self._state["context_enter_count"] += 1
            try:
                yield self._response
            finally:
                self._state["context_exit_count"] += 1
                self._response.close()

        return stream_ctx()


class _FakeAsyncHTTPClient:
    def __init__(self, response: _FakeAsyncStreamResponse, state: dict[str, Any]) -> None:
        self._response = response
        self._state = state

    def stream(self, method: str, url: str, *, json: Any = None):
        @contextlib.asynccontextmanager
        async def stream_ctx():
            self._state["stream_call"] = {"method": method, "url": url, "json": json}
            self._state["context_enter_count"] += 1
            try:
                yield self._response
            finally:
                self._state["context_exit_count"] += 1
                await self._response.aclose()

        return stream_ctx()


# ============================================================================
# Health
# ============================================================================


class TestHealth:
    def test_health_ok(self, client, httpx_mock):
        mock_response(httpx_mock, "GET", "/health", {"status": "ok"})
        result = client.health()
        assert result.status == "ok"

    def test_health_loading(self, client, httpx_mock):
        mock_response(
            httpx_mock, "GET", "/health",
            {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}},
            status=503,
        )
        with pytest.raises(UnavailableError):
            client.health()


# ============================================================================
# Completion
# ============================================================================


class TestCompletion:
    def test_completion_basic(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/completion", {
            "content": "Hello world!",
            "stop": True,
            "stop_type": "eos",
            "stopping_word": "",
            "model": "test-model",
            "tokens_cached": 5,
            "tokens_evaluated": 5,
            "truncated": False,
            "timings": {
                "prompt_n": 5,
                "prompt_ms": 10.0,
                "predicted_n": 10,
                "predicted_ms": 50.0,
            },
        })
        result = client.completion(prompt="Hello", n_predict=10)
        assert result.content == "Hello world!"
        assert result.stop is True
        assert result.stop_type == "eos"
        assert result.timings is not None
        assert result.timings.predicted_n == 10

    def test_completion_with_sampling(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/completion", {
            "content": "result",
            "stop": True,
        })
        result = client.completion(
            prompt="Test",
            temperature=0.5,
            top_k=20,
            top_p=0.9,
            min_p=0.1,
            repeat_penalty=1.2,
            mirostat=2,
            seed=42,
        )
        assert result.content == "result"

    def test_completion_parses_probs(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/completion", {
            "content": "Hello",
            "stop": True,
            "probs": [{
                "id": 1,
                "token": "Hello",
                "prob": 0.9,
                "top_probs": [{
                    "id": 1,
                    "token": "Hello",
                    "prob": 0.9,
                }],
            }],
        })
        result = client.completion(prompt="Hello", n_probs=1, post_sampling_probs=True)
        assert len(result.probs) == 1
        assert result.probs[0].token == "Hello"
        assert result.probs[0].top_probs[0].token == "Hello"


# ============================================================================
# Tokenize / Detokenize
# ============================================================================


class TestTokenize:
    def test_tokenize(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/tokenize", {"tokens": [123, 456, 789]})
        result = client.tokenize("Hello world")
        assert result.tokens == [123, 456, 789]

    def test_tokenize_with_pieces(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/tokenize", {
            "tokens": [
                {"id": 123, "piece": "Hello"},
                {"id": 456, "piece": " world"},
            ],
        })
        result = client.tokenize("Hello world", with_pieces=True)
        assert len(result.tokens) == 2
        assert result.tokens[0]["piece"] == "Hello"

    def test_detokenize(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/detokenize", {"content": "Hello world"})
        result = client.detokenize([123, 456])
        assert result.content == "Hello world"


# ============================================================================
# Apply Template
# ============================================================================


class TestApplyTemplate:
    def test_apply_template(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/apply-template", {
            "prompt": "<|user|>\nHello<|end|>\n<|assistant|>\n",
        })
        result = client.apply_template(messages=[{"role": "user", "content": "Hello"}])
        assert "<|user|>" in result.prompt


# ============================================================================
# Embeddings
# ============================================================================


class TestEmbeddings:
    def test_native_embedding(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/embedding", [
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
        ])
        result = client.embedding("Hello")
        assert len(result) == 1
        assert result[0].embedding == [0.1, 0.2, 0.3]

    def test_oai_embeddings(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/v1/embeddings", {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "test",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        })
        result = client.oai_embeddings(input="Hello", model="test")
        assert result.object == "list"
        assert len(result.data) == 1


# ============================================================================
# Reranking
# ============================================================================


class TestRerank:
    def test_rerank(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/reranking", {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.5},
            ],
            "model": "reranker",
        })
        result = client.rerank(
            query="What is a panda?",
            documents=["hi", "a bear", "The giant panda"],
            top_n=2,
        )
        assert len(result.results) == 2
        assert result.results[0].relevance_score == 0.95


# ============================================================================
# Chat Completion
# ============================================================================


class TestChatCompletion:
    def test_chat_completion(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/v1/chat/completions", {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "timings": {"prompt_n": 10, "predicted_n": 5, "predicted_ms": 25.0},
        })
        result = client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.content == "Hello!"
        assert result.choices[0].finish_reason == "stop"
        assert result.timings is not None
        assert result.timings.predicted_n == 5

    def test_chat_with_reasoning(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/v1/chat/completions", {
            "id": "chatcmpl-456",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "42",
                    "reasoning_content": "Let me think about this...",
                },
                "finish_reason": "stop",
            }],
        })
        result = client.chat_completion(
            messages=[{"role": "user", "content": "What is 6*7?"}],
            reasoning_format="deepseek",
        )
        assert result.content == "42"
        assert result.reasoning_content == "Let me think about this..."

    def test_chat_completion_content_extracts_structured_text(self):
        result = ChatCompletionResponse(choices=[ChatChoice(message={
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " world"},
                {"type": "image_url", "image_url": "ignored"},
            ],
        })])
        assert result.content == "Hello world"

    def test_chat_completion_chunk_delta_content_extracts_structured_text(self):
        chunk = ChatCompletionChunk(choices=[ChatChoice(delta={
            "content": [
                {"type": "text", "text": "Hel"},
                {"type": "text", "text": "lo"},
            ],
        })])
        assert chunk.delta_content == "Hello"

    def test_chat_completion_stream_keeps_stream_context_alive(self, client):
        state = {
            "iter_started": False,
            "closed_when_iter_started": None,
            "context_enter_count": 0,
            "context_exit_count": 0,
            "response_close_count": 0,
        }
        response = _FakeSyncStreamResponse(_chat_stream_lines("Hello!"), state)
        client._client = _FakeSyncHTTPClient(response, state)  # type: ignore[assignment]

        stream = client.chat_completion_stream(messages=[{"role": "user", "content": "Hi"}])

        chunk = next(stream)
        assert chunk.delta_content == "Hello!"
        assert state["iter_started"] is True
        assert state["closed_when_iter_started"] is False
        assert state["context_enter_count"] == 1
        assert state["context_exit_count"] == 0
        assert state["response_close_count"] == 0

        with pytest.raises(StopIteration):
            next(stream)

        assert response.closed is True
        assert state["context_exit_count"] == 1
        assert state["response_close_count"] == 1

        stream.close()
        assert state["context_exit_count"] == 1
        assert state["response_close_count"] == 1


# ============================================================================
# Responses API
# ============================================================================


class TestResponses:
    def test_responses(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/v1/responses", {
            "id": "resp-123",
            "object": "response",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }],
        })
        result = client.responses(model="test", input="Hi", instructions="Be helpful")
        assert result.output_text == "Hello!"


# ============================================================================
# Models
# ============================================================================


class TestModels:
    def test_list_models(self, client, httpx_mock):
        mock_response(httpx_mock, "GET", "/v1/models", {
            "object": "list",
            "data": [{
                "id": "test-model.gguf",
                "object": "model",
                "created": 1700000000,
                "owned_by": "llamacpp",
                "meta": {
                    "vocab_type": 2,
                    "n_vocab": 32000,
                    "n_ctx_train": 4096,
                    "n_embd": 4096,
                    "n_params": 7000000000,
                    "size": 4000000000,
                },
            }],
        })
        result = client.models()
        assert len(result.data) == 1
        assert result.data[0].id == "test-model.gguf"
        assert result.data[0].meta is not None
        assert result.data[0].meta.n_params == 7000000000


# ============================================================================
# Props
# ============================================================================


class TestProps:
    def test_get_props(self, client, httpx_mock):
        mock_response(httpx_mock, "GET", "/props", {
            "total_slots": 4,
            "model_path": "/models/test.gguf",
            "chat_template": "{% for msg in messages %}...",
            "modalities": {"vision": False},
            "build_info": "b1234-abc123",
            "is_sleeping": False,
        })
        result = client.props()
        assert result.total_slots == 4
        assert result.model_path == "/models/test.gguf"
        assert result.is_sleeping is False


# ============================================================================
# Slots
# ============================================================================


class TestSlots:
    def test_get_slots(self, client, httpx_mock):
        mock_response(httpx_mock, "GET", "/slots", [
            {"id": 0, "id_task": 10, "n_ctx": 4096, "is_processing": True, "params": {}, "next_token": {}},
            {"id": 1, "id_task": 0, "n_ctx": 4096, "is_processing": False, "params": {}, "next_token": {}},
        ])
        result = client.slots()
        assert len(result) == 2
        assert result[0].is_processing is True

    def test_slot_save(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url="http://test-server:8080/slots/0?action=save",
            json={"id_slot": 0, "filename": "test.bin", "n_saved": 100, "n_written": 50000},
        )
        result = client.slot_save(0, "test.bin")
        assert result.n_saved == 100

    def test_slot_erase(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url="http://test-server:8080/slots/0?action=erase",
            json={"id_slot": 0, "n_erased": 500},
        )
        result = client.slot_erase(0)
        assert result.n_erased == 500


# ============================================================================
# LoRA Adapters
# ============================================================================


class TestLoraAdapters:
    def test_list_lora(self, client, httpx_mock):
        mock_response(httpx_mock, "GET", "/lora-adapters", [
            {"id": 0, "path": "adapter1.gguf", "scale": 0.5},
            {"id": 1, "path": "adapter2.gguf", "scale": 0.0},
        ])
        result = client.lora_adapters()
        assert len(result) == 2
        assert result[0].scale == 0.5

    def test_set_lora(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/lora-adapters", [
            {"id": 0, "path": "adapter1.gguf", "scale": 0.8},
        ])
        result = client.set_lora_adapters([{"id": 0, "scale": 0.8}])
        assert result[0].scale == 0.8


# ============================================================================
# Anthropic Messages
# ============================================================================


class TestAnthropicMessages:
    def test_messages(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/v1/messages", {
            "id": "msg-123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "test",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        result = client.anthropic_messages(
            model="test",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.text == "Hello!"
        assert result.usage.input_tokens == 10

    def test_count_tokens(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/v1/messages/count_tokens", {"input_tokens": 15})
        result = client.anthropic_count_tokens(
            model="test",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert result.input_tokens == 15

    def test_anthropic_stream_event_preserves_tool_use_fields(self):
        event = P.parse_anthropic_stream_event({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"city": "Phoenix"},
            },
        })
        assert event.content_block is not None
        assert event.content_block.id == "toolu_123"
        assert event.content_block.name == "get_weather"
        assert event.content_block.input == {"city": "Phoenix"}


# ============================================================================
# Router Endpoints
# ============================================================================


class TestRouter:
    def test_router_models(self, client, httpx_mock):
        mock_response(httpx_mock, "GET", "/models", {
            "data": [{
                "id": "gemma-3-4b",
                "in_cache": True,
                "path": "/cache/gemma.gguf",
                "status": {"value": "loaded", "args": ["llama-server"]},
            }],
        })
        result = client.router_models()
        assert len(result.data) == 1
        assert result.data[0].status["value"] == "loaded"

    def test_model_load(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/models/load", {"success": True})
        result = client.model_load("gemma-3-4b")
        assert result.success is True

    def test_model_unload(self, client, httpx_mock):
        mock_response(httpx_mock, "POST", "/models/unload", {"success": True})
        result = client.model_unload("gemma-3-4b")
        assert result.success is True


# ============================================================================
# Error Handling
# ============================================================================


class TestErrors:
    def test_authentication_error(self, client, httpx_mock):
        mock_response(
            httpx_mock, "POST", "/completion",
            {"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}},
            status=401,
        )
        with pytest.raises(AuthenticationError) as exc_info:
            client.completion(prompt="test")
        assert exc_info.value.status_code == 401

    def test_bad_request_error(self, client, httpx_mock):
        mock_response(
            httpx_mock, "POST", "/completion",
            {"error": {"code": 400, "message": "Failed to parse grammar", "type": "invalid_request_error"}},
            status=400,
        )
        with pytest.raises(BadRequestError) as exc_info:
            client.completion(prompt="test", grammar="invalid")
        assert "grammar" in exc_info.value.message.lower()

    def test_generic_api_error(self, client, httpx_mock):
        mock_response(
            httpx_mock, "POST", "/completion",
            {"error": {"code": 418, "message": "I'm a teapot", "type": "teapot_error"}},
            status=418,
        )
        with pytest.raises(APIError) as exc_info:
            client.completion(prompt="test")
        assert exc_info.value.status_code == 418


class TestMetrics:
    def test_metrics_timeout_raises_client_timeout(self, client):
        class FakeHTTPClient:
            def get(self, *args: Any, **kwargs: Any) -> Any:
                raise httpx.ReadTimeout("boom")

        client._client = FakeHTTPClient()  # type: ignore[assignment]

        with pytest.raises(ClientTimeoutError):
            client.metrics()


# ============================================================================
# Async Client Smoke Tests
# ============================================================================


class TestAsyncClient:
    @pytest.mark.asyncio
    async def test_async_health(self, async_client, httpx_mock):
        mock_response(httpx_mock, "GET", "/health", {"status": "ok"})
        result = await async_client.health()
        assert result.status == "ok"

    @pytest.mark.asyncio
    async def test_async_chat_completion(self, async_client, httpx_mock):
        mock_response(httpx_mock, "POST", "/v1/chat/completions", {
            "id": "chatcmpl-async",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Async hello!"},
                "finish_reason": "stop",
            }],
        })
        result = await async_client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.content == "Async hello!"

    @pytest.mark.asyncio
    async def test_async_tokenize(self, async_client, httpx_mock):
        mock_response(httpx_mock, "POST", "/tokenize", {"tokens": [1, 2, 3]})
        result = await async_client.tokenize("test")
        assert result.tokens == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_async_anthropic(self, async_client, httpx_mock):
        mock_response(httpx_mock, "POST", "/v1/messages", {
            "id": "msg-async",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Async response"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        })
        result = await async_client.anthropic_messages(
            model="test",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.text == "Async response"

    @pytest.mark.asyncio
    async def test_async_metrics_timeout_raises_client_timeout(self, async_client):
        class FakeHTTPClient:
            async def get(self, *args: Any, **kwargs: Any) -> Any:
                raise httpx.ReadTimeout("boom")

        async_client._client = FakeHTTPClient()  # type: ignore[assignment]

        with pytest.raises(ClientTimeoutError):
            await async_client.metrics()

    @pytest.mark.asyncio
    async def test_async_chat_completion_stream_keeps_stream_context_alive(self, async_client):
        state = {
            "iter_started": False,
            "closed_when_iter_started": None,
            "context_enter_count": 0,
            "context_exit_count": 0,
            "response_close_count": 0,
        }
        response = _FakeAsyncStreamResponse(_chat_stream_lines("Async hello!"), state)
        async_client._client = _FakeAsyncHTTPClient(response, state)  # type: ignore[assignment]

        stream = async_client.chat_completion_stream(messages=[{"role": "user", "content": "Hi"}])

        chunk = await anext(stream)
        assert chunk.delta_content == "Async hello!"
        assert state["iter_started"] is True
        assert state["closed_when_iter_started"] is False
        assert state["context_enter_count"] == 1
        assert state["context_exit_count"] == 0
        assert state["response_close_count"] == 0

        with pytest.raises(StopAsyncIteration):
            await anext(stream)

        assert response.closed is True
        assert state["context_exit_count"] == 1
        assert state["response_close_count"] == 1

        await stream.aclose()
        assert state["context_exit_count"] == 1
        assert state["response_close_count"] == 1


# ============================================================================
# Client Configuration
# ============================================================================


class TestClientConfig:
    def test_api_prefix(self, httpx_mock):
        client = OCLlamaClient("http://test-server:8080", api_prefix="/api/v2")
        httpx_mock.add_response(
            method="GET",
            url="http://test-server:8080/api/v2/health",
            json={"status": "ok"},
        )
        result = client.health()
        assert result.status == "ok"

    def test_custom_headers(self, httpx_mock):
        client = OCLlamaClient(
            "http://test-server:8080",
            default_headers={"X-Custom": "value"},
        )
        mock_response(httpx_mock, "GET", "/health", {"status": "ok"})
        result = client.health()
        assert result.status == "ok"

    def test_context_manager(self, httpx_mock):
        mock_response(httpx_mock, "GET", "/health", {"status": "ok"})
        with OCLlamaClient("http://test-server:8080") as client:
            result = client.health()
            assert result.status == "ok"
