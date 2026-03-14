"""Asynchronous client for the llama.cpp HTTP server.

Usage::

    from OCLlamaServer import AsyncOCLlamaClient

    async def main():
        async with AsyncOCLlamaClient("http://localhost:8080") as client:
            health = await client.health()

            result = await client.chat_completion(messages=[
                {"role": "user", "content": "Hello!"}
            ])
            print(result.content)

            async for chunk in client.chat_completion_stream(
                messages=[{"role": "user", "content": "Tell me a story"}]
            ):
                print(chunk.delta_content, end="", flush=True)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from . import _parsing as P
from . import types as t
from ._sse import AsyncSSEIterator
from .exceptions import (
    ConnectionError,
    TimeoutError,
    raise_for_status,
)


class AsyncOCLlamaClient:
    """Asynchronous client for the llama.cpp server API.

    Parameters:
        base_url: Server URL, e.g. ``"http://localhost:8080"``.
        api_key: Optional API key for authentication.
        timeout: Request timeout in seconds (default 600).
        api_prefix: URL prefix for all endpoints (default ``""``).
        default_headers: Additional headers to send with every request.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        api_prefix: str = "",
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_prefix = api_prefix.rstrip("/") if api_prefix else ""
        self._api_key = api_key

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["x-api-key"] = api_key
        if default_headers:
            headers.update(default_headers)

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    # ── Context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> AsyncOCLlamaClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── Internal helpers ────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self.api_prefix}{path}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = await self._client.request(
                method,
                self._url(path),
                json=json,
                params=params,
            )
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Could not connect to {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"Request timed out: {exc}") from exc

        try:
            body = response.json()
        except Exception:
            body = None

        raise_for_status(response.status_code, body)
        return body

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return await self._request("GET", path, params=params)

    async def _post(self, path: str, *, json: Any = None, params: dict[str, Any] | None = None) -> Any:
        return await self._request("POST", path, json=json, params=params)

    async def _stream_post(self, path: str, *, json: Any = None) -> AsyncSSEIterator:
        try:
            stream_ctx = self._client.stream(
                "POST",
                self._url(path),
                json=json,
            )
            response = await stream_ctx.__aenter__()
            return AsyncSSEIterator(response, stream_ctx)
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Could not connect to {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"Request timed out: {exc}") from exc

    # ====================================================================
    # Core (non-OAI) Endpoints
    # ====================================================================

    async def health(self) -> t.HealthResponse:
        """``GET /health`` — Health check."""
        data = await self._get("/health")
        return P.parse_health(data)

    async def completion(self, **kwargs: Any) -> t.CompletionResponse:
        """``POST /completion`` — Generate text completion."""
        kwargs.setdefault("stream", False)
        data = await self._post("/completion", json=kwargs)
        return P.parse_completion(data)

    async def completion_stream(self, **kwargs: Any) -> AsyncIterator[t.CompletionChunk]:
        """``POST /completion`` with streaming — Yields chunks."""
        kwargs["stream"] = True
        sse_iter = await self._stream_post("/completion", json=kwargs)
        try:
            async for event in sse_iter:
                yield P.parse_completion_chunk(event.json())
        finally:
            await sse_iter.close()

    async def tokenize(
        self,
        content: str,
        *,
        add_special: bool = False,
        parse_special: bool = True,
        with_pieces: bool = False,
    ) -> t.TokenizeResponse:
        """``POST /tokenize`` — Tokenize text."""
        data = await self._post("/tokenize", json={
            "content": content,
            "add_special": add_special,
            "parse_special": parse_special,
            "with_pieces": with_pieces,
        })
        return P.parse_tokenize(data)

    async def detokenize(self, tokens: list[int]) -> t.DetokenizeResponse:
        """``POST /detokenize`` — Convert token IDs to text."""
        data = await self._post("/detokenize", json={"tokens": tokens})
        return P.parse_detokenize(data)

    async def apply_template(self, messages: list[dict[str, Any]]) -> t.ApplyTemplateResponse:
        """``POST /apply-template`` — Format messages with the chat template."""
        data = await self._post("/apply-template", json={"messages": messages})
        return P.parse_apply_template(data)

    async def embedding(
        self,
        content: str,
        *,
        embd_normalize: int = 2,
    ) -> list[t.EmbeddingResponse]:
        """``POST /embedding`` — Generate embeddings (native)."""
        data = await self._post("/embedding", json={
            "content": content,
            "embd_normalize": embd_normalize,
        })
        return P.parse_native_embedding(data)

    async def embeddings(self, **kwargs: Any) -> list[t.EmbeddingResponse]:
        """``POST /embeddings`` — Non-OAI embeddings (all poolings)."""
        data = await self._post("/embeddings", json=kwargs)
        return P.parse_native_embedding(data)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str = "",
        top_n: int | None = None,
    ) -> t.RerankResponse:
        """``POST /reranking`` — Rerank documents against a query."""
        body: dict[str, Any] = {"query": query, "documents": documents}
        if model:
            body["model"] = model
        if top_n is not None:
            body["top_n"] = top_n
        data = await self._post("/reranking", json=body)
        return P.parse_rerank(data)

    async def infill(self, **kwargs: Any) -> t.CompletionResponse:
        """``POST /infill`` — Code fill-in-the-middle completion."""
        kwargs.setdefault("stream", False)
        data = await self._post("/infill", json=kwargs)
        return P.parse_completion(data)

    async def infill_stream(self, **kwargs: Any) -> AsyncIterator[t.CompletionChunk]:
        """``POST /infill`` with streaming."""
        kwargs["stream"] = True
        sse_iter = await self._stream_post("/infill", json=kwargs)
        try:
            async for event in sse_iter:
                yield P.parse_completion_chunk(event.json())
        finally:
            await sse_iter.close()

    async def props(self, *, model: str | None = None) -> t.PropsResponse:
        """``GET /props`` — Server global properties."""
        params = {"model": model} if model else None
        data = await self._get("/props", params=params)
        return P.parse_props(data)

    async def set_props(self, **kwargs: Any) -> dict[str, Any]:
        """``POST /props`` — Update server global properties."""
        data = await self._post("/props", json=kwargs)
        if not isinstance(data, dict):
            raise TypeError("Expected /props to return a JSON object")
        return data

    async def slots(self, *, fail_on_no_slot: bool = False) -> list[t.SlotInfo]:
        """``GET /slots`` — Current slot processing state."""
        params: dict[str, Any] = {}
        if fail_on_no_slot:
            params["fail_on_no_slot"] = "1"
        data = await self._get("/slots", params=params or None)
        return P.parse_slots(data)

    async def metrics(self) -> str:
        """``GET /metrics`` — Prometheus-compatible metrics text."""
        try:
            response = await self._client.get(self._url("/metrics"))
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Could not connect to {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"Request timed out: {exc}") from exc
        if response.status_code >= 400:
            try:
                body = response.json()
            except Exception:
                body = None
            raise_for_status(response.status_code, body)
        return response.text

    async def slot_save(self, slot_id: int, filename: str) -> t.SlotActionResponse:
        """``POST /slots/{id}?action=save`` — Save slot prompt cache."""
        data = await self._post(
            f"/slots/{slot_id}",
            json={"filename": filename},
            params={"action": "save"},
        )
        return P.parse_slot_action(data)

    async def slot_restore(self, slot_id: int, filename: str) -> t.SlotActionResponse:
        """``POST /slots/{id}?action=restore`` — Restore slot prompt cache."""
        data = await self._post(
            f"/slots/{slot_id}",
            json={"filename": filename},
            params={"action": "restore"},
        )
        return P.parse_slot_action(data)

    async def slot_erase(self, slot_id: int) -> t.SlotActionResponse:
        """``POST /slots/{id}?action=erase`` — Erase slot prompt cache."""
        data = await self._post(
            f"/slots/{slot_id}",
            params={"action": "erase"},
        )
        return P.parse_slot_action(data)

    async def lora_adapters(self) -> list[t.LoraAdapter]:
        """``GET /lora-adapters`` — List loaded LoRA adapters."""
        data = await self._get("/lora-adapters")
        return P.parse_lora_adapters(data)

    async def set_lora_adapters(self, adapters: list[dict[str, Any]]) -> list[t.LoraAdapter]:
        """``POST /lora-adapters`` — Set global LoRA adapter scales."""
        data = await self._post("/lora-adapters", json=adapters)
        return P.parse_lora_adapters(data)

    # ====================================================================
    # OpenAI-Compatible Endpoints
    # ====================================================================

    async def models(self) -> t.ModelsResponse:
        """``GET /v1/models`` — List models (OAI-compatible)."""
        data = await self._get("/v1/models")
        return P.parse_models(data)

    async def oai_completion(self, **kwargs: Any) -> t.CompletionResponse:
        """``POST /v1/completions`` — OAI-compatible text completion."""
        kwargs.setdefault("stream", False)
        data = await self._post("/v1/completions", json=kwargs)
        return P.parse_completion(data)

    async def chat_completion(self, **kwargs: Any) -> t.ChatCompletionResponse:
        """``POST /v1/chat/completions`` — OAI-compatible chat completion."""
        kwargs.setdefault("stream", False)
        data = await self._post("/v1/chat/completions", json=kwargs)
        return P.parse_chat_completion(data)

    async def chat_completion_stream(self, **kwargs: Any) -> AsyncIterator[t.ChatCompletionChunk]:
        """``POST /v1/chat/completions`` with streaming."""
        kwargs["stream"] = True
        sse_iter = await self._stream_post("/v1/chat/completions", json=kwargs)
        try:
            async for event in sse_iter:
                yield P.parse_chat_chunk(event.json())
        finally:
            await sse_iter.close()

    async def responses(self, **kwargs: Any) -> t.ResponsesResponse:
        """``POST /v1/responses`` — OAI-compatible Responses API."""
        data = await self._post("/v1/responses", json=kwargs)
        return P.parse_responses(data)

    async def oai_embeddings(self, **kwargs: Any) -> t.OAIEmbeddingResponse:
        """``POST /v1/embeddings`` — OAI-compatible embeddings."""
        data = await self._post("/v1/embeddings", json=kwargs)
        return P.parse_oai_embedding(data)

    async def oai_rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str = "",
        top_n: int | None = None,
    ) -> t.RerankResponse:
        """``POST /v1/rerank`` — OAI-compatible reranking."""
        body: dict[str, Any] = {"query": query, "documents": documents}
        if model:
            body["model"] = model
        if top_n is not None:
            body["top_n"] = top_n
        data = await self._post("/v1/rerank", json=body)
        return P.parse_rerank(data)

    # ====================================================================
    # Anthropic-Compatible Endpoints
    # ====================================================================

    async def anthropic_messages(self, **kwargs: Any) -> t.AnthropicResponse:
        """``POST /v1/messages`` — Anthropic-compatible Messages API."""
        kwargs.setdefault("stream", False)
        data = await self._post("/v1/messages", json=kwargs)
        return P.parse_anthropic_response(data)

    async def anthropic_messages_stream(self, **kwargs: Any) -> AsyncIterator[t.AnthropicStreamEvent]:
        """``POST /v1/messages`` with streaming."""
        kwargs["stream"] = True
        sse_iter = await self._stream_post("/v1/messages", json=kwargs)
        try:
            async for event in sse_iter:
                yield P.parse_anthropic_stream_event(event.json())
        finally:
            await sse_iter.close()

    async def anthropic_count_tokens(self, **kwargs: Any) -> t.CountTokensResponse:
        """``POST /v1/messages/count_tokens`` — Count tokens."""
        data = await self._post("/v1/messages/count_tokens", json=kwargs)
        return P.parse_count_tokens(data)

    # ====================================================================
    # Router / Multi-Model Endpoints
    # ====================================================================

    async def router_models(self) -> t.ModelsResponse:
        """``GET /models`` — List all models in router mode."""
        data = await self._get("/models")
        return P.parse_models(data)

    async def model_load(self, model: str) -> t.ModelActionResponse:
        """``POST /models/load`` — Load a model in router mode."""
        data = await self._post("/models/load", json={"model": model})
        return P.parse_model_action(data)

    async def model_unload(self, model: str) -> t.ModelActionResponse:
        """``POST /models/unload`` — Unload a model in router mode."""
        data = await self._post("/models/unload", json={"model": model})
        return P.parse_model_action(data)
