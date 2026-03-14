"""Synchronous client for the llama.cpp HTTP server.

Usage::

    from OCLlamaServer import OCLlamaClient

    client = OCLlamaClient("http://localhost:8080")

    # Health check
    health = client.health()

    # Text completion
    result = client.completion(prompt="The meaning of life is")

    # Chat completion
    result = client.chat_completion(messages=[
        {"role": "user", "content": "Hello!"}
    ])

    # Streaming
    for chunk in client.completion_stream(prompt="Once upon a time"):
        print(chunk.content, end="", flush=True)

    # Context manager
    with OCLlamaClient("http://localhost:8080") as client:
        result = client.health()
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Unpack

import httpx

from . import _parsing as P
from . import types as t
from ._sse import SSEIterator
from .exceptions import (
    ConnectionError,
    TimeoutError,
    raise_for_status,
)


class OCLlamaClient:
    """Synchronous client for the llama.cpp server API.

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

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    # ── Context manager ─────────────────────────────────────────────────

    def __enter__(self) -> OCLlamaClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    # ── Internal helpers ────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self.api_prefix}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make a request and return parsed JSON."""
        try:
            response = self._client.request(
                method,
                self._url(path),
                json=json,
                params=params,
            )
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Could not connect to {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"Request timed out: {exc}") from exc

        # Try to parse JSON body for error reporting
        try:
            body = response.json()
        except Exception:
            body = None

        raise_for_status(response.status_code, body)
        return body

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, *, json: Any = None, params: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, json=json, params=params)

    def _stream_post(self, path: str, *, json: Any = None) -> SSEIterator:
        """Start a streaming POST request and return an SSE iterator."""
        try:
            stream_ctx = self._client.stream(
                "POST",
                self._url(path),
                json=json,
            )
            response = stream_ctx.__enter__()
            return SSEIterator(response, stream_ctx)
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Could not connect to {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"Request timed out: {exc}") from exc

    # ====================================================================
    # Core (non-OAI) Endpoints
    # ====================================================================

    def health(self) -> t.HealthResponse:
        """``GET /health`` — Health check.

        Returns:
            :class:`~types.HealthResponse` with ``status`` field.

        Raises:
            UnavailableError: If the model is still loading (503).
        """
        data = self._get("/health")
        return P.parse_health(data)

    def completion(self, **kwargs: Any) -> t.CompletionResponse:
        """``POST /completion`` — Generate text completion.

        Accepts all parameters from :class:`~types.CompletionRequest`.

        Args:
            prompt: The input prompt (required).
            **kwargs: Any other completion parameter.

        Returns:
            :class:`~types.CompletionResponse` with generated text.
        """
        kwargs.setdefault("stream", False)
        data = self._post("/completion", json=kwargs)
        return P.parse_completion(data)

    def completion_stream(self, **kwargs: Any) -> Iterator[t.CompletionChunk]:
        """``POST /completion`` with streaming — Yields chunks in real-time.

        Args:
            prompt: The input prompt (required).
            **kwargs: Any other completion parameter.

        Yields:
            :class:`~types.CompletionChunk` for each token.
        """
        kwargs["stream"] = True
        sse_iter = self._stream_post("/completion", json=kwargs)
        try:
            for event in sse_iter:
                yield P.parse_completion_chunk(event.json())
        finally:
            sse_iter.close()

    def tokenize(
        self,
        content: str,
        *,
        add_special: bool = False,
        parse_special: bool = True,
        with_pieces: bool = False,
    ) -> t.TokenizeResponse:
        """``POST /tokenize`` — Tokenize text.

        Args:
            content: Text to tokenize.
            add_special: Whether to insert BOS and other special tokens.
            parse_special: Whether to tokenize special tokens (vs plaintext).
            with_pieces: Whether to return token text pieces alongside IDs.

        Returns:
            :class:`~types.TokenizeResponse` with ``tokens`` list.
        """
        data = self._post("/tokenize", json={
            "content": content,
            "add_special": add_special,
            "parse_special": parse_special,
            "with_pieces": with_pieces,
        })
        return P.parse_tokenize(data)

    def detokenize(self, tokens: list[int]) -> t.DetokenizeResponse:
        """``POST /detokenize`` — Convert token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            :class:`~types.DetokenizeResponse` with ``content`` string.
        """
        data = self._post("/detokenize", json={"tokens": tokens})
        return P.parse_detokenize(data)

    def apply_template(self, messages: list[dict[str, Any]]) -> t.ApplyTemplateResponse:
        """``POST /apply-template`` — Format messages with the chat template.

        Args:
            messages: Chat messages in OpenAI format.

        Returns:
            :class:`~types.ApplyTemplateResponse` with ``prompt`` string.
        """
        data = self._post("/apply-template", json={"messages": messages})
        return P.parse_apply_template(data)

    def embedding(
        self,
        content: str,
        *,
        embd_normalize: int = 2,
    ) -> list[t.EmbeddingResponse]:
        """``POST /embedding`` — Generate embeddings (native, non-OAI).

        Args:
            content: Text to embed.
            embd_normalize: Normalization type (-1=none, 0=max, 1=taxicab, 2=L2).

        Returns:
            List of :class:`~types.EmbeddingResponse`.
        """
        data = self._post("/embedding", json={
            "content": content,
            "embd_normalize": embd_normalize,
        })
        return P.parse_native_embedding(data)

    def embeddings(self, **kwargs: Any) -> list[t.EmbeddingResponse]:
        """``POST /embeddings`` — Non-OAI embeddings (supports all poolings).

        Accepts same options as :meth:`oai_embeddings`.

        Returns:
            List of :class:`~types.EmbeddingResponse` with per-token embeddings
            when pooling is ``none``.
        """
        data = self._post("/embeddings", json=kwargs)
        return P.parse_native_embedding(data)

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str = "",
        top_n: int | None = None,
    ) -> t.RerankResponse:
        """``POST /reranking`` — Rerank documents against a query.

        Args:
            query: Query string.
            documents: Documents to rank.
            model: Model name.
            top_n: Number of top results to return.

        Returns:
            :class:`~types.RerankResponse` with ranked results.
        """
        body: dict[str, Any] = {"query": query, "documents": documents}
        if model:
            body["model"] = model
        if top_n is not None:
            body["top_n"] = top_n
        data = self._post("/reranking", json=body)
        return P.parse_rerank(data)

    def infill(self, **kwargs: Any) -> t.CompletionResponse:
        """``POST /infill`` — Code fill-in-the-middle completion.

        Args:
            input_prefix: Code before cursor.
            input_suffix: Code after cursor.
            **kwargs: Any other completion parameter.

        Returns:
            :class:`~types.CompletionResponse`.
        """
        kwargs.setdefault("stream", False)
        data = self._post("/infill", json=kwargs)
        return P.parse_completion(data)

    def infill_stream(self, **kwargs: Any) -> Iterator[t.CompletionChunk]:
        """``POST /infill`` with streaming — Yields chunks in real-time.

        Yields:
            :class:`~types.CompletionChunk` for each token.
        """
        kwargs["stream"] = True
        sse_iter = self._stream_post("/infill", json=kwargs)
        try:
            for event in sse_iter:
                yield P.parse_completion_chunk(event.json())
        finally:
            sse_iter.close()

    def props(self, *, model: str | None = None) -> t.PropsResponse:
        """``GET /props`` — Server global properties.

        Args:
            model: Model name (for router mode).

        Returns:
            :class:`~types.PropsResponse`.
        """
        params = {"model": model} if model else None
        data = self._get("/props", params=params)
        return P.parse_props(data)

    def set_props(self, **kwargs: Any) -> dict[str, Any]:
        """``POST /props`` — Update server global properties.

        Requires ``--props`` flag on the server.

        Returns:
            Raw response dict.
        """
        data = self._post("/props", json=kwargs)
        if not isinstance(data, dict):
            raise TypeError("Expected /props to return a JSON object")
        return data

    def slots(self, *, fail_on_no_slot: bool = False) -> list[t.SlotInfo]:
        """``GET /slots`` — Current slot processing state.

        Args:
            fail_on_no_slot: If True, returns 503 when no slot is available.

        Returns:
            List of :class:`~types.SlotInfo`.
        """
        params: dict[str, Any] = {}
        if fail_on_no_slot:
            params["fail_on_no_slot"] = "1"
        data = self._get("/slots", params=params or None)
        return P.parse_slots(data)

    def metrics(self) -> str:
        """``GET /metrics`` — Prometheus-compatible metrics.

        Requires ``--metrics`` flag on the server.

        Returns:
            Raw Prometheus metrics text.
        """
        try:
            response = self._client.get(self._url("/metrics"))
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

    def slot_save(self, slot_id: int, filename: str) -> t.SlotActionResponse:
        """``POST /slots/{id}?action=save`` — Save slot prompt cache.

        Args:
            slot_id: Slot ID.
            filename: File to save to (relative to ``--slot-save-path``).

        Returns:
            :class:`~types.SlotActionResponse`.
        """
        data = self._post(
            f"/slots/{slot_id}",
            json={"filename": filename},
            params={"action": "save"},
        )
        return P.parse_slot_action(data)

    def slot_restore(self, slot_id: int, filename: str) -> t.SlotActionResponse:
        """``POST /slots/{id}?action=restore`` — Restore slot prompt cache.

        Args:
            slot_id: Slot ID.
            filename: File to restore from (relative to ``--slot-save-path``).

        Returns:
            :class:`~types.SlotActionResponse`.
        """
        data = self._post(
            f"/slots/{slot_id}",
            json={"filename": filename},
            params={"action": "restore"},
        )
        return P.parse_slot_action(data)

    def slot_erase(self, slot_id: int) -> t.SlotActionResponse:
        """``POST /slots/{id}?action=erase`` — Erase slot prompt cache.

        Args:
            slot_id: Slot ID.

        Returns:
            :class:`~types.SlotActionResponse`.
        """
        data = self._post(
            f"/slots/{slot_id}",
            params={"action": "erase"},
        )
        return P.parse_slot_action(data)

    def lora_adapters(self) -> list[t.LoraAdapter]:
        """``GET /lora-adapters`` — List loaded LoRA adapters.

        Returns:
            List of :class:`~types.LoraAdapter`.
        """
        data = self._get("/lora-adapters")
        return P.parse_lora_adapters(data)

    def set_lora_adapters(self, adapters: list[dict[str, Any]]) -> list[t.LoraAdapter]:
        """``POST /lora-adapters`` — Set global LoRA adapter scales.

        Args:
            adapters: List of ``{"id": int, "scale": float}`` dicts.

        Returns:
            Updated list of :class:`~types.LoraAdapter`.
        """
        data = self._post("/lora-adapters", json=adapters)
        return P.parse_lora_adapters(data)

    # ====================================================================
    # OpenAI-Compatible Endpoints
    # ====================================================================

    def models(self) -> t.ModelsResponse:
        """``GET /v1/models`` — List models (OAI-compatible).

        Returns:
            :class:`~types.ModelsResponse`.
        """
        data = self._get("/v1/models")
        return P.parse_models(data)

    def oai_completion(self, **kwargs: Any) -> t.CompletionResponse:
        """``POST /v1/completions`` — OAI-compatible text completion.

        Args:
            model: Model name.
            prompt: The input prompt.
            **kwargs: Any OpenAI completions API parameter.

        Returns:
            :class:`~types.CompletionResponse`.
        """
        kwargs.setdefault("stream", False)
        data = self._post("/v1/completions", json=kwargs)
        return P.parse_completion(data)

    def chat_completion(self, **kwargs: Any) -> t.ChatCompletionResponse:
        """``POST /v1/chat/completions`` — OAI-compatible chat completion.

        Args:
            messages: List of chat message dicts (required).
            model: Model name.
            **kwargs: Any OpenAI chat completions API parameter, plus
                llama.cpp extensions (``mirostat``, ``grammar``, etc.).

        Returns:
            :class:`~types.ChatCompletionResponse`.
        """
        kwargs.setdefault("stream", False)
        data = self._post("/v1/chat/completions", json=kwargs)
        return P.parse_chat_completion(data)

    def chat_completion_stream(self, **kwargs: Any) -> Iterator[t.ChatCompletionChunk]:
        """``POST /v1/chat/completions`` with streaming.

        Yields:
            :class:`~types.ChatCompletionChunk` for each token.
        """
        kwargs["stream"] = True
        sse_iter = self._stream_post("/v1/chat/completions", json=kwargs)
        try:
            for event in sse_iter:
                yield P.parse_chat_chunk(event.json())
        finally:
            sse_iter.close()

    def responses(self, **kwargs: Any) -> t.ResponsesResponse:
        """``POST /v1/responses`` — OAI-compatible Responses API.

        Args:
            model: Model name.
            input: User input text.
            instructions: System instructions.
            **kwargs: Any OpenAI Responses API parameter.

        Returns:
            :class:`~types.ResponsesResponse`.
        """
        data = self._post("/v1/responses", json=kwargs)
        return P.parse_responses(data)

    def oai_embeddings(self, **kwargs: Any) -> t.OAIEmbeddingResponse:
        """``POST /v1/embeddings`` — OAI-compatible embeddings.

        Args:
            input: Text or list of texts to embed.
            model: Model name.
            encoding_format: ``"float"`` (default).
            **kwargs: Any OpenAI embeddings API parameter.

        Returns:
            :class:`~types.OAIEmbeddingResponse`.
        """
        data = self._post("/v1/embeddings", json=kwargs)
        return P.parse_oai_embedding(data)

    def oai_rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str = "",
        top_n: int | None = None,
    ) -> t.RerankResponse:
        """``POST /v1/rerank`` — OAI-compatible reranking.

        Same parameters as :meth:`rerank`.

        Returns:
            :class:`~types.RerankResponse`.
        """
        body: dict[str, Any] = {"query": query, "documents": documents}
        if model:
            body["model"] = model
        if top_n is not None:
            body["top_n"] = top_n
        data = self._post("/v1/rerank", json=body)
        return P.parse_rerank(data)

    # ====================================================================
    # Anthropic-Compatible Endpoints
    # ====================================================================

    def anthropic_messages(self, **kwargs: Any) -> t.AnthropicResponse:
        """``POST /v1/messages`` — Anthropic-compatible Messages API.

        Args:
            model: Model name (required).
            messages: List of message dicts (required).
            max_tokens: Maximum tokens to generate (default 4096).
            **kwargs: Any Anthropic Messages API parameter.

        Returns:
            :class:`~types.AnthropicResponse`.
        """
        kwargs.setdefault("stream", False)
        data = self._post("/v1/messages", json=kwargs)
        return P.parse_anthropic_response(data)

    def anthropic_messages_stream(self, **kwargs: Any) -> Iterator[t.AnthropicStreamEvent]:
        """``POST /v1/messages`` with streaming.

        Yields:
            :class:`~types.AnthropicStreamEvent` for each SSE event.
        """
        kwargs["stream"] = True
        sse_iter = self._stream_post("/v1/messages", json=kwargs)
        try:
            for event in sse_iter:
                yield P.parse_anthropic_stream_event(event.json())
        finally:
            sse_iter.close()

    def anthropic_count_tokens(self, **kwargs: Any) -> t.CountTokensResponse:
        """``POST /v1/messages/count_tokens`` — Count tokens without generating.

        Accepts the same parameters as :meth:`anthropic_messages` (except
        ``max_tokens`` is not required).

        Returns:
            :class:`~types.CountTokensResponse`.
        """
        data = self._post("/v1/messages/count_tokens", json=kwargs)
        return P.parse_count_tokens(data)

    # ====================================================================
    # Router / Multi-Model Endpoints
    # ====================================================================

    def router_models(self) -> t.ModelsResponse:
        """``GET /models`` — List all models in router mode.

        Returns:
            :class:`~types.ModelsResponse` with model status info.
        """
        data = self._get("/models")
        return P.parse_models(data)

    def model_load(self, model: str) -> t.ModelActionResponse:
        """``POST /models/load`` — Load a model in router mode.

        Args:
            model: Model identifier to load.

        Returns:
            :class:`~types.ModelActionResponse`.
        """
        data = self._post("/models/load", json={"model": model})
        return P.parse_model_action(data)

    def model_unload(self, model: str) -> t.ModelActionResponse:
        """``POST /models/unload`` — Unload a model in router mode.

        Args:
            model: Model identifier to unload.

        Returns:
            :class:`~types.ModelActionResponse`.
        """
        data = self._post("/models/unload", json={"model": model})
        return P.parse_model_action(data)
