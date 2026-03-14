"""OCLlamaServer — A complete Python client for the llama.cpp HTTP server.

Provides both synchronous and asynchronous clients covering all API endpoints:
core completions, OpenAI-compatible chat/embeddings/responses, Anthropic-compatible
messages, reranking, infill, tokenization, slot management, LoRA control, metrics,
and multi-model router operations.

Quick start::

    from OCLlamaServer import OCLlamaClient

    client = OCLlamaClient("http://localhost:8080")
    result = client.chat_completion(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(result.content)

Async usage::

    from OCLlamaServer import AsyncOCLlamaClient

    async with AsyncOCLlamaClient("http://localhost:8080") as client:
        result = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(result.content)
"""

from .async_client import AsyncOCLlamaClient
from .client import OCLlamaClient
from .exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    InternalServerError,
    LlamaCppError,
    NotFoundError,
    NotSupportedError,
    RateLimitError,
    StreamError,
    TimeoutError,
    UnavailableError,
)
from .local_server import LocalLlamaServer
from .types import (
    # Enums
    ModelStatus,
    ReasoningFormat,
    StopType,
    # Response types
    AnthropicContentBlock,
    AnthropicResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
    ApplyTemplateResponse,
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    CompletionChunk,
    CompletionResponse,
    CountTokensResponse,
    DetokenizeResponse,
    EmbeddingResponse,
    HealthResponse,
    LoraAdapter,
    ModelActionResponse,
    ModelInfo,
    ModelMeta,
    ModelsResponse,
    OAIEmbeddingResponse,
    PropsResponse,
    RerankResponse,
    RerankResult,
    ResponsesResponse,
    SlotActionResponse,
    SlotInfo,
    Timings,
    TokenizeResponse,
    TokenProbability,
    TokenProbInfo,
)

__version__ = "1.0.0"

__all__ = [
    # Clients
    "OCLlamaClient",
    "AsyncOCLlamaClient",
    "LocalLlamaServer",
    # Exceptions
    "LlamaCppError",
    "APIError",
    "AuthenticationError",
    "BadRequestError",
    "NotFoundError",
    "UnavailableError",
    "NotSupportedError",
    "RateLimitError",
    "InternalServerError",
    "ConnectionError",
    "TimeoutError",
    "StreamError",
    # Enums
    "StopType",
    "ReasoningFormat",
    "ModelStatus",
    # Response types
    "HealthResponse",
    "CompletionResponse",
    "CompletionChunk",
    "TokenizeResponse",
    "DetokenizeResponse",
    "ApplyTemplateResponse",
    "EmbeddingResponse",
    "OAIEmbeddingResponse",
    "RerankResponse",
    "RerankResult",
    "ChatChoice",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "ResponsesResponse",
    "ModelInfo",
    "ModelMeta",
    "ModelsResponse",
    "PropsResponse",
    "SlotInfo",
    "SlotActionResponse",
    "LoraAdapter",
    "Timings",
    "TokenProbability",
    "TokenProbInfo",
    "AnthropicContentBlock",
    "AnthropicUsage",
    "AnthropicResponse",
    "AnthropicStreamEvent",
    "CountTokensResponse",
    "ModelActionResponse",
]
