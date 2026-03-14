"""Type definitions for llama.cpp server API requests and responses.

Uses :class:`TypedDict` for request bodies (so callers can pass plain dicts)
and :func:`dataclasses.dataclass` for response objects (immutable, with
attribute access).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict, Union


# ============================================================================
# Enums
# ============================================================================


class StopType(str, Enum):
    """Reason generation stopped."""
    NONE = "none"
    EOS = "eos"
    LIMIT = "limit"
    WORD = "word"


class ReasoningFormat(str, Enum):
    """Reasoning / thinking output format."""
    NONE = "none"
    DEEPSEEK = "deepseek"
    DEEPSEEK_LEGACY = "deepseek-legacy"
    AUTO = "auto"


class ModelStatus(str, Enum):
    """Model loading status in router mode."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"


# ============================================================================
# Request TypedDicts — callers may pass plain dicts that match these shapes
# ============================================================================


class CompletionRequest(TypedDict, total=False):
    """Body for ``POST /completion``."""
    prompt: Union[str, list[Any]]
    temperature: float
    dynatemp_range: float
    dynatemp_exponent: float
    top_k: int
    top_p: float
    min_p: float
    n_predict: int
    n_indent: int
    n_keep: int
    n_cmpl: int
    n_cache_reuse: int
    stream: bool
    stop: list[str]
    typical_p: float
    repeat_penalty: float
    repeat_last_n: int
    presence_penalty: float
    frequency_penalty: float
    dry_multiplier: float
    dry_base: float
    dry_allowed_length: int
    dry_penalty_last_n: int
    dry_sequence_breakers: list[str]
    xtc_probability: float
    xtc_threshold: float
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    grammar: str
    json_schema: dict[str, Any]
    seed: int
    ignore_eos: bool
    logit_bias: Union[list[list[Any]], dict[str, float]]
    n_probs: int
    min_keep: int
    t_max_predict_ms: int
    id_slot: int
    cache_prompt: bool
    return_tokens: bool
    samplers: list[str]
    timings_per_token: bool
    return_progress: bool
    post_sampling_probs: bool
    response_fields: list[str]
    lora: list[dict[str, Any]]


class ChatMessage(TypedDict, total=False):
    """A single chat message."""
    role: str
    content: Union[str, list[dict[str, Any]]]
    reasoning_content: str


class ChatCompletionRequest(TypedDict, total=False):
    """Body for ``POST /v1/chat/completions``."""
    model: str
    messages: list[ChatMessage]
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    n_predict: int
    max_tokens: int
    stream: bool
    stop: list[str]
    presence_penalty: float
    frequency_penalty: float
    repeat_penalty: float
    seed: int
    response_format: dict[str, Any]
    tools: list[dict[str, Any]]
    tool_choice: Union[str, dict[str, Any]]
    grammar: str
    json_schema: dict[str, Any]
    logit_bias: Union[list[list[Any]], dict[str, float]]
    n_probs: int
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    samplers: list[str]
    chat_template_kwargs: dict[str, Any]
    reasoning_format: str
    thinking_forced_open: bool
    parse_tool_calls: bool
    parallel_tool_calls: bool
    timings_per_token: bool
    lora: list[dict[str, Any]]


class ResponsesRequest(TypedDict, total=False):
    """Body for ``POST /v1/responses``."""
    model: str
    instructions: str
    input: str
    temperature: float
    top_p: float
    max_tokens: int
    stream: bool
    stop: list[str]


class EmbeddingRequest(TypedDict, total=False):
    """Body for ``POST /v1/embeddings``."""
    input: Union[str, list[str]]
    model: str
    encoding_format: str


class NativeEmbeddingRequest(TypedDict, total=False):
    """Body for ``POST /embedding`` (native, non-OAI)."""
    content: str
    embd_normalize: int


class RerankRequest(TypedDict, total=False):
    """Body for ``POST /v1/rerank``."""
    model: str
    query: str
    documents: list[str]
    top_n: int


class InfillRequest(TypedDict, total=False):
    """Body for ``POST /infill``."""
    input_prefix: str
    input_suffix: str
    input_extra: list[dict[str, str]]
    prompt: str
    # Also accepts all CompletionRequest fields
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    n_predict: int
    stream: bool
    stop: list[str]
    seed: int
    grammar: str
    cache_prompt: bool


class TokenizeRequest(TypedDict, total=False):
    """Body for ``POST /tokenize``."""
    content: str
    add_special: bool
    parse_special: bool
    with_pieces: bool


class DetokenizeRequest(TypedDict, total=False):
    """Body for ``POST /detokenize``."""
    tokens: list[int]


class ApplyTemplateRequest(TypedDict, total=False):
    """Body for ``POST /apply-template``."""
    messages: list[ChatMessage]


class AnthropicMessage(TypedDict, total=False):
    """A single message in Anthropic format."""
    role: str
    content: Union[str, list[dict[str, Any]]]


class AnthropicMessagesRequest(TypedDict, total=False):
    """Body for ``POST /v1/messages``."""
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    system: Union[str, list[dict[str, Any]]]
    temperature: float
    top_p: float
    top_k: int
    stop_sequences: list[str]
    stream: bool
    tools: list[dict[str, Any]]
    tool_choice: dict[str, Any]


class SlotActionRequest(TypedDict, total=False):
    """Body for slot save/restore actions."""
    filename: str


class LoraAdapterSetting(TypedDict):
    """Single LoRA adapter scale setting."""
    id: int
    scale: float


class ModelLoadRequest(TypedDict):
    """Body for ``POST /models/load``."""
    model: str


class ModelUnloadRequest(TypedDict):
    """Body for ``POST /models/unload``."""
    model: str


# ============================================================================
# Response dataclasses
# ============================================================================


@dataclass(frozen=True)
class HealthResponse:
    """Result from ``GET /health``."""
    status: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class TokenProbability:
    """A single token with its probability / logprob."""
    id: int
    token: str
    logprob: float | None = None
    prob: float | None = None
    bytes: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class TokenProbInfo:
    """Token probability info for one position in the output."""
    id: int
    token: str
    logprob: float | None = None
    prob: float | None = None
    bytes: list[int] = field(default_factory=list)
    top_logprobs: list[TokenProbability] = field(default_factory=list)
    top_probs: list[TokenProbability] = field(default_factory=list)


@dataclass(frozen=True)
class Timings:
    """Generation timing information."""
    prompt_n: int = 0
    prompt_ms: float = 0.0
    prompt_per_token_ms: float = 0.0
    prompt_per_second: float = 0.0
    predicted_n: int = 0
    predicted_ms: float = 0.0
    predicted_per_token_ms: float = 0.0
    predicted_per_second: float = 0.0
    cache_n: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Timings:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class CompletionResponse:
    """Result from ``POST /completion``."""
    content: str = ""
    tokens: list[int] = field(default_factory=list)
    stop: bool = False
    stop_type: str = "none"
    stopping_word: str = ""
    model: str = ""
    prompt: Any = None
    tokens_cached: int = 0
    tokens_evaluated: int = 0
    truncated: bool = False
    generation_settings: dict[str, Any] = field(default_factory=dict)
    timings: Timings | None = None
    probs: list[TokenProbInfo] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class CompletionChunk:
    """A single chunk from a streamed completion."""
    content: str = ""
    tokens: list[int] = field(default_factory=list)
    stop: bool = False
    stop_type: str = "none"
    stopping_word: str = ""
    prompt_progress: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class TokenizeResponse:
    """Result from ``POST /tokenize``."""
    tokens: list[Any]  # int or {"id": int, "piece": str|list[int]}
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class DetokenizeResponse:
    """Result from ``POST /detokenize``."""
    content: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ApplyTemplateResponse:
    """Result from ``POST /apply-template``."""
    prompt: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class EmbeddingResponse:
    """Result from embedding endpoints."""
    embedding: list[Any]  # list[float] or list[list[float]]
    index: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class OAIEmbeddingResponse:
    """Result from ``POST /v1/embeddings`` (OpenAI format)."""
    object: str = "list"
    data: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class RerankResult:
    """A single reranking result."""
    index: int
    relevance_score: float
    document: str | None = None


@dataclass(frozen=True)
class RerankResponse:
    """Result from reranking endpoints."""
    results: list[RerankResult] = field(default_factory=list)
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ChatChoice:
    """A single choice in a chat completion response."""
    index: int = 0
    message: dict[str, Any] = field(default_factory=dict)
    delta: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


def _extract_text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_optional_text(value: Any) -> str | None:
    if value is None:
        return None

    text = _extract_text_content(value)
    return text or None


@dataclass(frozen=True)
class ChatCompletionResponse:
    """Result from ``POST /v1/chat/completions``."""
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatChoice] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    timings: Timings | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def content(self) -> str:
        """Shortcut to first choice's message content."""
        if self.choices:
            msg = self.choices[0].message or self.choices[0].delta
            return _extract_text_content(msg.get("content", ""))
        return ""

    @property
    def reasoning_content(self) -> str | None:
        """Shortcut to first choice's reasoning content."""
        if self.choices:
            msg = self.choices[0].message or self.choices[0].delta
            return _extract_optional_text(msg.get("reasoning_content"))
        return None


@dataclass(frozen=True)
class ChatCompletionChunk:
    """A single chunk from a streamed chat completion."""
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[ChatChoice] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def delta_content(self) -> str:
        """Shortcut to first choice's delta content."""
        if self.choices:
            return _extract_text_content(self.choices[0].delta.get("content", ""))
        return ""


@dataclass(frozen=True)
class ResponsesResponse:
    """Result from ``POST /v1/responses``."""
    id: str = ""
    object: str = "response"
    output: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def output_text(self) -> str:
        """Concatenated text from all output items."""
        parts: list[str] = []
        for item in self.output:
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
        return "".join(parts)


@dataclass(frozen=True)
class ModelMeta:
    """Model metadata from ``/v1/models``."""
    vocab_type: int = 0
    n_vocab: int = 0
    n_ctx_train: int = 0
    n_embd: int = 0
    n_params: int = 0
    size: int = 0


@dataclass(frozen=True)
class ModelInfo:
    """A single model entry."""
    id: str = ""
    object: str = "model"
    created: int = 0
    owned_by: str = "llamacpp"
    meta: ModelMeta | None = None
    # Router-mode fields
    in_cache: bool | None = None
    path: str | None = None
    status: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ModelsResponse:
    """Result from ``GET /v1/models`` or ``GET /models``."""
    object: str = "list"
    data: list[ModelInfo] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class PropsResponse:
    """Result from ``GET /props``."""
    default_generation_settings: dict[str, Any] = field(default_factory=dict)
    total_slots: int = 0
    model_path: str = ""
    chat_template: str = ""
    chat_template_caps: dict[str, Any] = field(default_factory=dict)
    modalities: dict[str, bool] = field(default_factory=dict)
    build_info: str = ""
    is_sleeping: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class SlotInfo:
    """A single processing slot."""
    id: int = 0
    id_task: int = 0
    n_ctx: int = 0
    speculative: bool = False
    is_processing: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    next_token: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class SlotActionResponse:
    """Result from slot save / restore / erase actions."""
    id_slot: int = 0
    filename: str = ""
    n_saved: int = 0
    n_restored: int = 0
    n_read: int = 0
    n_written: int = 0
    n_erased: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class LoraAdapter:
    """A loaded LoRA adapter."""
    id: int
    path: str
    scale: float


@dataclass(frozen=True)
class AnthropicContentBlock:
    """A content block in an Anthropic response."""
    type: str = "text"
    text: str = ""
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


@dataclass(frozen=True)
class AnthropicUsage:
    """Token usage in Anthropic format."""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class AnthropicResponse:
    """Result from ``POST /v1/messages``."""
    id: str = ""
    type: str = "message"
    role: str = "assistant"
    content: list[AnthropicContentBlock] = field(default_factory=list)
    model: str = ""
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage = field(default_factory=AnthropicUsage)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def text(self) -> str:
        """Concatenated text from all text content blocks."""
        return "".join(b.text for b in self.content if b.type == "text")


@dataclass(frozen=True)
class AnthropicStreamEvent:
    """A single event from an Anthropic streaming response."""
    type: str = ""
    delta: dict[str, Any] = field(default_factory=dict)
    content_block: AnthropicContentBlock | None = None
    index: int = 0
    message: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class CountTokensResponse:
    """Result from ``POST /v1/messages/count_tokens``."""
    input_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ModelActionResponse:
    """Result from ``POST /models/load`` or ``POST /models/unload``."""
    success: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
