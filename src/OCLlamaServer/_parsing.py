"""Shared response-parsing helpers for sync and async clients."""

from __future__ import annotations

from typing import Any

from . import types as t


def _parse_token_probability(data: dict[str, Any]) -> t.TokenProbability:
    return t.TokenProbability(
        id=data.get("id", 0),
        token=data.get("token", ""),
        logprob=data.get("logprob"),
        prob=data.get("prob"),
        bytes=data.get("bytes", []),
    )


def _parse_token_prob_info(data: dict[str, Any]) -> t.TokenProbInfo:
    top_logprobs = [
        _parse_token_probability(item)
        for item in data.get("top_logprobs", [])
        if isinstance(item, dict)
    ]
    top_probs = [
        _parse_token_probability(item)
        for item in data.get("top_probs", [])
        if isinstance(item, dict)
    ]
    return t.TokenProbInfo(
        id=data.get("id", 0),
        token=data.get("token", ""),
        logprob=data.get("logprob"),
        prob=data.get("prob"),
        bytes=data.get("bytes", []),
        top_logprobs=top_logprobs,
        top_probs=top_probs,
    )


def parse_health(data: dict[str, Any]) -> t.HealthResponse:
    return t.HealthResponse(status=data.get("status", "ok"), raw=data)


def parse_completion(data: dict[str, Any]) -> t.CompletionResponse:
    timings_raw = data.get("timings")
    timings = t.Timings.from_dict(timings_raw) if timings_raw else None
    probs = [
        _parse_token_prob_info(item)
        for item in data.get("probs", [])
        if isinstance(item, dict)
    ]
    return t.CompletionResponse(
        content=data.get("content", ""),
        tokens=data.get("tokens", []),
        stop=data.get("stop", False),
        stop_type=data.get("stop_type", "none"),
        stopping_word=data.get("stopping_word", ""),
        model=data.get("model", ""),
        prompt=data.get("prompt"),
        tokens_cached=data.get("tokens_cached", 0),
        tokens_evaluated=data.get("tokens_evaluated", 0),
        truncated=data.get("truncated", False),
        generation_settings=data.get("generation_settings", {}),
        timings=timings,
        probs=probs,
        raw=data,
    )


def parse_completion_chunk(data: dict[str, Any]) -> t.CompletionChunk:
    return t.CompletionChunk(
        content=data.get("content", ""),
        tokens=data.get("tokens", []),
        stop=data.get("stop", False),
        stop_type=data.get("stop_type", "none"),
        stopping_word=data.get("stopping_word", ""),
        prompt_progress=data.get("prompt_progress"),
        raw=data,
    )


def parse_tokenize(data: dict[str, Any]) -> t.TokenizeResponse:
    return t.TokenizeResponse(tokens=data.get("tokens", []), raw=data)


def parse_detokenize(data: dict[str, Any]) -> t.DetokenizeResponse:
    return t.DetokenizeResponse(content=data.get("content", ""), raw=data)


def parse_apply_template(data: dict[str, Any]) -> t.ApplyTemplateResponse:
    return t.ApplyTemplateResponse(prompt=data.get("prompt", ""), raw=data)


def parse_native_embedding(data: Any) -> list[t.EmbeddingResponse]:
    if isinstance(data, list):
        return [
            t.EmbeddingResponse(
                embedding=item.get("embedding", []),
                index=item.get("index", i),
                raw=item,
            )
            for i, item in enumerate(data)
        ]
    return [t.EmbeddingResponse(
        embedding=data.get("embedding", []),
        index=data.get("index", 0),
        raw=data if isinstance(data, dict) else {},
    )]


def parse_oai_embedding(data: dict[str, Any]) -> t.OAIEmbeddingResponse:
    return t.OAIEmbeddingResponse(
        object=data.get("object", "list"),
        data=data.get("data", []),
        model=data.get("model", ""),
        usage=data.get("usage", {}),
        raw=data,
    )


def parse_rerank(data: dict[str, Any]) -> t.RerankResponse:
    results = []
    for r in data.get("results", []):
        results.append(t.RerankResult(
            index=r.get("index", 0),
            relevance_score=r.get("relevance_score", 0.0),
            document=r.get("document", {}).get("text") if isinstance(r.get("document"), dict) else r.get("document"),
        ))
    return t.RerankResponse(
        results=results,
        model=data.get("model", ""),
        raw=data,
    )


def _parse_chat_choice(raw: dict[str, Any]) -> t.ChatChoice:
    return t.ChatChoice(
        index=raw.get("index", 0),
        message=raw.get("message", {}),
        delta=raw.get("delta", {}),
        finish_reason=raw.get("finish_reason"),
    )


def parse_chat_completion(data: dict[str, Any]) -> t.ChatCompletionResponse:
    choices = [_parse_chat_choice(c) for c in data.get("choices", [])]
    timings_raw = data.get("timings")
    timings = t.Timings.from_dict(timings_raw) if timings_raw else None
    return t.ChatCompletionResponse(
        id=data.get("id", ""),
        object=data.get("object", "chat.completion"),
        created=data.get("created", 0),
        model=data.get("model", ""),
        choices=choices,
        usage=data.get("usage", {}),
        timings=timings,
        raw=data,
    )


def parse_chat_chunk(data: dict[str, Any]) -> t.ChatCompletionChunk:
    choices = [_parse_chat_choice(c) for c in data.get("choices", [])]
    return t.ChatCompletionChunk(
        id=data.get("id", ""),
        object=data.get("object", "chat.completion.chunk"),
        created=data.get("created", 0),
        model=data.get("model", ""),
        choices=choices,
        raw=data,
    )


def parse_responses(data: dict[str, Any]) -> t.ResponsesResponse:
    return t.ResponsesResponse(
        id=data.get("id", ""),
        object=data.get("object", "response"),
        output=data.get("output", []),
        raw=data,
    )


def parse_model_info(raw: dict[str, Any]) -> t.ModelInfo:
    meta_raw = raw.get("meta")
    meta = None
    if meta_raw and isinstance(meta_raw, dict):
        meta = t.ModelMeta(
            vocab_type=meta_raw.get("vocab_type", 0),
            n_vocab=meta_raw.get("n_vocab", 0),
            n_ctx_train=meta_raw.get("n_ctx_train", 0),
            n_embd=meta_raw.get("n_embd", 0),
            n_params=meta_raw.get("n_params", 0),
            size=meta_raw.get("size", 0),
        )
    return t.ModelInfo(
        id=raw.get("id", ""),
        object=raw.get("object", "model"),
        created=raw.get("created", 0),
        owned_by=raw.get("owned_by", "llamacpp"),
        meta=meta,
        in_cache=raw.get("in_cache"),
        path=raw.get("path"),
        status=raw.get("status"),
        raw=raw,
    )


def parse_models(data: dict[str, Any]) -> t.ModelsResponse:
    models = [parse_model_info(m) for m in data.get("data", [])]
    return t.ModelsResponse(
        object=data.get("object", "list"),
        data=models,
        raw=data,
    )


def parse_props(data: dict[str, Any]) -> t.PropsResponse:
    return t.PropsResponse(
        default_generation_settings=data.get("default_generation_settings", {}),
        total_slots=data.get("total_slots", 0),
        model_path=data.get("model_path", ""),
        chat_template=data.get("chat_template", ""),
        chat_template_caps=data.get("chat_template_caps", {}),
        modalities=data.get("modalities", {}),
        build_info=data.get("build_info", ""),
        is_sleeping=data.get("is_sleeping", False),
        raw=data,
    )


def parse_slots(data: list[dict[str, Any]]) -> list[t.SlotInfo]:
    return [
        t.SlotInfo(
            id=s.get("id", 0),
            id_task=s.get("id_task", 0),
            n_ctx=s.get("n_ctx", 0),
            speculative=s.get("speculative", False),
            is_processing=s.get("is_processing", False),
            params=s.get("params", {}),
            next_token=s.get("next_token", {}),
            raw=s,
        )
        for s in data
    ]


def parse_slot_action(data: dict[str, Any]) -> t.SlotActionResponse:
    return t.SlotActionResponse(
        id_slot=data.get("id_slot", 0),
        filename=data.get("filename", ""),
        n_saved=data.get("n_saved", 0),
        n_restored=data.get("n_restored", 0),
        n_read=data.get("n_read", 0),
        n_written=data.get("n_written", 0),
        n_erased=data.get("n_erased", 0),
        timings=data.get("timings", {}),
        raw=data,
    )


def parse_lora_adapters(data: list[dict[str, Any]]) -> list[t.LoraAdapter]:
    return [
        t.LoraAdapter(id=a.get("id", 0), path=a.get("path", ""), scale=a.get("scale", 0.0))
        for a in data
    ]


def parse_anthropic_response(data: dict[str, Any]) -> t.AnthropicResponse:
    content_blocks = []
    for block in data.get("content", []):
        content_blocks.append(t.AnthropicContentBlock(
            type=block.get("type", "text"),
            text=block.get("text", ""),
            id=block.get("id"),
            name=block.get("name"),
            input=block.get("input"),
        ))
    usage_raw = data.get("usage", {})
    usage = t.AnthropicUsage(
        input_tokens=usage_raw.get("input_tokens", 0),
        output_tokens=usage_raw.get("output_tokens", 0),
    )
    return t.AnthropicResponse(
        id=data.get("id", ""),
        type=data.get("type", "message"),
        role=data.get("role", "assistant"),
        content=content_blocks,
        model=data.get("model", ""),
        stop_reason=data.get("stop_reason"),
        stop_sequence=data.get("stop_sequence"),
        usage=usage,
        raw=data,
    )


def parse_anthropic_stream_event(data: dict[str, Any]) -> t.AnthropicStreamEvent:
    content_block = None
    if "content_block" in data:
        cb = data["content_block"]
        content_block = t.AnthropicContentBlock(
            type=cb.get("type", "text"),
            text=cb.get("text", ""),
            id=cb.get("id"),
            name=cb.get("name"),
            input=cb.get("input"),
        )
    return t.AnthropicStreamEvent(
        type=data.get("type", ""),
        delta=data.get("delta", {}),
        content_block=content_block,
        index=data.get("index", 0),
        message=data.get("message", {}),
        raw=data,
    )


def parse_count_tokens(data: dict[str, Any]) -> t.CountTokensResponse:
    return t.CountTokensResponse(input_tokens=data.get("input_tokens", 0), raw=data)


def parse_model_action(data: dict[str, Any]) -> t.ModelActionResponse:
    return t.ModelActionResponse(success=data.get("success", False), raw=data)
