# oc-llama-server

A complete, type-safe Python client library for the [llama.cpp HTTP server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) API.

Covers **all 33 endpoints** across five API categories - core completions, OpenAI-compatible, Anthropic-compatible, reranking, and multi-model router - with both synchronous and asynchronous clients.

## Features

- **Full API coverage** - every endpoint from the latest llama.cpp server
- **Sync + Async** - `OCLlamaClient` and `AsyncOCLlamaClient`
- **Optional local server launcher** - start a wheel-backed `llama-server` process with `LocalLlamaServer`
- **Streaming support** - SSE-based real-time token streaming for completions, chat, infill, and Anthropic messages
- **Typed responses** - frozen dataclasses with convenience properties
- **Type-safe requests** - TypedDict definitions for all request bodies
- **Proper error handling** - exception hierarchy mapping HTTP status codes
- **PEP 561 compliant** - ships `py.typed` for mypy/pyright integration
- **Zero config** - sensible defaults, just point at your server

## Installation

```bash
pip install --extra-index-url "https://Olivercomputing.github.io/LlamaCPPServerWheel/simple/" oc-llama-server
```

Or install from source:

```bash
pip install --extra-index-url "https://Olivercomputing.github.io/LlamaCPPServerWheel/simple/" -e ".[dev]"
```

The extra index is required so `pip` can resolve the bundled
`llama-cpp-cuda-binaries` dependency from:
`https://Olivercomputing.github.io/LlamaCPPServerWheel/simple/`

## Development

The package uses a standard `src/` layout, with library code in `src/OCLlamaServer/`
and tests in `tests/`.

### Local editable install

Create a virtual environment, install the package in editable mode with dev tools,
then run the test suite and build a wheel locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url "https://Olivercomputing.github.io/LlamaCPPServerWheel/simple/" -e ".[dev]"
pytest -q
python -m build
```

### Wheel smoke test

After building, validate the built wheel in a fresh environment:

```bash
python -m venv .venv-wheel
source .venv-wheel/bin/activate
pip install --upgrade pip
pip install --extra-index-url "https://Olivercomputing.github.io/LlamaCPPServerWheel/simple/" dist/*.whl
python - <<'PY'
import importlib.resources as resources

import OCLlamaServer
from OCLlamaServer import LocalLlamaServer, OCLlamaClient

assert OCLlamaServer.__version__
assert OCLlamaClient
assert LocalLlamaServer
assert resources.files("OCLlamaServer").joinpath("py.typed").is_file()

print("wheel smoke test passed")
PY
```

## Quick Start

### Synchronous

```python
from OCLlamaServer import LocalLlamaServer

with LocalLlamaServer("/models/model.gguf", variant="cuda-12.8") as server:
    client = server.client()

    # Health check
    health = client.health()
    print(health.status)  # "ok"

    # Chat completion (OpenAI-compatible)
    result = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        temperature=0.7,
        max_tokens=100,
    )
    print(result.content)  # "4"

    # Streaming
    for chunk in client.chat_completion_stream(
        messages=[{"role": "user", "content": "Tell me a story"}]
    ):
        print(chunk.delta_content, end="", flush=True)
```

### Asynchronous

```python
import asyncio
from OCLlamaServer import AsyncOCLlamaClient, LocalLlamaServer

async def main():
    with LocalLlamaServer("/models/model.gguf", variant="cuda-12.8") as server:
        async with AsyncOCLlamaClient(server.base_url) as client:
            result = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello!"}]
            )
            print(result.content)

            # Async streaming
            async for chunk in client.chat_completion_stream(
                messages=[{"role": "user", "content": "Count to 10"}]
            ):
                print(chunk.delta_content, end="", flush=True)

asyncio.run(main())
```

## API Reference

### Client Configuration

```python
client = OCLlamaClient(
    base_url="http://localhost:8080",  # Server URL
    api_key="sk-my-key",              # Optional API key
    timeout=600.0,                     # Request timeout (seconds)
    api_prefix="/api",                 # Optional URL prefix
    default_headers={"X-Custom": "v"}, # Extra headers
)
```

### Core Endpoints

```python
# Text completion
result = client.completion(prompt="Once upon a time", n_predict=128, temperature=0.8)
print(result.content)

# Streaming completion
for chunk in client.completion_stream(prompt="The quick brown fox"):
    print(chunk.content, end="")

# Tokenize / Detokenize
tokens = client.tokenize("Hello world", with_pieces=True)
text = client.detokenize([123, 456, 789])

# Apply chat template (get formatted prompt without inference)
formatted = client.apply_template(messages=[
    {"role": "user", "content": "Hello"},
])
print(formatted.prompt)

# Embeddings (native)
embeddings = client.embedding("Hello world", embd_normalize=2)

# Reranking
ranked = client.rerank(
    query="What is a panda?",
    documents=["A bear", "A car", "The giant panda is a bear endemic to China"],
    top_n=2,
)
for r in ranked.results:
    print(f"  #{r.index}: score={r.relevance_score:.4f}")

# Code infill (Fill-in-the-Middle)
result = client.infill(
    input_prefix="def fibonacci(n):\n    ",
    input_suffix="\n    return result",
    n_predict=64,
)
```

### Server Management

```python
# Server properties
props = client.props()
print(f"Slots: {props.total_slots}, Template: {props.chat_template[:50]}")

# Slot monitoring
for slot in client.slots():
    print(f"Slot {slot.id}: processing={slot.is_processing}")

# Prometheus metrics
print(client.metrics())

# Slot cache management
client.slot_save(0, "checkpoint.bin")
client.slot_restore(0, "checkpoint.bin")
client.slot_erase(0)

# LoRA adapters
adapters = client.lora_adapters()
client.set_lora_adapters([{"id": 0, "scale": 0.5}])
```

### OpenAI-Compatible Endpoints

```python
# List models
models = client.models()
for m in models.data:
    print(f"{m.id} (params: {m.meta.n_params if m.meta else 'N/A'})")

# Chat completion with tools
result = client.chat_completion(
    model="my-model",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }],
    tool_choice="auto",
)

# Structured output with JSON schema
result = client.chat_completion(
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"colors": {"type": "array", "items": {"type": "string"}}},
        },
    },
)

# Reasoning / thinking models
result = client.chat_completion(
    messages=[{"role": "user", "content": "Solve: 15 * 23 + 7"}],
    reasoning_format="deepseek",
)
print(f"Thinking: {result.reasoning_content}")
print(f"Answer: {result.content}")

# OAI-compatible completions
result = client.oai_completion(prompt="The capital of France is", max_tokens=10)

# OAI-compatible embeddings
emb = client.oai_embeddings(input="Hello world", model="my-model")

# Responses API
resp = client.responses(
    model="gpt-4.1",
    instructions="You are helpful.",
    input="Hello!",
)
print(resp.output_text)
```

### Anthropic-Compatible Endpoints

```python
# Messages API
result = client.anthropic_messages(
    model="my-model",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(result.text)

# Streaming
for event in client.anthropic_messages_stream(
    model="my-model",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a joke"}],
):
    if event.type == "content_block_delta":
        print(event.delta.get("text", ""), end="")

# Count tokens
count = client.anthropic_count_tokens(
    model="my-model",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(f"Input tokens: {count.input_tokens}")
```

### Router / Multi-Model Endpoints

```python
# List available models (router mode)
models = client.router_models()
for m in models.data:
    status = m.status.get("value", "unknown") if m.status else "unknown"
    print(f"{m.id}: {status}")

# Load / unload models
client.model_load("ggml-org/gemma-3-4b-it-GGUF:Q4_K_M")
client.model_unload("ggml-org/gemma-3-4b-it-GGUF:Q4_K_M")
```

## Error Handling

```python
from OCLlamaServer import (
    OCLlamaClient,
    LlamaCppError,
    APIError,
    AuthenticationError,
    UnavailableError,
    ConnectionError,
    TimeoutError,
)

client = OCLlamaClient("http://localhost:8080")

try:
    result = client.chat_completion(messages=[{"role": "user", "content": "Hi"}])
except AuthenticationError:
    print("Invalid API key")
except UnavailableError:
    print("Model is still loading, try again later")
except ConnectionError:
    print("Cannot reach the server")
except TimeoutError:
    print("Request timed out")
except APIError as exc:
    print(f"API error [{exc.status_code}]: {exc.message}")
except LlamaCppError as exc:
    print(f"Client error: {exc.message}")
```

## Exception Hierarchy

```
LlamaCppError
├── APIError
│   ├── AuthenticationError  (401)
│   ├── BadRequestError      (400)
│   ├── NotFoundError        (404)
│   ├── RateLimitError       (429)
│   ├── InternalServerError  (500)
│   ├── NotSupportedError    (501)
│   └── UnavailableError     (503)
├── ConnectionError
├── TimeoutError
└── StreamError
```

## Complete Endpoint Map

| Client Method | HTTP | Path | Category |
|---|---|---|---|
| `health()` | GET | `/health` | Core |
| `completion()` | POST | `/completion` | Core |
| `completion_stream()` | POST | `/completion` | Core (streaming) |
| `tokenize()` | POST | `/tokenize` | Core |
| `detokenize()` | POST | `/detokenize` | Core |
| `apply_template()` | POST | `/apply-template` | Core |
| `embedding()` | POST | `/embedding` | Core |
| `embeddings()` | POST | `/embeddings` | Core |
| `rerank()` | POST | `/reranking` | Core |
| `infill()` | POST | `/infill` | Core |
| `infill_stream()` | POST | `/infill` | Core (streaming) |
| `props()` | GET | `/props` | Core |
| `set_props()` | POST | `/props` | Core |
| `slots()` | GET | `/slots` | Core |
| `metrics()` | GET | `/metrics` | Core |
| `slot_save()` | POST | `/slots/{id}?action=save` | Core |
| `slot_restore()` | POST | `/slots/{id}?action=restore` | Core |
| `slot_erase()` | POST | `/slots/{id}?action=erase` | Core |
| `lora_adapters()` | GET | `/lora-adapters` | Core |
| `set_lora_adapters()` | POST | `/lora-adapters` | Core |
| `models()` | GET | `/v1/models` | OpenAI |
| `oai_completion()` | POST | `/v1/completions` | OpenAI |
| `chat_completion()` | POST | `/v1/chat/completions` | OpenAI |
| `chat_completion_stream()` | POST | `/v1/chat/completions` | OpenAI (streaming) |
| `responses()` | POST | `/v1/responses` | OpenAI |
| `oai_embeddings()` | POST | `/v1/embeddings` | OpenAI |
| `oai_rerank()` | POST | `/v1/rerank` | OpenAI |
| `anthropic_messages()` | POST | `/v1/messages` | Anthropic |
| `anthropic_messages_stream()` | POST | `/v1/messages` | Anthropic (streaming) |
| `anthropic_count_tokens()` | POST | `/v1/messages/count_tokens` | Anthropic |
| `router_models()` | GET | `/models` | Router |
| `model_load()` | POST | `/models/load` | Router |
| `model_unload()` | POST | `/models/unload` | Router |

## License

MIT
