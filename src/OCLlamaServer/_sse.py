"""Server-Sent Events (SSE) stream parser.

Handles both sync (``httpx.Response``) and async (``httpx.AsyncClient``)
streaming.  Supports the ``data:`` / ``event:`` / ``id:`` fields used by
the llama.cpp server.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from .exceptions import StreamError


@dataclass
class SSEEvent:
    """A parsed Server-Sent Event."""
    data: str = ""
    event: str = ""
    id: str = ""
    retry: int | None = None

    def json(self) -> dict[str, Any]:
        """Parse the ``data`` field as JSON."""
        try:
            data = json.loads(self.data)
        except json.JSONDecodeError as exc:
            raise StreamError(f"Failed to parse SSE data as JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise StreamError("Expected SSE data to decode to a JSON object")

        return data


def _parse_sse_lines(lines: list[str]) -> SSEEvent | None:
    """Parse a block of SSE lines into an :class:`SSEEvent`."""
    event = SSEEvent()
    has_data = False
    data_parts: list[str] = []

    for line in lines:
        if line.startswith("data:"):
            value = line[5:].lstrip(" ")
            if value == "[DONE]":
                return None  # Stream complete
            data_parts.append(value)
            has_data = True
        elif line.startswith("event:"):
            event.event = line[6:].lstrip(" ")
        elif line.startswith("id:"):
            event.id = line[3:].lstrip(" ")
        elif line.startswith("retry:"):
            try:
                event.retry = int(line[6:].strip())
            except ValueError:
                pass

    if not has_data:
        return None

    event.data = "\n".join(data_parts)
    return event


# ── Synchronous iterator ────────────────────────────────────────────────────


class SSEIterator:
    """Iterate over SSE events from a synchronous ``httpx`` streaming response.

    Usage::

        with client.stream("POST", url, json=body) as response:
            for event in SSEIterator(response):
                data = event.json()
    """

    def __init__(self, response: Any, owner: Any | None = None) -> None:
        self._response = response
        self._owner = owner
        self._closed = False
        self._lines_iter = response.iter_lines()

    def __iter__(self) -> Iterator[SSEEvent]:
        return self._iter_events()

    def _iter_events(self) -> Iterator[SSEEvent]:
        buffer: list[str] = []

        for line in self._lines_iter:
            if line == "":
                if buffer:
                    event = _parse_sse_lines(buffer)
                    buffer = []
                    if event is not None:
                        yield event
                    # else: [DONE] or empty block → skip
            else:
                buffer.append(line)

        # Flush remaining
        if buffer:
            event = _parse_sse_lines(buffer)
            if event is not None:
                yield event

    def close(self) -> None:
        """Close the underlying response."""
        if self._closed:
            return

        self._closed = True
        if self._owner is not None:
            self._owner.__exit__(None, None, None)
            return

        self._response.close()


# ── Asynchronous iterator ───────────────────────────────────────────────────


class AsyncSSEIterator:
    """Iterate over SSE events from an async ``httpx`` streaming response.

    Usage::

        async with client.stream("POST", url, json=body) as response:
            async for event in AsyncSSEIterator(response):
                data = event.json()
    """

    def __init__(self, response: Any, owner: Any | None = None) -> None:
        self._response = response
        self._owner = owner
        self._closed = False
        self._lines_iter = response.aiter_lines()

    def __aiter__(self) -> AsyncIterator[SSEEvent]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[SSEEvent]:
        buffer: list[str] = []

        async for line in self._lines_iter:
            if line == "":
                if buffer:
                    event = _parse_sse_lines(buffer)
                    buffer = []
                    if event is not None:
                        yield event
            else:
                buffer.append(line)

        # Flush remaining
        if buffer:
            event = _parse_sse_lines(buffer)
            if event is not None:
                yield event

    async def close(self) -> None:
        """Close the underlying response."""
        if self._closed:
            return

        self._closed = True
        if self._owner is not None:
            await self._owner.__aexit__(None, None, None)
            return

        await self._response.aclose()
