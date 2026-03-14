"""Exception classes for the llama.cpp client library."""

from __future__ import annotations

from typing import Any


class LlamaCppError(Exception):
    """Base exception for all llama.cpp client errors."""

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(self.message)


class APIError(LlamaCppError):
    """Error returned by the llama.cpp server API.

    Attributes:
        status_code: HTTP status code.
        error_type: Error type string from the API (e.g. ``"authentication_error"``).
        message: Human-readable error message.
        body: Raw response body, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "unknown_error",
        body: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.error_type = error_type
        self.body = body
        super().__init__(f"[{status_code}] {error_type}: {message}")


class AuthenticationError(APIError):
    """401 — Invalid or missing API key."""


class BadRequestError(APIError):
    """400 — Malformed request (e.g. invalid grammar)."""


class NotFoundError(APIError):
    """404 — Endpoint or resource not found."""


class UnavailableError(APIError):
    """503 — Model loading or server not ready."""


class NotSupportedError(APIError):
    """501 — Endpoint is disabled on the server."""


class RateLimitError(APIError):
    """429 — Too many requests."""


class InternalServerError(APIError):
    """500 — Unexpected server error."""


class ConnectionError(LlamaCppError):  # noqa: A001
    """Could not connect to the llama.cpp server."""


class TimeoutError(LlamaCppError):  # noqa: A001
    """Request timed out."""


class StreamError(LlamaCppError):
    """Error while processing a Server-Sent Events stream."""


# ---------------------------------------------------------------------------
# Helper to map HTTP status codes to exception classes
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[int, type[APIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    404: NotFoundError,
    429: RateLimitError,
    500: InternalServerError,
    501: NotSupportedError,
    503: UnavailableError,
}


def raise_for_status(status_code: int, body: dict[str, Any] | None) -> None:
    """Raise an appropriate :class:`APIError` sub-class from a response."""
    if 200 <= status_code < 300:
        return

    error_data = (body or {}).get("error", {})
    message = error_data.get("message", "Unknown error")
    error_type = error_data.get("type", "unknown_error")

    exc_cls = _STATUS_MAP.get(status_code, APIError)
    raise exc_cls(
        message,
        status_code=status_code,
        error_type=error_type,
        body=body,
    )
