"""Helpers for launching a local llama.cpp server from bundled binaries."""

from __future__ import annotations

import importlib
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import httpx

from .async_client import AsyncOCLlamaClient
from .client import OCLlamaClient

_STOP_WAIT_SECONDS = 5.0
_LOG_TAIL_BYTES = 4096


class LocalLlamaServer:
    """Launch ``llama-server`` from the ``llama-cpp-cuda-binaries`` wheel."""

    def __init__(
        self,
        model_path: str,
        *,
        variant: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        server_args: Sequence[str] = (),
        startup_timeout: float = 60.0,
        poll_interval: float = 0.5,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.variant = variant
        self.host = host
        self.port = port
        self.server_args = tuple(server_args)
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval
        self.env = dict(env or {})
        self._process: subprocess.Popen[str] | None = None
        self._log_file: TextIO | None = None

    @property
    def base_url(self) -> str:
        """HTTP base URL for the launched server."""
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> LocalLlamaServer:
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def client(self, **kwargs: Any) -> OCLlamaClient:
        """Build a synchronous client for the launched server."""
        return OCLlamaClient(self.base_url, **kwargs)

    def async_client(self, **kwargs: Any) -> AsyncOCLlamaClient:
        """Build an async client for the launched server."""
        return AsyncOCLlamaClient(self.base_url, **kwargs)

    def start(self) -> LocalLlamaServer:
        """Launch ``llama-server`` and wait until ``/health`` is ready."""
        if self._process is not None and self._process.poll() is None:
            return self
        if self._process is not None:
            self.stop()

        variant_dir = self._resolve_variant_dir()
        executable = self._resolve_executable(variant_dir)
        env = self._build_env(variant_dir)
        cmd = [
            str(executable),
            "--model",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self.server_args,
        ]

        self._log_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.monotonic() + self.startup_timeout
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            while time.monotonic() < deadline:
                exit_code = self._process.poll()
                if exit_code is not None:
                    message = self._startup_error(
                        f"llama-server exited before becoming ready (exit code {exit_code})"
                    )
                    self.stop()
                    raise RuntimeError(message)

                try:
                    response = client.get(f"{self.base_url}/health")
                except httpx.ConnectError:
                    time.sleep(self.poll_interval)
                    continue
                except httpx.TimeoutException:
                    time.sleep(self.poll_interval)
                    continue

                if response.status_code == 200:
                    return self
                if response.status_code == 503:
                    time.sleep(self.poll_interval)
                    continue

                message = self._startup_error(
                    f"llama-server returned unexpected status {response.status_code} during startup"
                )
                self.stop()
                raise RuntimeError(message)

        message = self._startup_error(
            f"Timed out waiting for llama-server to become ready at {self.base_url}"
        )
        self.stop()
        raise TimeoutError(message)

    def stop(self) -> None:
        """Stop the launched server if it is running."""
        process = self._process
        if process is None:
            self._close_log_file()
            return

        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=_STOP_WAIT_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=_STOP_WAIT_SECONDS)
            else:
                process.wait(timeout=0)
        finally:
            self._process = None
            self._close_log_file()

    def _resolve_variant_dir(self) -> Path:
        module = _binaries_module()
        available = module.available_variants()
        if not available:
            raise FileNotFoundError("No llama.cpp binary variants are available in llama_cpp_cuda_binaries.")

        variant = self.variant
        if variant is None:
            if len(available) == 1:
                variant = available[0]
            else:
                choices = ", ".join(available)
                raise ValueError(
                    "Multiple llama.cpp binary variants are available. "
                    f"Specify one of: {choices}"
                )

        return Path(module.bin_path(variant))

    def _resolve_executable(self, variant_dir: Path) -> Path:
        candidates = [
            variant_dir / "llama-server",
            variant_dir / "bin" / "llama-server",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find llama-server in: {searched}")

    def _build_env(self, variant_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.env)

        path_entries = [str(variant_dir)]
        bin_dir = variant_dir / "bin"
        if bin_dir.is_dir():
            path_entries.append(str(bin_dir))
        env["PATH"] = _prepend_env_path(path_entries, env.get("PATH"))

        env["LD_LIBRARY_PATH"] = _prepend_env_path([str(variant_dir)], env.get("LD_LIBRARY_PATH"))
        return env

    def _startup_error(self, prefix: str) -> str:
        log_tail = self._log_tail()
        if log_tail:
            return f"{prefix}\n\nllama-server output:\n{log_tail}"
        return prefix

    def _log_tail(self) -> str:
        if self._log_file is None:
            return ""

        self._log_file.flush()
        self._log_file.seek(0, os.SEEK_END)
        size = self._log_file.tell()
        self._log_file.seek(max(size - _LOG_TAIL_BYTES, 0))
        return self._log_file.read().strip()

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


def _binaries_module() -> Any:
    try:
        return importlib.import_module("llama_cpp_cuda_binaries")
    except ImportError as exc:  # pragma: no cover - exercised in real installations
        raise RuntimeError(
            "llama_cpp_cuda_binaries is required for LocalLlamaServer. "
            "Install with: pip install --extra-index-url "
            "https://Olivercomputing.github.io/LlamaCPPServerWheel/simple/ "
            "oc-llama-server"
        ) from exc


def _prepend_env_path(entries: Sequence[str], existing: str | None) -> str:
    filtered = [entry for entry in entries if entry]
    if existing:
        filtered.append(existing)
    return os.pathsep.join(filtered)
