"""Tests for local llama-server process management."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from OCLlamaServer import AsyncOCLlamaClient, LocalLlamaServer, OCLlamaClient
from OCLlamaServer import local_server as LS


class FakeBinariesModule:
    def __init__(self, variants: dict[str, Path]) -> None:
        self._variants = variants

    def available_variants(self) -> list[str]:
        return sorted(self._variants)

    def bin_path(self, variant: str | None = None) -> Path:
        if variant is None:
            return Path("/unused")
        return self._variants[variant]


class FakeProcess:
    def __init__(
        self,
        poll_results: list[int | None] | None = None,
        *,
        wait_raises_timeout: bool = False,
    ) -> None:
        self._poll_results = list(poll_results or [None])
        self._last_poll = self._poll_results[-1]
        self.wait_raises_timeout = wait_raises_timeout
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        if self._poll_results:
            self._last_poll = self._poll_results.pop(0)
        self.returncode = self._last_poll
        return self._last_poll

    def terminate(self) -> None:
        self.terminated = True
        if not self.wait_raises_timeout:
            self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.wait_raises_timeout and not self.killed:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self.returncode or 0


class FakeHTTPClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)

    def __enter__(self) -> FakeHTTPClient:
        return self

    def __exit__(self, *args) -> None:
        return None

    def get(self, url: str):
        if not self._responses:
            raise AssertionError(f"Unexpected GET {url}")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(status_code=response)


def make_variant_dir(tmp_path: Path, *, in_bin: bool = False) -> Path:
    variant_dir = tmp_path / "cuda-12.8"
    if in_bin:
        target = variant_dir / "bin"
    else:
        target = variant_dir
    target.mkdir(parents=True)
    executable = target / "llama-server"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    return variant_dir


def test_local_server_selects_single_variant_and_starts(tmp_path, monkeypatch):
    variant_dir = make_variant_dir(tmp_path)
    binaries = FakeBinariesModule({"cuda-12.8": variant_dir})
    fake_process = FakeProcess([None, None])
    captured: dict[str, object] = {}

    monkeypatch.setattr(LS, "_binaries_module", lambda: binaries)
    monkeypatch.setattr(LS.httpx, "Client", lambda timeout=None: FakeHTTPClient([503, 200]))
    monkeypatch.setattr(LS.time, "sleep", lambda _: None)

    def fake_popen(cmd, env, stdout, stderr, text):
        captured["cmd"] = cmd
        captured["env"] = env
        stdout.write("starting\n")
        stdout.flush()
        return fake_process

    monkeypatch.setattr(LS.subprocess, "Popen", fake_popen)

    with LocalLlamaServer("/models/model.gguf", startup_timeout=1.0) as server:
        assert server.base_url == "http://127.0.0.1:8080"
        assert captured["cmd"] == [
            str(variant_dir / "llama-server"),
            "--model",
            "/models/model.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
        ]
        env = captured["env"]
        assert isinstance(env, dict)
        assert str(variant_dir) in env["PATH"]
        assert str(variant_dir) in env["LD_LIBRARY_PATH"]

    assert fake_process.terminated is True


def test_local_server_resolves_executable_from_bin_dir(tmp_path, monkeypatch):
    variant_dir = make_variant_dir(tmp_path, in_bin=True)
    binaries = FakeBinariesModule({"cuda-12.8": variant_dir})
    fake_process = FakeProcess([None])
    captured: dict[str, object] = {}

    monkeypatch.setattr(LS, "_binaries_module", lambda: binaries)
    monkeypatch.setattr(LS.httpx, "Client", lambda timeout=None: FakeHTTPClient([200]))
    monkeypatch.setattr(LS.time, "sleep", lambda _: None)

    def fake_popen(cmd, env, stdout, stderr, text):
        captured["cmd"] = cmd
        return fake_process

    monkeypatch.setattr(LS.subprocess, "Popen", fake_popen)

    with LocalLlamaServer("/models/model.gguf", startup_timeout=1.0):
        pass

    assert captured["cmd"][0] == str(variant_dir / "bin" / "llama-server")


def test_local_server_requires_variant_when_multiple(monkeypatch, tmp_path):
    binaries = FakeBinariesModule({
        "cuda-12.8": make_variant_dir(tmp_path / "a"),
        "cuda-12.9": make_variant_dir(tmp_path / "b"),
    })
    monkeypatch.setattr(LS, "_binaries_module", lambda: binaries)

    with pytest.raises(ValueError, match="Specify one of"):
        LocalLlamaServer("/models/model.gguf").start()


def test_local_server_raises_when_no_variants(monkeypatch):
    monkeypatch.setattr(LS, "_binaries_module", lambda: FakeBinariesModule({}))

    with pytest.raises(FileNotFoundError, match="No llama.cpp binary variants"):
        LocalLlamaServer("/models/model.gguf").start()


def test_local_server_raises_when_process_exits_early(tmp_path, monkeypatch):
    variant_dir = make_variant_dir(tmp_path)
    binaries = FakeBinariesModule({"cuda-12.8": variant_dir})
    fake_process = FakeProcess([1])

    monkeypatch.setattr(LS, "_binaries_module", lambda: binaries)
    monkeypatch.setattr(LS.httpx, "Client", lambda timeout=None: FakeHTTPClient([]))
    monkeypatch.setattr(LS.time, "sleep", lambda _: None)

    def fake_popen(cmd, env, stdout, stderr, text):
        stdout.write("failed fast\n")
        stdout.flush()
        return fake_process

    monkeypatch.setattr(LS.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="failed fast"):
        LocalLlamaServer("/models/model.gguf", startup_timeout=1.0).start()


def test_local_server_times_out_and_stops_process(tmp_path, monkeypatch):
    variant_dir = make_variant_dir(tmp_path)
    binaries = FakeBinariesModule({"cuda-12.8": variant_dir})
    fake_process = FakeProcess([None, None, None])
    monotonic_values = iter([0.0, 0.25, 0.5, 0.75])

    monkeypatch.setattr(LS, "_binaries_module", lambda: binaries)
    monkeypatch.setattr(
        LS.httpx,
        "Client",
        lambda timeout=None: FakeHTTPClient([httpx.ConnectError("down"), httpx.ConnectError("down")]),
    )
    monkeypatch.setattr(LS.time, "sleep", lambda _: None)
    monkeypatch.setattr(LS.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(LS.subprocess, "Popen", lambda *args, **kwargs: fake_process)

    with pytest.raises(TimeoutError, match="Timed out"):
        LocalLlamaServer("/models/model.gguf", startup_timeout=0.5, poll_interval=0.1).start()

    assert fake_process.terminated is True


def test_local_server_stop_kills_if_terminate_hangs(tmp_path, monkeypatch):
    variant_dir = make_variant_dir(tmp_path)
    binaries = FakeBinariesModule({"cuda-12.8": variant_dir})
    fake_process = FakeProcess([None], wait_raises_timeout=True)

    monkeypatch.setattr(LS, "_binaries_module", lambda: binaries)
    monkeypatch.setattr(LS.httpx, "Client", lambda timeout=None: FakeHTTPClient([200]))
    monkeypatch.setattr(LS.time, "sleep", lambda _: None)
    monkeypatch.setattr(LS.subprocess, "Popen", lambda *args, **kwargs: fake_process)

    server = LocalLlamaServer("/models/model.gguf", startup_timeout=1.0)
    server.start()
    server.stop()

    assert fake_process.terminated is True
    assert fake_process.killed is True


def test_local_server_client_helpers():
    server = LocalLlamaServer("/models/model.gguf", host="0.0.0.0", port=9000)
    client = server.client(api_key="abc")
    async_client = server.async_client(api_key="abc")

    assert isinstance(client, OCLlamaClient)
    assert isinstance(async_client, AsyncOCLlamaClient)
    assert client.base_url == "http://0.0.0.0:9000"
    assert async_client.base_url == "http://0.0.0.0:9000"

    client.close()
    import asyncio
    asyncio.run(async_client.close())


def test_top_level_exports_local_server():
    from OCLlamaServer import LocalLlamaServer as ExportedLocalLlamaServer

    assert ExportedLocalLlamaServer is LocalLlamaServer
