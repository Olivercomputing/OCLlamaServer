from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

import pytest

import OCLlamaServer
from OCLlamaServer.__about__ import __version__ as package_version


def test_runtime_version_comes_from_single_source() -> None:
    assert OCLlamaServer.__version__ == package_version


def test_hatch_uses_single_source_version_path() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        config = tomllib.load(f)

    assert "version" not in config["project"]
    assert config["project"]["dynamic"] == ["version"]
    assert config["tool"]["hatch"]["version"]["path"] == "src/OCLlamaServer/__about__.py"


def test_runtime_version_matches_installed_metadata_when_available() -> None:
    try:
        installed_version = version("oc-llama-server")
    except PackageNotFoundError:
        pytest.skip("package metadata unavailable without an installed distribution")

    assert OCLlamaServer.__version__ == installed_version
