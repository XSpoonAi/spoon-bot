from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pymupdf_is_a_required_runtime_dependency() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    required = {item.split(">=", 1)[0].casefold() for item in project["dependencies"]}
    optional_document = {
        item.split(">=", 1)[0].casefold()
        for item in project["optional-dependencies"]["document"]
    }

    assert "pymupdf" in required
    assert "pymupdf" not in optional_document


def test_all_runtime_images_verify_pymupdf_import() -> None:
    for name in ("Dockerfile", "Dockerfile.modal", "Dockerfile.legacy"):
        content = (ROOT / name).read_text(encoding="utf-8")
        assert 'import fitz; assert fitz.open' in content
