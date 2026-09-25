from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest


def _stub_module(name: str, functions: list[str]) -> types.ModuleType:
    module = types.ModuleType(name)

    def _not_patched(*args: Any, **kwargs: Any) -> int:
        raise AssertionError(f"{name} handler was called without a test patch")

    for function_name in functions:
        setattr(module, function_name, _not_patched)

    return module


@pytest.fixture
def cli_module(monkeypatch: pytest.MonkeyPatch):
    """Import the CLI without requiring image/model runtime dependencies."""
    monkeypatch.setitem(
        sys.modules,
        "fly_det.preprocess",
        _stub_module(
            "fly_det.preprocess",
            ["preprocess_yolo_to_yolo", "preprocess_flat_to_flat"],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "fly_det.annotate",
        _stub_module(
            "fly_det.annotate",
            ["annotate_yolo_to_yolo", "annotate_flat_to_flat", "annotate_file_to_file"],
        ),
    )
    sys.modules.pop("fly_det.cli", None)
    module = importlib.import_module("fly_det.cli")

    yield module

    sys.modules.pop("fly_det.cli", None)


@pytest.fixture
def call_recorder() -> Callable[[int], tuple[dict[str, Any], Callable[..., int]]]:
    def factory(return_code: int = 0) -> tuple[dict[str, Any], Callable[..., int]]:
        recorded: dict[str, Any] = {}

        def handler(*args: Any, **kwargs: Any) -> int:
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return return_code

        return recorded, handler

    return factory
