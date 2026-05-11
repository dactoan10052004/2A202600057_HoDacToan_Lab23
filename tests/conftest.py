"""Test configuration: force safe defaults regardless of .env or shell environment."""

import pytest


@pytest.fixture(autouse=True)
def _safe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override dangerous env vars so tests are always offline and deterministic."""
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "false")
    monkeypatch.setenv("USE_LLM", "false")
