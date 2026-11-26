from __future__ import annotations

from app import _apply_runtime_secrets


def test_streamlit_gemini_secret_enables_live_provider() -> None:
    environ: dict[str, str] = {}

    _apply_runtime_secrets({"GEMINI_API_KEY": "test-key"}, environ)

    assert environ == {
        "GEMINI_API_KEY": "test-key",
        "LLM_PROVIDER": "gemini",
        "LLM_MODE": "live",
    }


def test_streamlit_secret_does_not_override_explicit_environment() -> None:
    environ = {
        "GEMINI_API_KEY": "environment-key",
        "LLM_PROVIDER": "mock",
        "LLM_MODE": "mock",
    }

    _apply_runtime_secrets(
        {"GEMINI_API_KEY": "streamlit-key", "LLM_PROVIDER": "gemini", "LLM_MODE": "live"},
        environ,
    )

    assert environ == {
        "GEMINI_API_KEY": "environment-key",
        "LLM_PROVIDER": "mock",
        "LLM_MODE": "mock",
    }


def test_streamlit_secrets_without_api_key_leave_mock_defaults_untouched() -> None:
    environ: dict[str, str] = {}

    _apply_runtime_secrets({}, environ)

    assert environ == {}
