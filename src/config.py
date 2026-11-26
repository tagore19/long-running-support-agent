"""Application settings and YAML business rule loader (Pydantic v2)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RefundPolicyConfig(BaseModel):
    """Deterministic refund policy rules loaded from YAML."""

    refund_window_days: int = Field(default=30, ge=1)
    maximum_refund_amount: float = Field(default=500.0, gt=0)
    currency: str = Field(default="BRL")
    require_delivered_order: bool = True
    allow_partial_refunds: bool = True
    allow_duplicate_refunds: bool = False


class ContextBudgetConfig(BaseModel):
    """Context window and adaptive memory compaction thresholds."""

    max_model_context_tokens: int = Field(default=12000, ge=500)
    compaction_trigger_tokens: int = Field(default=8500, ge=100)
    recent_messages_to_preserve: int = Field(default=10, ge=2)
    max_memory_tokens: int = Field(default=3000, ge=200)
    retrieval_top_k: int = Field(default=5, ge=1)
    max_episodes_in_context: int = Field(default=6, ge=1)
    episode_importance_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class GuardrailConfig(BaseModel):
    """Runtime execution guardrails for the LangGraph agent."""

    max_steps: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=5, ge=1)
    max_retries: int = Field(default=2, ge=0)
    tool_timeout_seconds: int = Field(default=10, ge=1)
    max_input_chars: int = Field(default=4000, ge=100)


class SupportPolicyConfig(BaseModel):
    """Support ticket and escalation policy configuration."""

    default_priority: str = "medium"
    valid_priorities: list[str] = Field(
        default_factory=lambda: ["low", "medium", "high", "urgent"]
    )
    escalation_triggers: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    """Top-level configuration parsed from config/business_rules.yaml."""

    policy_version: str = "2026.1"
    refund_policy: RefundPolicyConfig = Field(default_factory=RefundPolicyConfig)
    context: ContextBudgetConfig = Field(default_factory=ContextBudgetConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    support_policy: SupportPolicyConfig = Field(default_factory=SupportPolicyConfig)

    @property
    def agent_guardrails(self) -> GuardrailConfig:
        return self.guardrails

    @property
    def memory_budget(self) -> ContextBudgetConfig:
        return self.context


# Backward-compatible aliases
BusinessRulesConfig = AppConfig
MemoryBudgetConfig = ContextBudgetConfig
AgentGuardrailsConfig = GuardrailConfig


class Settings(BaseSettings):
    """Environment-driven application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    model_name: str = "gemini-2.5-flash"
    compaction_model_name: str = "gemini-2.0-flash-lite"
    db_path: str = "data/app.db"
    seed_db_path: str = "data/olist_seed.db"
    log_level: str = "INFO"

    llm_mode: Literal["mock", "live"] = "mock"
    llm_provider: Literal["mock", "gemini", "openrouter", "ollama"] = "mock"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    app_env: str = "development"
    app_db_path: str = ""
    checkpoint_db_path: str = "data/checkpoints.db"
    business_rules_path: str = "config/business_rules.yaml"
    trace_log_path: str = "data/traces/agent_traces.jsonl"
    agent_mode: Literal["improved", "baseline"] = "improved"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    @property
    def effective_db_path(self) -> str:
        return self.app_db_path or self.db_path

    @property
    def effective_llm_provider(self) -> str:
        if self.llm_mode == "mock" and not self.gemini_api_key and self.llm_provider == "gemini":
            return "mock"
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            return "mock"
        return self.llm_provider

    def resolve_path(self, path_str: str) -> Path:
        """Resolve relative paths against the repository root or CWD."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / p
        if candidate.parent.exists() or p.parts[0] in {"data", "config", "src", "evals"}:
            return candidate
        return Path.cwd() / p


@lru_cache(maxsize=4)
def load_app_config(rules_path: str = "config/business_rules.yaml") -> AppConfig:
    """Load and validate application and business rule configuration from YAML."""
    path = Path(rules_path)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / path
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)


def load_business_rules(rules_path: str = "config/business_rules.yaml") -> AppConfig:
    """Alias for load_app_config."""
    return load_app_config(rules_path)


def get_settings() -> Settings:
    """Return a freshly loaded Settings instance."""
    return Settings()
