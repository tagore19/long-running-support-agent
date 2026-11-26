"""Shared pytest fixtures for isolated SQLite testing."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.seed_data import seed_databases
from src.config import AppConfig, Settings, load_app_config
from src.database.repositories import SupportRepository


@pytest.fixture()
def rules() -> AppConfig:
    return load_app_config("config/business_rules.yaml")


@pytest.fixture()
def seeded_env(tmp_path: Path) -> dict[str, Path]:
    app_db = tmp_path / "app.db"
    seed_db = tmp_path / "olist_seed.db"
    ckpt_db = tmp_path / "checkpoints.db"
    trace_log = tmp_path / "traces.jsonl"
    seed_databases(seed_db_path=seed_db, app_db_path=app_db, reset=True)
    return {
        "app_db": app_db,
        "seed_db": seed_db,
        "ckpt_db": ckpt_db,
        "trace_log": trace_log,
    }


@pytest.fixture()
def repo(seeded_env: dict[str, Path]) -> SupportRepository:
    return SupportRepository(seeded_env["app_db"])


@pytest.fixture()
def seeded_db(seeded_env: dict[str, Path]) -> tuple[Path, SupportRepository]:
    return seeded_env["app_db"], SupportRepository(seeded_env["app_db"])


@pytest.fixture()
def test_settings(seeded_env: dict[str, Path]) -> Settings:
    return Settings(
        llm_mode="mock",
        llm_provider="mock",
        db_path=str(seeded_env["app_db"]),
        app_db_path=str(seeded_env["app_db"]),
        seed_db_path=str(seeded_env["seed_db"]),
        checkpoint_db_path=str(seeded_env["ckpt_db"]),
        trace_log_path=str(seeded_env["trace_log"]),
        agent_mode="improved",
    )
