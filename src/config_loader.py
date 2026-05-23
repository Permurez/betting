"""Ladowanie config/sources.yaml i zmiennych srodowiskowych."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "sources.yaml"
PIPELINE_PATH = ROOT / "config" / "pipeline.yaml"


def load_env() -> None:
    load_dotenv(ROOT / ".env")


def load_sources_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_cache_dir() -> Path:
    cfg = load_sources_config()
    rel = cfg.get("storage", {}).get("cache_dir", "data/cache")
    path = ROOT / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def env(key: str, default: str = "") -> str:
    load_env()
    return os.getenv(key, default)


def load_pipeline_config() -> Dict[str, Any]:
    if not PIPELINE_PATH.exists():
        return {}
    with PIPELINE_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def use_ensemble() -> bool:
    return bool(load_pipeline_config().get("models", {}).get("ensemble", True))


def execution_mode() -> str:
    load_env()
    return os.getenv("EXECUTION_MODE") or load_pipeline_config().get("execution", {}).get("mode", "PAPER")
