"""Warstwa realizmu backtestu: limity, opoznienia, slippage."""

from __future__ import annotations

from dataclasses import dataclass

from config_loader import load_pipeline_config


@dataclass
class RealismConfig:
    enabled: bool = True
    bookmaker_min_stake: float = 5.0
    bookmaker_max_stake: float = 1000.0
    market_max_stake_pct: float = 0.08
    execution_latency_minutes: int = 2
    slippage_bps: float = 25.0

    @staticmethod
    def from_config() -> "RealismConfig":
        cfg = load_pipeline_config().get("execution", {}).get("realism", {})
        return RealismConfig(
            enabled=bool(cfg.get("enabled", True)),
            bookmaker_min_stake=float(cfg.get("bookmaker_min_stake", 5.0)),
            bookmaker_max_stake=float(cfg.get("bookmaker_max_stake", 1000.0)),
            market_max_stake_pct=float(cfg.get("market_max_stake_pct", 0.08)),
            execution_latency_minutes=int(cfg.get("execution_latency_minutes", 2)),
            slippage_bps=float(cfg.get("slippage_bps", 25.0)),
        )
