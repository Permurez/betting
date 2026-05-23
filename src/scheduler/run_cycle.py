"""Jeden cykl: fetch -> features -> predict -> paper bet."""

from __future__ import annotations

from typing import Any, Dict

from config_loader import load_pipeline_config
from execution.paper_trader import PaperTrader
from services.odds_comparator import OddsComparator
from services.runner import CollectorRunner


def run_cycle(summoner_names: list[str] | None = None) -> Dict[str, Any]:
    cfg = load_pipeline_config()
    exec_cfg = cfg.get("execution", {})
    report: Dict[str, Any] = {"steps": []}

    runner = CollectorRunner()
    results = runner.run_all(
        include_news=True,
        include_odds=True,
        include_patches=True,
        include_riot=True,
    )
    for key, res in results.items():
        report["steps"].append({key: res.message})

    odds = results.get("odds")
    if odds is None or odds.data.empty:
        report["status"] = "no_odds"
        return report

    comp = OddsComparator(odds.data)
    summary = comp.event_summary()

    trader = PaperTrader(
        bankroll=exec_cfg.get("initial_bankroll", 10000),
        kelly_fraction=exec_cfg.get("kelly_fraction", 0.1),
        min_ev=exec_cfg.get("min_ev", 0.03),
    )

    pseudo = summary.copy()
    pseudo["p_model"] = 0.5
    pseudo["odds_home"] = pseudo["best_home_odds"]
    pseudo["odds_away"] = pseudo["best_away_odds"]
    pseudo["team_a"] = pseudo["home_team"]
    pseudo["team_b"] = pseudo["away_team"]
    pseudo["match_id"] = pseudo["event_id"]

    placed = trader.place_from_signals(pseudo)
    report["paper_bets_placed"] = len(placed)
    report["paper_summary"] = trader.summary()
    report["status"] = "ok"
    report["events_compared"] = len(summary)
    return report
