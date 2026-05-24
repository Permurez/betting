import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import VectorizedBacktester
from pipeline.point_in_time import merge_odds_at_decision
from pipeline.real_feed import _prepare_odds_table, _prepare_pandascore_matches
from pipeline.team_mapping import TeamNameNormalizer
from storage.odds_timeseries import OddsTimeseriesStore


def test_team_mapping_alias_and_fuzzy():
    n = TeamNameNormalizer(alias_map={"navi": "Natus Vincere"}, fuzzy_threshold=0.7)
    alias = n.resolve("NaVi")
    assert alias.normalized == "Natus Vincere"
    fuzzy = n.resolve("G2 Esport", candidates=["G2 Esports", "FaZe"])
    assert fuzzy.normalized == "G2 Esports"
    assert not fuzzy.unresolved


def test_prepare_feed_merge_tables():
    normalizer = TeamNameNormalizer(alias_map={"navi": "Natus Vincere"}, fuzzy_threshold=0.8)
    odds = pd.DataFrame(
        [
            {"event_id": "E1", "home_team": "navi", "away_team": "G2", "outcome_name": "navi", "odds": 2.1, "commence_time": "2026-01-01T12:00:00Z"},
            {"event_id": "E1", "home_team": "navi", "away_team": "G2", "outcome_name": "G2", "odds": 1.8, "commence_time": "2026-01-01T12:00:00Z"},
        ]
    )
    matches = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "team_a": "Natus Vincere",
                "team_b": "G2",
                "begin_at": "2026-01-01T12:00:00Z",
                "target": 1,
                "team_a_kills": 16,
                "team_b_kills": 10,
            }
        ]
    )
    odds_table = _prepare_odds_table(odds, normalizer)
    match_table = _prepare_pandascore_matches(matches, normalizer)
    merged = match_table.merge(odds_table.drop(columns=["date"]), on="pair_key", how="left")
    assert merged.iloc[0]["odds_home"] == 2.1
    assert merged.iloc[0]["target"] == 1


def test_asof_odds_from_storage_fallback(monkeypatch):
    payload = {
        "event_id": "E1",
        "home_team": "A",
        "away_team": "B",
        "outcome_name": "A",
        "odds": 2.0,
    }
    fake_events = pd.DataFrame(
        [
            {
                "payload_json": json.dumps(payload),
                "fetched_at": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc).isoformat(),
            }
        ]
    )

    monkeypatch.setattr("storage.odds_timeseries.events_before", lambda *args, **kwargs: fake_events)
    out = OddsTimeseriesStore(dsn="").snapshots_before(datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc), event_ids=["E1"])
    assert len(out) == 1
    assert out.iloc[0]["event_id"] == "E1"


def test_merge_odds_asof_and_realism_limits(monkeypatch):
    snapshots = pd.DataFrame(
        [
            {"event_id": "M1", "home_team": "A", "away_team": "B", "fetched_at": "2026-01-01T10:00:00Z", "outcome_name": "A", "odds": 2.1},
            {"event_id": "M1", "home_team": "A", "away_team": "B", "fetched_at": "2026-01-01T10:00:00Z", "outcome_name": "B", "odds": 1.8},
            {"event_id": "M1", "home_team": "A", "away_team": "B", "fetched_at": "2026-01-01T11:59:00Z", "outcome_name": "A", "odds": 1.95},
            {"event_id": "M1", "home_team": "A", "away_team": "B", "fetched_at": "2026-01-01T11:59:00Z", "outcome_name": "B", "odds": 1.9},
        ]
    )
    monkeypatch.setattr(
        "pipeline.point_in_time.OddsTimeseriesStore.snapshots_before",
        lambda self, as_of, event_ids=None: snapshots,
    )
    matches = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "date": "2026-01-01T12:00:00Z",
                "team_a": "A",
                "team_b": "B",
                "closing_odds_home": 1.9,
                "closing_odds_away": 1.95,
                "target": 1,
                "p_model": 0.7,
            }
        ]
    )
    merged = merge_odds_at_decision(matches, odds_snapshots=None, latency_minutes=0)
    assert merged.iloc[0]["odds_home"] == 1.95

    bt = VectorizedBacktester(initial_bankroll=1000.0, kelly_fraction=0.3, min_ev=0.0)
    out, metrics = bt.run(
        merged.assign(
            odds_home=merged["odds_home"],
            odds_away=merged["odds_away"],
            closing_odds_home=1.9,
            closing_odds_away=1.95,
        )
    )
    assert out.iloc[0]["stake_amount"] <= 80.0  # market_max_stake_pct=0.08 w config
    assert metrics["total_bets"] == 1.0
