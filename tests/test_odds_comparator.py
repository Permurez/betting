import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.odds_comparator import OddsComparator


def test_best_home_odds():
    df = pd.DataFrame(
        [
            {"event_id": "1", "home_team": "A", "away_team": "B", "commence_time": "x",
             "bookmaker": "X", "outcome_name": "A", "odds": 2.0},
            {"event_id": "1", "home_team": "A", "away_team": "B", "commence_time": "x",
             "bookmaker": "Y", "outcome_name": "A", "odds": 2.1},
            {"event_id": "1", "home_team": "A", "away_team": "B", "commence_time": "x",
             "bookmaker": "X", "outcome_name": "B", "odds": 1.9},
            {"event_id": "1", "home_team": "A", "away_team": "B", "commence_time": "x",
             "bookmaker": "Y", "outcome_name": "B", "odds": 1.85},
        ]
    )
    summary = OddsComparator(df).event_summary()
    assert summary.iloc[0]["best_home_odds"] == 2.1
    assert summary.iloc[0]["best_home_book"] == "Y"


def test_arbitrage_detection():
    df = pd.DataFrame(
        [
            {"event_id": "1", "home_team": "A", "away_team": "B", "commence_time": "x",
             "bookmaker": "X", "outcome_name": "A", "odds": 2.2},
            {"event_id": "1", "home_team": "A", "away_team": "B", "commence_time": "x",
             "bookmaker": "Y", "outcome_name": "B", "odds": 2.2},
        ]
    )
    arbs = OddsComparator(df).find_arbitrage()
    assert not arbs.empty
