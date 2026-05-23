"""Kursy z wielu bukmacherow – The Odds API (https://the-odds-api.com/)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
import pandas as pd

from collectors.base import BaseCollector, CollectorResult
from config_loader import env, load_sources_config

API_BASE = "https://api.the-odds-api.com/v4"


class TheOddsAPICollector(BaseCollector):
    name = "odds_api"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or env("ODDS_API_KEY")
        cfg = load_sources_config().get("odds", {})
        self.sports: List[str] = cfg.get("sports", ["esports_csgo"])
        self.regions = cfg.get("regions", "eu")
        self.markets = cfg.get("markets", "h2h")
        self.odds_format = cfg.get("odds_format", "decimal")

    def fetch(self) -> CollectorResult:
        if not self.api_key:
            return CollectorResult(
                name=self.name,
                success=False,
                data=pd.DataFrame(),
                message="Brak ODDS_API_KEY w pliku .env – zarejestruj sie na https://the-odds-api.com/",
            )

        all_rows: List[dict] = []
        remaining = None
        used = None

        with httpx.Client(timeout=30.0) as client:
            for sport in self.sports:
                url = f"{API_BASE}/sports/{sport}/odds"
                params = {
                    "apiKey": self.api_key,
                    "regions": self.regions,
                    "markets": self.markets,
                    "oddsFormat": self.odds_format,
                }
                resp = client.get(url, params=params)
                remaining = resp.headers.get("x-requests-remaining")
                used = resp.headers.get("x-requests-used")
                if resp.status_code != 200:
                    continue
                events = resp.json()
                all_rows.extend(self._flatten_events(events, sport))

        df = pd.DataFrame(all_rows)
        msg = f"Pobrano {len(df)} linii kursowych ({len(self.sports)} dyscyplin)."
        if remaining is not None:
            msg += f" Limit API: pozostalo {remaining} zapytan (uzyte: {used})."

        return CollectorResult(
            name=self.name,
            success=not df.empty,
            data=df,
            message=msg,
            meta={"requests_remaining": remaining, "requests_used": used},
        )

    @staticmethod
    def _flatten_events(events: List[Dict[str, Any]], sport: str) -> List[dict]:
        rows: List[dict] = []
        now = datetime.now(timezone.utc)
        for ev in events:
            event_id = ev.get("id", "")
            commence = ev.get("commence_time", "")
            home = ev.get("home_team", "")
            away = ev.get("away_team", "")
            for book in ev.get("bookmakers", []):
                book_key = book.get("key", "")
                book_title = book.get("title", book_key)
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        rows.append(
                            {
                                "sport": sport,
                                "event_id": event_id,
                                "commence_time": commence,
                                "home_team": home,
                                "away_team": away,
                                "bookmaker_key": book_key,
                                "bookmaker": book_title,
                                "outcome_name": outcome.get("name", ""),
                                "odds": float(outcome.get("price", 0)),
                                "fetched_at": now,
                            }
                        )
        return rows
