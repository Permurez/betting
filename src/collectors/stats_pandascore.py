"""Statystyki meczow e-sport – PandaScore API (opcjonalnie)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import httpx
import pandas as pd

from collectors.base import BaseCollector, CollectorResult
from config_loader import env, load_sources_config

API_BASE = "https://api.pandascore.co"


class PandaScoreCollector(BaseCollector):
    name = "pandascore"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or env("PANDASCORE_API_KEY")
        cfg = load_sources_config().get("stats", {}).get("pandascore", {})
        self.enabled = cfg.get("enabled", False)
        self.games: List[str] = cfg.get("games", ["csgo"])
        self.per_page = int(cfg.get("per_page", 100))

    def fetch(self) -> CollectorResult:
        if not self.enabled:
            return CollectorResult(
                name=self.name,
                success=False,
                data=pd.DataFrame(),
                message="PandaScore wylaczony w config/sources.yaml (stats.pandascore.enabled: true).",
            )
        if not self.api_key:
            return CollectorResult(
                name=self.name,
                success=False,
                data=pd.DataFrame(),
                message="Brak PANDASCORE_API_KEY w .env",
            )

        rows: List[dict] = []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        now = datetime.now(timezone.utc)

        with httpx.Client(timeout=30.0, headers=headers) as client:
            for game in self.games:
                for status in ("past", "upcoming"):
                    url = f"{API_BASE}/{game}/matches/{status}"
                    resp = client.get(url, params={"per_page": self.per_page})
                    if resp.status_code != 200:
                        continue
                    for m in resp.json():
                        opponents = m.get("opponents", [])
                        if len(opponents) < 2:
                            continue
                        team_a = opponents[0].get("opponent", {}) or {}
                        team_b = opponents[1].get("opponent", {}) or {}
                        winner_id = m.get("winner_id")
                        target = None
                        if winner_id is not None and team_a.get("id") and team_b.get("id"):
                            target = 1 if team_a.get("id") == winner_id else 0 if team_b.get("id") == winner_id else None
                        results_map = {
                            r.get("team_id"): r.get("score", 0)
                            for r in m.get("results", [])
                            if isinstance(r, dict)
                        }
                        rows.append(
                            {
                                "game": game,
                                "match_id": str(m.get("id")),
                                "name": m.get("name", ""),
                                "status": status,
                                "team_a": team_a.get("name", ""),
                                "team_b": team_b.get("name", ""),
                                "team_a_id": team_a.get("id"),
                                "team_b_id": team_b.get("id"),
                                "team_a_kills": int(results_map.get(team_a.get("id"), 0) or 0),
                                "team_b_kills": int(results_map.get(team_b.get("id"), 0) or 0),
                                "begin_at": m.get("begin_at"),
                                "league": m.get("league", {}).get("name", ""),
                                "serie": m.get("serie", {}).get("full_name", ""),
                                "target": target,
                                "fetched_at": now,
                            }
                        )

        df = pd.DataFrame(rows)
        return CollectorResult(
            name=self.name,
            success=not df.empty,
            data=df,
            message=f"Pobrano {len(df)} meczow PandaScore (past+upcoming).",
        )
