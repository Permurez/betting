"""Riot Games API – ranked / match data (LoL)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from collectors.base import BaseCollector, CollectorResult
from config_loader import env, load_pipeline_config
from storage.db import insert_event

REGION_ROUTING = {
    "europe": ("euw1", "europe"),
    "euw": ("euw1", "europe"),
    "eune": ("eun1", "europe"),
    "na": ("na1", "americas"),
}


class RiotLoLCollector(BaseCollector):
    name = "riot_lol"

    def __init__(self, api_key: Optional[str] = None, summoner_names: Optional[List[str]] = None):
        self.api_key = api_key or env("RIOT_API_KEY")
        cfg = load_pipeline_config().get("sources", {}).get("riot_lol", {})
        region = cfg.get("region", "europe")
        self.platform, self.regional = REGION_ROUTING.get(region, ("euw1", "europe"))
        self.summoner_names = summoner_names or []

    def _headers(self) -> Dict[str, str]:
        return {"X-Riot-Token": self.api_key}

    def fetch(self) -> CollectorResult:
        if not self.api_key:
            return CollectorResult(
                name=self.name,
                success=False,
                data=pd.DataFrame(),
                message="Brak RIOT_API_KEY w .env – https://developer.riotgames.com/",
            )

        rows: List[dict] = []
        errors: List[str] = []
        now = datetime.now(timezone.utc)

        with httpx.Client(timeout=30.0, headers=self._headers()) as client:
            # Status serwera / wersja gry
            try:
                status_url = f"https://{self.platform}.api.riotgames.com/lol/status/v4/platform-data"
                resp = client.get(status_url)
                if resp.status_code == 200:
                    payload = resp.json()
                    insert_event("riot_lol", payload, entity_type="platform_status", fetched_at=now)
                    for inc in payload.get("incidents", []):
                        rows.append(
                            {
                                "type": "incident",
                                "name": inc.get("name", ""),
                                "status": "open",
                                "fetched_at": now,
                            }
                        )
            except Exception as exc:
                errors.append(f"status: {exc}")

            for name in self.summoner_names[:5]:
                try:
                    summoner = self._get_summoner(client, name)
                    if not summoner:
                        continue
                    puuid = summoner.get("puuid", "")
                    insert_event(
                        "riot_lol",
                        summoner,
                        entity_type="summoner",
                        entity_id=puuid,
                        fetched_at=now,
                    )
                    rank = self._get_rank(client, puuid)
                    if rank:
                        insert_event(
                            "riot_lol",
                            rank,
                            entity_type="rank",
                            entity_id=puuid,
                            fetched_at=now,
                        )
                        rows.append(
                            {
                                "type": "ranked",
                                "summoner": name,
                                "tier": rank.get("tier", ""),
                                "rank": rank.get("rank", ""),
                                "lp": rank.get("leaguePoints", 0),
                                "wins": rank.get("wins", 0),
                                "losses": rank.get("losses", 0),
                                "fetched_at": now,
                            }
                        )
                    match_ids = self._get_recent_matches(client, puuid, count=5)
                    for mid in match_ids:
                        match = self._get_match(client, mid)
                        if match:
                            insert_event(
                                "riot_lol",
                                match,
                                entity_type="match",
                                entity_id=mid,
                                fetched_at=now,
                            )
                            info = match.get("info", {})
                            rows.append(
                                {
                                    "type": "match",
                                    "match_id": mid,
                                    "game_mode": info.get("gameMode", ""),
                                    "game_duration": info.get("gameDuration", 0),
                                    "summoner": name,
                                    "fetched_at": now,
                                }
                            )
                except Exception as exc:
                    errors.append(f"{name}: {exc}")

        df = pd.DataFrame(rows)
        msg = f"Riot: {len(df)} rekordow."
        if errors:
            msg += f" Bledy: {'; '.join(errors[:3])}"
        if not self.summoner_names:
            msg += " Podaj summoner_names w config lub UI."

        return CollectorResult(
            name=self.name,
            success=len(df) > 0 or not errors,
            data=df,
            message=msg,
            meta={"errors": errors},
        )

    def _get_summoner(self, client: httpx.Client, name: str) -> Optional[Dict[str, Any]]:
        url = f"https://{self.platform}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{name}"
        resp = client.get(url)
        return resp.json() if resp.status_code == 200 else None

    def _get_rank(self, client: httpx.Client, puuid: str) -> Optional[Dict[str, Any]]:
        url = f"https://{self.platform}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
        resp = client.get(url)
        if resp.status_code != 200:
            return None
        data = resp.json()
        solo = [e for e in data if e.get("queueType") == "RANKED_SOLO_5x5"]
        return solo[0] if solo else (data[0] if data else None)

    def _get_recent_matches(self, client: httpx.Client, puuid: str, count: int = 5) -> List[str]:
        url = f"https://{self.regional}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        resp = client.get(url, params={"start": 0, "count": count})
        return resp.json() if resp.status_code == 200 else []

    def _get_match(self, client: httpx.Client, match_id: str) -> Optional[Dict[str, Any]]:
        url = f"https://{self.regional}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        resp = client.get(url)
        return resp.json() if resp.status_code == 200 else None
