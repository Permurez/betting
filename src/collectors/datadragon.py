"""Data Dragon – wersje patchy i championy LoL."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import httpx
import pandas as pd

from collectors.base import BaseCollector, CollectorResult
from storage.db import insert_event

DDRAGON = "https://ddragon.leagueoflegends.com"


class DataDragonCollector(BaseCollector):
    name = "datadragon"

    def fetch(self) -> CollectorResult:
        rows: List[dict] = []
        now = datetime.now(timezone.utc)

        with httpx.Client(timeout=30.0) as client:
            versions_resp = client.get(f"{DDRAGON}/api/versions.json")
            if versions_resp.status_code != 200:
                return CollectorResult(
                    name=self.name,
                    success=False,
                    data=pd.DataFrame(),
                    message="Nie udalo sie pobrac wersji Data Dragon.",
                )

            versions = versions_resp.json()
            latest = versions[0] if versions else "14.1.1"
            insert_event(
                "datadragon",
                {"versions": versions[:10], "latest": latest},
                entity_type="versions",
                fetched_at=now,
            )

            champ_resp = client.get(f"{DDRAGON}/cdn/{latest}/data/en_US/champion.json")
            if champ_resp.status_code == 200:
                champs = champ_resp.json().get("data", {})
                insert_event(
                    "datadragon",
                    {"patch": latest, "champion_count": len(champs)},
                    entity_type="champions",
                    entity_id=latest,
                    fetched_at=now,
                )
                rows.append(
                    {
                        "patch_version": latest,
                        "champion_count": len(champs),
                        "fetched_at": now,
                    }
                )

        patch_df = pd.DataFrame(rows)
        if not patch_df.empty:
            patch_df["released_at"] = patch_df["fetched_at"]

        return CollectorResult(
            name=self.name,
            success=not patch_df.empty,
            data=patch_df,
            message=f"Data Dragon: patch {latest}, {len(rows)} wpisow.",
            meta={"latest_patch": latest},
        )
