"""Orkiestracja zbieraczy + zapis cache i SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from collectors.base import CollectorResult
from collectors.datadragon import DataDragonCollector
from collectors.news_rss import RSSNewsCollector
from collectors.odds_api import TheOddsAPICollector
from collectors.riot_lol import RiotLoLCollector
from collectors.social_instagram import InstagramCollector
from collectors.hype_crawler import HypeCrawler
from collectors.reddit_crawler import RedditCrawler
from collectors.stats_pandascore import PandaScoreCollector
from config_loader import get_cache_dir, load_pipeline_config
from storage.db import insert_event


class CollectorRunner:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or get_cache_dir()

    def run_all(
        self,
        include_news: bool = True,
        include_odds: bool = True,
        include_instagram: bool = False,
        include_stats: bool = False,
        include_riot: bool = False,
        include_patches: bool = True,
    ) -> Dict[str, CollectorResult]:
        results: Dict[str, CollectorResult] = {}
        now = datetime.now(timezone.utc)

        if include_patches:
            results["patches"] = DataDragonCollector().safe_fetch()
            self._save(results["patches"])

        if include_news:
            results["news"] = RSSNewsCollector().safe_fetch()
            self._save(results["news"])
            for _, row in results["news"].data.head(100).iterrows():
                insert_event("news_rss", row.to_dict(), entity_type="article", fetched_at=now)

        if include_odds:
            results["odds"] = TheOddsAPICollector().safe_fetch()
            self._save(results["odds"])
            for _, row in results["odds"].data.iterrows():
                insert_event(
                    "odds_api",
                    row.to_dict(),
                    entity_type="odds_line",
                    entity_id=str(row.get("event_id", "")),
                    fetched_at=now,
                )

        if include_riot:
            cfg = load_pipeline_config().get("sources", {}).get("riot_lol", {})
            names = cfg.get("summoner_names", [])
            results["riot"] = RiotLoLCollector(summoner_names=names).safe_fetch()
            self._save(results["riot"])

        if include_instagram:
            # Zastepujemy niestabilny Instagram calkowicie legalnym i otwartym na zawsze darmowym Web Crawlerem
            results["hype"] = HypeCrawler().safe_fetch()
            self._save(results["hype"])
            
            # Dokladamy reddita dla twardych opinii tłumu
            results["reddit"] = RedditCrawler().safe_fetch()
            self._save(results["reddit"])

        if include_stats:
            results["stats"] = PandaScoreCollector().safe_fetch()
            self._save(results["stats"])

        return results

    def _save(self, result: CollectorResult) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.cache_dir / f"{result.name}_{ts}.csv"
        if not result.data.empty:
            result.data.to_csv(path, index=False)
        result.meta["cache_path"] = str(path)
        return path

    def load_latest(self, name: str) -> Optional[pd.DataFrame]:
        files = sorted(self.cache_dir.glob(f"{name}_*.csv"), reverse=True)
        if not files:
            return None
        return pd.read_csv(files[0])
