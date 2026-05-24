"""Lekki scraper HLTV RSS (match headlines)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import feedparser
import pandas as pd

from collectors.base import BaseCollector, CollectorResult
from config_loader import load_sources_config


class HLTVScraperCollector(BaseCollector):
    name = "hltv_scraper"

    def __init__(self):
        cfg = load_sources_config().get("hltv", {})
        self.feed_url = cfg.get("feed_url", "https://www.hltv.org/rss/news")
        self.max_items = int(cfg.get("max_items", 200))

    def fetch(self) -> CollectorResult:
        feed = feedparser.parse(self.feed_url)
        rows: List[dict] = []
        now = datetime.now(timezone.utc)

        for entry in feed.entries[: self.max_items]:
            title = str(getattr(entry, "title", "") or "")
            if " vs " not in title.lower():
                continue

            parts = title.split(" vs ", 1)
            team_a = parts[0].strip()
            team_b = parts[1].split(" - ")[0].strip() if len(parts) > 1 else ""
            if not team_a or not team_b:
                continue

            published = getattr(entry, "published", "")
            rows.append(
                {
                    "source": "hltv",
                    "match_id": str(getattr(entry, "id", getattr(entry, "link", ""))),
                    "title": title,
                    "team_a": team_a,
                    "team_b": team_b,
                    "date": published,
                    "fetched_at": now,
                    "url": str(getattr(entry, "link", "")),
                }
            )

        df = pd.DataFrame(rows)
        return CollectorResult(
            name=self.name,
            success=not df.empty,
            data=df,
            message=f"HLTV scraper: {len(df)} wpisow meczowych.",
        )
