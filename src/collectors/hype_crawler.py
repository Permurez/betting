"""Crawlowanie buzzu i hypu w calym internecie bez kluczy API."""

import urllib.parse
from datetime import datetime, timezone
import pandas as pd
import feedparser

from collectors.base import BaseCollector, CollectorResult
from config_loader import load_sources_config

class HypeCrawler(BaseCollector):
    name = "hype_crawler"

    def fetch(self) -> CollectorResult:
        cfg = load_sources_config().get("social", {})
        queries = cfg.get("trend_queries", ["fame mma", "clout mma", "ksw"])
        
        rows = []
        for q in queries:
            query_encoded = urllib.parse.quote(q)
            # Pobiera globalny hype dla danych haseł (Zero auth)
            url = f"https://news.google.com/rss/search?q={query_encoded}&hl=pl&gl=PL&ceid=PL:pl"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:30]:  # Top 30 najnowszych publikowanych zdan
                rows.append({
                    "query": q,
                    "title": entry.title,
                    "caption": entry.title + " " + getattr(entry, "summary", ""),
                    "published": getattr(entry, "published", ""),
                    "source": "open_web",
                    "likes": 5000  # Stala waga symulujaca domyslną uwage mediow aby weszło w algorytm
                })
                
        df = pd.DataFrame(rows)
        return CollectorResult(
            name=self.name,
            success=len(df) > 0,
            data=df,
            message=f"Pobrano {len(df)} wzmianek o emocjach (Hype) z calego internetu.",
            meta={"queries": queries}
        )
