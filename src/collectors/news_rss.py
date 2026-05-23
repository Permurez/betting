"""Aktualnosci z wielu serwisow przez RSS (legalne, stabilne)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import feedparser
import pandas as pd

from collectors.base import BaseCollector, CollectorResult
from config_loader import load_sources_config


class RSSNewsCollector(BaseCollector):
    name = "news_rss"

    def __init__(self, feeds: List[Dict[str, str]] | None = None):
        cfg = load_sources_config()
        self.feeds = feeds or cfg.get("news", {}).get("rss_feeds", [])

    def fetch(self) -> CollectorResult:
        rows: List[dict] = []
        errors: List[str] = []

        for feed in self.feeds:
            name = feed.get("name", feed.get("url", "unknown"))
            url = feed["url"]
            try:
                parsed = feedparser.parse(url)
                if getattr(parsed, "bozo", False) and not parsed.entries:
                    errors.append(f"{name}: parse error")
                    continue
                for entry in parsed.entries[:30]:
                    published = entry.get("published_parsed") or entry.get("updated_parsed")
                    if published:
                        dt = datetime(*published[:6], tzinfo=timezone.utc)
                    else:
                        dt = datetime.now(timezone.utc)
                    rows.append(
                        {
                            "source": name,
                            "title": entry.get("title", ""),
                            "link": entry.get("link", ""),
                            "summary": entry.get("summary", "")[:500],
                            "published_at": dt,
                        }
                    )
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

        msg = f"Pobrano {len(df)} wpisow z {len(self.feeds)} zrodel."
        if errors:
            msg += f" Ostrzezenia: {'; '.join(errors[:5])}"

        return CollectorResult(
            name=self.name,
            success=len(df) > 0 or len(errors) < len(self.feeds),
            data=df,
            message=msg,
            meta={"errors": errors, "feed_count": len(self.feeds)},
        )
