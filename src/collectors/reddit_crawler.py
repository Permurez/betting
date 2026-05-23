"""API Crawler dla Reddit JSON (Bez koniecznosci API Key, wystarczy User-Agent)."""

import urllib.parse
from datetime import datetime
import pandas as pd
import httpx

from collectors.base import BaseCollector, CollectorResult
from config_loader import load_sources_config

class RedditCrawler(BaseCollector):
    name = "reddit_crawler"

    def fetch(self) -> CollectorResult:
        cfg = load_sources_config().get("social", {})
        queries = cfg.get("trend_queries", ["fame mma", "ksw"])
        
        rows = []
        # Reddit wymaga customowego User-Agenta zeby nie rzucal bledu 429 lub 403
        headers = {"User-Agent": "windows:quantbet:v1.0 (by /u/QuantBetAutomator)"}
        
        with httpx.Client(headers=headers, timeout=15.0) as client:
            for q in queries:
                query_encoded = urllib.parse.quote(q)
                # Szukamy najnowszych goracych postow globalnie
                url = f"https://www.reddit.com/search.json?q={query_encoded}&sort=new&limit=25"
                
                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        for child in data.get("data", {}).get("children", []):
                            post = child.get("data", {})
                            
                            # Hype zdefiniowany jako ilosc strzalek w gore (ups) i komentarzy
                            engagement = post.get("ups", 0) + post.get("num_comments", 0)
                            
                            rows.append({
                                "query": q,
                                "title": post.get("title", ""),
                                "caption": post.get("selftext", ""),
                                "source": "reddit",
                                "likes": engagement,
                                "published": post.get("created_utc", "")
                            })
                except Exception as e:
                    print(f"Blad pobierania z reddita dla {q}: {e}")
        
        df = pd.DataFrame(rows)
        return CollectorResult(
            name=self.name,
            success=len(df) > 0,
            data=df,
            message=f"Pobrano {len(df)} postow dyskusyjnych z Reddit dla Hype-u.",
            meta={"queries": queries}
        )
