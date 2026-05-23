from collectors.base import CollectorResult
from collectors.news_rss import RSSNewsCollector
from collectors.odds_api import TheOddsAPICollector
from collectors.social_instagram import InstagramCollector
from collectors.stats_pandascore import PandaScoreCollector

__all__ = [
    "CollectorResult",
    "RSSNewsCollector",
    "TheOddsAPICollector",
    "InstagramCollector",
    "PandaScoreCollector",
]
