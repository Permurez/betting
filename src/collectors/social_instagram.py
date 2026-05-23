"""Monitor social – Instagram (szkielet + opcjonalny instaloader).

UWAGA: Scraping Instagram moze naruszac regulamin Meta.
Produkcja: Instagram Graph API + konto biznesowe (INSTAGRAM_ACCESS_TOKEN w .env).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pandas as pd

from collectors.base import BaseCollector, CollectorResult
from config_loader import env, load_sources_config


class InstagramCollector(BaseCollector):
    name = "instagram"

    def __init__(self, handles: List[str] | None = None, max_posts: int = 15):
        cfg = load_sources_config().get("social", {}).get("instagram", {})
        self.handles = handles or cfg.get("handles", [])
        self.max_posts = max_posts or cfg.get("max_posts_per_handle", 15)
        self.token = env("INSTAGRAM_ACCESS_TOKEN")

    def fetch(self) -> CollectorResult:
        if not self.handles:
            return CollectorResult(
                name=self.name,
                success=False,
                data=pd.DataFrame(),
                message="Brak profili w config/sources.yaml (social.instagram.handles).",
            )

        if self.token:
            return self._fetch_graph_api()

        return self._fetch_instaloader()

    def _fetch_graph_api(self) -> CollectorResult:
        # Miejsce na oficjalne API – wymaga Business/Creator + Facebook App
        return CollectorResult(
            name=self.name,
            success=False,
            data=pd.DataFrame(),
            message=(
                "Token Instagram ustawiony, ale integracja Graph API wymaga "
                "konfiguracji media_id per konto. Uzyj instaloader lub uzupelnij _fetch_graph_api."
            ),
        )

    def _fetch_instaloader(self) -> CollectorResult:
        try:
            import instaloader
        except ImportError:
            return CollectorResult(
                name=self.name,
                success=False,
                data=pd.DataFrame(),
                message="Zainstaluj: pip install instaloader (opcjonalna zaleznosc).",
            )

        loader = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_comments=False,
            save_metadata=False,
        )
        rows: List[dict] = []
        errors: List[str] = []

        for handle in self.handles:
            handle = handle.lstrip("@")
            try:
                profile = instaloader.Profile.from_username(loader.context, handle)
                for post in profile.get_posts():
                    if len([r for r in rows if r["handle"] == handle]) >= self.max_posts:
                        break
                    rows.append(
                        {
                            "handle": handle,
                            "post_id": post.shortcode,
                            "caption": (post.caption or "")[:1000],
                            "likes": post.likes,
                            "comments": post.comments,
                            "posted_at": post.date_utc.replace(tzinfo=timezone.utc),
                            "url": f"https://www.instagram.com/p/{post.shortcode}/",
                            "is_video": post.is_video,
                        }
                    )
            except Exception as exc:
                errors.append(f"@{handle}: {exc}")

        df = pd.DataFrame(rows)
        msg = f"Pobrano {len(df)} postow z {len(self.handles)} profili."
        if errors:
            msg += f" Bledy: {'; '.join(errors)}"

        return CollectorResult(
            name=self.name,
            success=len(df) > 0,
            data=df,
            message=msg,
            meta={"errors": errors},
        )
