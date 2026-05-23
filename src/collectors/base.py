"""Bazowy interfejs zbieraczy danych."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class CollectorResult:
    name: str
    success: bool
    data: pd.DataFrame
    message: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseCollector(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self) -> CollectorResult:
        """Pobiera dane i zwraca znormalizowany DataFrame."""

    def safe_fetch(self) -> CollectorResult:
        try:
            return self.fetch()
        except Exception as exc:
            return CollectorResult(
                name=self.name,
                success=False,
                data=pd.DataFrame(),
                message=str(exc),
            )
