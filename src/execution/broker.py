"""Abstrakcja brokera – LIVE celowo wylaczony w v1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class BetOrder:
    event_id: str
    home_team: str
    away_team: str
    side: str
    odds: float
    stake: float
    bookmaker: str = ""


class Broker(ABC):
    @abstractmethod
    def place_bet(self, order: BetOrder) -> str:
        """Zwraca id zlecenia."""


class LiveBroker(Broker):
    def place_bet(self, order: BetOrder) -> str:
        raise NotImplementedError(
            "LIVE betting wylaczone w v1. Uzyj EXECUTION_MODE=PAPER. "
            "Integracja z brokerem wymaga zgody prawnej i oficjalnego API bukmachera."
        )
