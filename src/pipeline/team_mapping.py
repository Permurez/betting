"""Normalizacja nazw druzyn: aliasy + fuzzy fallback."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

from config_loader import load_sources_config


def _clean_name(name: str) -> str:
    return " ".join((name or "").strip().lower().replace("-", " ").replace("_", " ").split())


@dataclass(frozen=True)
class TeamResolution:
    original: str
    normalized: str
    method: str
    confidence: float
    unresolved: bool = False


class TeamNameNormalizer:
    def __init__(self, alias_map: Dict[str, str] | None = None, fuzzy_threshold: float | None = None):
        cfg = load_sources_config().get("team_mapping", {})
        raw_aliases = cfg.get("aliases", {}) if alias_map is None else alias_map
        threshold = cfg.get("fuzzy_threshold", 0.86) if fuzzy_threshold is None else fuzzy_threshold

        self.aliases: Dict[str, str] = {_clean_name(k): v for k, v in raw_aliases.items() if k}
        self.fuzzy_threshold = float(threshold)

    def resolve(self, name: str, candidates: Iterable[str] | None = None) -> TeamResolution:
        if not name:
            return TeamResolution(name, "", "empty", 0.0, unresolved=True)

        cleaned = _clean_name(name)
        if cleaned in self.aliases:
            return TeamResolution(name, self.aliases[cleaned], "alias", 1.0, unresolved=False)

        if not candidates:
            return TeamResolution(name, name, "identity", 1.0, unresolved=False)

        best_name = ""
        best_score = -1.0
        for c in candidates:
            score = SequenceMatcher(None, cleaned, _clean_name(c)).ratio()
            if score > best_score:
                best_score = score
                best_name = c

        if best_score >= self.fuzzy_threshold and best_name:
            return TeamResolution(name, best_name, "fuzzy", float(best_score), unresolved=False)
        return TeamResolution(name, name, "unresolved", float(max(best_score, 0.0)), unresolved=True)

    def normalize_frame(
        self,
        df,
        columns: List[str],
        candidates: Iterable[str] | None = None,
    ):
        out = df.copy()
        unresolved_flags = []
        for col in columns:
            if col not in out.columns:
                continue
            resolutions = out[col].apply(lambda x: self.resolve(str(x), candidates=candidates))
            out[col] = resolutions.apply(lambda r: r.normalized)
            out[f"{col}_map_method"] = resolutions.apply(lambda r: r.method)
            out[f"{col}_map_confidence"] = resolutions.apply(lambda r: r.confidence)
            flag_col = f"{col}_unresolved"
            out[flag_col] = resolutions.apply(lambda r: r.unresolved)
            unresolved_flags.append(flag_col)
        if unresolved_flags:
            out["team_mapping_unresolved"] = out[unresolved_flags].any(axis=1)
        return out


def canonical_pair(team_a: str, team_b: str) -> Tuple[str, str]:
    return tuple(sorted([team_a, team_b]))
