"""UI Streamlit: zrodla danych, kursy, social, statystyki."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from collectors.news_rss import RSSNewsCollector
from collectors.odds_api import TheOddsAPICollector
from collectors.social_instagram import InstagramCollector
from collectors.stats_pandascore import PandaScoreCollector
from config_loader import env, get_cache_dir
from services.odds_comparator import OddsComparator
from services.runner import CollectorRunner
from services.trend_signals import annotate_news


def render_sources_panel() -> None:
    st.subheader("Zrodla danych na zywo")
    st.caption(
        "RSS (aktualnosci), The Odds API (kursy wielu bukow), opcjonalnie Instagram i PandaScore. "
        "Klucze API w pliku `.env` (wzor: `.env.example`)."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    fetch_news = col_a.checkbox("Aktualnosci RSS", value=True)
    fetch_odds = col_b.checkbox("Kursy (API)", value=True)
    fetch_ig = col_c.checkbox("Instagram", value=False)
    fetch_stats = col_d.checkbox("PandaScore", value=False)

    ig_handles = st.text_input(
        "Profile Instagram (po przecinku, bez @)",
        value="",
        help="Wymaga: pip install instaloader oraz wpis w config/sources.yaml",
    )

    if st.button("Pobierz dane ze wszystkich zrodel", type="primary"):
        runner = CollectorRunner()
        with st.spinner("Pobieranie..."):
            if ig_handles.strip():
                handles = [h.strip() for h in ig_handles.split(",") if h.strip()]
                ig_result = InstagramCollector(handles=handles).safe_fetch()
                runner._save(ig_result)
                st.session_state["collector_ig"] = ig_result
            results = runner.run_all(
                include_news=fetch_news,
                include_odds=fetch_odds,
                include_instagram=False,
                include_stats=fetch_stats,
            )
            st.session_state["collector_results"] = results

        for key, res in results.items():
            if res.success:
                st.success(f"{key}: {res.message}")
            else:
                st.warning(f"{key}: {res.message}")

    st.divider()
    tab_odds, tab_news, tab_social, tab_stats = st.tabs(
        ["Porownanie kursow", "Aktualnosci", "Instagram", "Statystyki"]
    )

    with tab_odds:
        _render_odds_tab()

    with tab_news:
        _render_cached_or_fetch("news_rss", RSSNewsCollector, "Aktualnosci RSS")

    with tab_social:
        if "collector_ig" in st.session_state:
            res = st.session_state["collector_ig"]
            st.write(res.message)
            if not res.data.empty:
                st.dataframe(res.data, use_container_width=True)
        else:
            st.info("Zaznacz Instagram i podaj profile, potem pobierz dane.")

    with tab_stats:
        _render_cached_or_fetch("pandascore", PandaScoreCollector, "PandaScore")


def _render_cached_or_fetch(name: str, collector_cls, label: str) -> None:
    runner = CollectorRunner()
    cached = runner.load_latest(name)
    if cached is not None:
        st.caption(f"Ostatni cache: {get_cache_dir()}")
        if name == "news_rss":
            cached = annotate_news(cached)
            flagged = cached[cached["signal_any"]]
            st.metric("Wpisy z sygnalem (kontuzja/sklad/form)", len(flagged))
            if not flagged.empty:
                st.dataframe(
                    flagged[["published_at", "source", "title", "signal_injury", "signal_roster"]].head(30),
                    use_container_width=True,
                )
        st.dataframe(cached.head(50), use_container_width=True)
    else:
        st.info(f"Brak cache dla {label}. Uzyj przycisku pobierania powyzej.")
    if st.button(f"Odswiez tylko {label}", key=f"refresh_{name}"):
        res = collector_cls().safe_fetch()
        runner._save(res)
        st.session_state[f"collector_{name}"] = res
        st.rerun()


def _render_odds_tab() -> None:
    has_key = bool(env("ODDS_API_KEY"))
    if not has_key:
        st.warning(
            "Ustaw **ODDS_API_KEY** w `.env` (darmowa rejestracja: https://the-odds-api.com/). "
            "Bez klucza mozesz testowac porownywacz na wczesniej zapisanym CSV z cache."
        )

    runner = CollectorRunner()
    odds_df = runner.load_latest("odds_api")

    if odds_df is None and has_key:
        if st.button("Pobierz kursy teraz"):
            res = TheOddsAPICollector().safe_fetch()
            runner._save(res)
            st.session_state["odds_result"] = res
            st.rerun()
        return

    if "odds_result" in st.session_state and st.session_state["odds_result"].success:
        odds_df = st.session_state["odds_result"].data

    if odds_df is None or odds_df.empty:
        st.info("Brak danych kursowych. Pobierz dane lub dodaj klucz API.")
        _show_demo_odds_comparator()
        return

    st.success(f"Zaladowano {len(odds_df)} linii kursowych.")
    comp = OddsComparator(odds_df)

    summary = comp.event_summary()
    st.markdown("#### Gdzie najlepiej postawic (najwyzszy kurs vs srednia rynku)")
    st.dataframe(
        summary[
            [
                "commence_time",
                "home_team",
                "away_team",
                "best_home_odds",
                "best_home_book",
                "home_edge_vs_avg_pct",
                "best_away_odds",
                "best_away_book",
                "away_edge_vs_avg_pct",
            ]
        ],
        use_container_width=True,
    )

    arbs = comp.find_arbitrage(min_profit_pct=0.1)
    st.markdown("#### Arbitraz (rzadkie)")
    if arbs.empty:
        st.caption("Brak prostego arbitrazu 2-way powyzej 0.1%.")
    else:
        st.dataframe(arbs, use_container_width=True)

    with st.expander("Surowe linie kursowe"):
        st.dataframe(odds_df.head(100), use_container_width=True)

    st.download_button(
        "Eksportuj podsumowanie kursow CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="odds_comparison.csv",
        mime="text/csv",
    )


def _show_demo_odds_comparator() -> None:
    """Demo bez API – pokazuje dzialanie porownywacza."""
    st.markdown("#### Demo porownywacza (dane przykladowe)")
    demo = pd.DataFrame(
        [
            {"event_id": "1", "home_team": "NaVi", "away_team": "G2", "commence_time": "2026-05-25",
             "bookmaker": "BukA", "outcome_name": "NaVi", "odds": 1.85},
            {"event_id": "1", "home_team": "NaVi", "away_team": "G2", "commence_time": "2026-05-25",
             "bookmaker": "BukB", "outcome_name": "NaVi", "odds": 1.92},
            {"event_id": "1", "home_team": "NaVi", "away_team": "G2", "commence_time": "2026-05-25",
             "bookmaker": "BukA", "outcome_name": "G2", "odds": 2.05},
            {"event_id": "1", "home_team": "NaVi", "away_team": "G2", "commence_time": "2026-05-25",
             "bookmaker": "BukB", "outcome_name": "G2", "odds": 1.98},
        ]
    )
    st.dataframe(OddsComparator(demo).event_summary(), use_container_width=True)
