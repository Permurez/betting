"""UI: walk-forward i paper trading."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from backtest.walk_forward import run_walk_forward
from config_loader import load_pipeline_config
from execution.paper_trader import PaperTrader
from pipeline.real_feed import load_pipeline_input_data
from storage.db import list_paper_bets


def render_walk_forward_panel() -> None:
    st.subheader("Walk-forward backtest (ensemble LGBM + MLP)")
    cfg = load_pipeline_config().get("backtest", {})
    n_folds = st.slider("Liczba foldow", 3, 10, cfg.get("walk_forward_folds", 5))
    source = st.selectbox("Zrodlo danych", ["API (domyslne)", "Syntetyczne"], index=0)
    n_matches = st.select_slider("Liczba meczow", [500, 1000, 2000, 5000, 10000], value=2000)
    use_ens = st.checkbox("Ensemble (LGBM + MLP)", value=True)

    if st.button("Uruchom walk-forward", type="primary"):
        with st.spinner("Trening wielu okien czasowych..."):
            raw = load_pipeline_input_data(
                n_matches=n_matches,
                source="api" if source.startswith("API") else "synthetic",
                fallback_to_synthetic=True,
            )
            try:
                results, summary = run_walk_forward(
                    raw,
                    n_folds=n_folds,
                    min_train_rows=max(cfg.get("min_train_rows", 500), 200),
                    use_ensemble=use_ens,
                )
            except ValueError as exc:
                st.error(str(exc))
                return

        st.session_state["wf_results"] = results
        st.session_state["wf_summary"] = summary

    if "wf_summary" in st.session_state:
        s = st.session_state["wf_summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Foldy", int(s.get("folds", 0)))
        c2.metric("Sredni ROC-AUC", f"{s.get('mean_roc_auc', 0):.3f}")
        c3.metric("Sredni log-loss", f"{s.get('mean_log_loss', 0):.4f}")
        c4.metric("Sredni CLV", f"{s.get('avg_clv', 0):.2f}%")
        if s.get("ensemble_weights"):
            st.caption(f"Wagi ensemble: {s['ensemble_weights']}")
        st.line_chart(st.session_state["wf_results"].set_index("date")["bankroll"])


def render_paper_panel() -> None:
    st.subheader("Paper trading (symulacja)")
    st.caption("EXECUTION_MODE=PAPER w .env — bez prawdziwych stawek.")

    if st.button("Podsumowanie paper bets"):
        bets = list_paper_bets(200)
        if bets.empty:
            st.info("Brak paper betow. Uruchom worker: python scripts/run_worker.py --once")
        else:
            st.dataframe(bets, use_container_width=True)
            trader = PaperTrader()
            st.json(trader.summary())
