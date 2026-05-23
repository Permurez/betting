import streamlit as st
import pandas as pd

from io_data import export_bets_csv, load_matches_csv, template_csv_bytes
from main import (
    FEATURE_COLUMNS,
    generate_full_pipeline_data,
    run_pipeline_from_dataframe,
)
from ui_pipeline import render_paper_panel, render_walk_forward_panel
from ui_sources import render_sources_panel

st.set_page_config(page_title="QuantBet ML", layout="wide")

st.title("QuantBet: ML Betting Engine")
st.caption("Value betting: model vs kurs bukmachera, backtest out-of-sample, CLV.")

st.sidebar.header("Zrodlo danych")
data_source = st.sidebar.radio(
    "Dane wejsciowe",
    ["Symulacja syntetyczna", "Plik CSV"],
    help="CSV: kolumny jak w szablonie data/sample_matches.csv (min. 200 wierszy).",
)

st.sidebar.header("Parametry backtestu")
init_bankroll = st.sidebar.number_input("Kapital poczatkowy (PLN)", value=10000.0, step=1000.0)
kelly_fraction = st.sidebar.slider(
    "Ulamkowy Kelly", min_value=0.01, max_value=0.50, value=0.10, step=0.01
)
min_ev = st.sidebar.slider("Minimalny prog EV", min_value=0.0, max_value=0.10, value=0.03, step=0.01)
save_model_after = st.sidebar.checkbox("Zapisz model po treningu", value=False)

n_matches = 5000
uploaded_df = None

if data_source == "Symulacja syntetyczna":
    n_matches = st.sidebar.select_slider(
        "Liczba meczow", options=[2000, 5000, 10000, 20000], value=5000
    )
else:
    st.sidebar.download_button(
        "Pobierz szablon CSV",
        data=template_csv_bytes(),
        file_name="sample_matches.csv",
        mime="text/csv",
    )
    uploaded = st.sidebar.file_uploader("Wgraj CSV z meczami", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = load_matches_csv(uploaded)
            st.sidebar.success(f"Wczytano {len(uploaded_df)} meczow.")
            if len(uploaded_df) < 200:
                st.sidebar.warning("Zalecane min. 200 meczow — wyniki moga byc niestabilne.")
        except ValueError as exc:
            st.sidebar.error(str(exc))

run_clicked = st.sidebar.button("Uruchom pelny pipeline", use_container_width=True, type="primary")

tab_backtest, tab_model, tab_data, tab_sources, tab_wf, tab_paper = st.tabs(
    ["Backtest", "Model i kalibracja", "Dane", "Zrodla i kursy", "Walk-forward", "Paper"]
)

with tab_sources:
    render_sources_panel()

with tab_wf:
    render_walk_forward_panel()

with tab_paper:
    render_paper_panel()

with tab_data:
    st.markdown(
        """
        **Wymagane kolumny CSV:** `match_id`, `date`, `team_a`, `team_b`,
        `team_a_kills`, `team_b_kills`, `odds_home`, `odds_away`,
        `closing_odds_home`, `closing_odds_away`, `target` (1 = wygrana team_a).

        Kursy zamkniecia sluza do metryki **CLV** (czy weszles lepiej niz rynek przed meczem).
        """
    )
    if uploaded_df is not None:
        st.dataframe(uploaded_df.head(20), use_container_width=True)

if run_clicked:
    if data_source == "Plik CSV" and uploaded_df is None:
        st.error("Wgraj plik CSV albo wybierz symulacje syntetyczna.")
        st.stop()

    try:
        with st.spinner("Przygotowanie danych i cech (bez lookahead bias)..."):
            if data_source == "Symulacja syntetyczna":
                raw_data = generate_full_pipeline_data(n_matches)
            else:
                raw_data = uploaded_df

            if data_source == "Plik CSV" and len(raw_data) < 200:
                st.warning("Malo danych — rozwaz wiekszy plik CSV dla wiarygodnego testu.")

            results_df, model, metrics, model_metrics = run_pipeline_from_dataframe(
                raw_data,
                initial_bankroll=init_bankroll,
                kelly_fraction=kelly_fraction,
                min_ev=min_ev,
                save_model=save_model_after,
            )

        st.session_state["results_df"] = results_df
        st.session_state["metrics"] = metrics
        st.session_state["model_metrics"] = model_metrics
        st.session_state["model"] = model
        st.session_state["raw_rows"] = len(raw_data)

        if save_model_after:
            st.session_state["model_saved"] = True

    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

if "results_df" in st.session_state:
    results_df = st.session_state["results_df"]
    metrics = st.session_state["metrics"]
    model_metrics = st.session_state["model_metrics"]

    with tab_backtest:
        st.success("Analiza zakonczona.")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Bankroll koncowy", f"{metrics['final_bankroll']:.0f} PLN", f"{metrics['roi']:.1f}% ROI")
        c2.metric("Zaklady", int(metrics["total_bets"]))
        c3.metric("Win rate", f"{metrics['win_rate']:.1f}%")
        c4.metric("Sredni CLV", f"{metrics['avg_clv']:.2f}%")
        c5.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.1f}%")
        c6.metric("Sharpe (zaklady)", f"{metrics['sharpe_ratio']:.2f}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Krzywa kapitalu (test OOS)")
            st.line_chart(results_df.set_index("date")["bankroll"])
        with right:
            st.markdown("#### Rozklad EV zawartych zakladow")
            placed = results_df[results_df["bet_side"] != "none"]
            if len(placed) > 0:
                st.bar_chart(placed["bet_ev"].clip(0, 0.25))
            else:
                st.info("Brak zakladow spelniajacych prog EV.")

        st.download_button(
            "Eksportuj zaklady do CSV",
            data=export_bets_csv(results_df),
            file_name="quantbet_bets.csv",
            mime="text/csv",
        )

        with st.expander("Ostatnie decyzje algorytmu"):
            view_cols = [
                "date",
                "team_a",
                "team_b",
                "p_model",
                "p_market_home",
                "edge_home_pct",
                "bet_side",
                "bet_ev",
                "stake_fraction",
                "clv_pct",
                "return_factor",
            ]
            view_df = results_df[results_df["bet_side"] != "none"][
                [c for c in view_cols if c in results_df.columns]
            ]
            st.dataframe(view_df.tail(25), use_container_width=True)

    with tab_model:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{model_metrics['roc_auc']:.3f}")
        m2.metric("Log-loss", f"{model_metrics['log_loss']:.4f}")
        m3.metric("Brier score", f"{model_metrics['brier_score']:.4f}", help="Im nizej, tym lepsza kalibracja")
        m4.metric("Srednie EV zakladow", f"{metrics['avg_bet_ev_pct']:.2f}%")
        if model_metrics.get("ensemble_weights"):
            st.caption(f"Ensemble: {model_metrics['ensemble_weights']}")

        st.markdown(
            """
            **Interpretacja:** dodatni CLV w dlugim terminie oznacza, ze wchodzisz lepiej niz kurs zamkniecia.
            ROI na danych syntetycznych bywa zawyzony — traktuj jako test architektury, nie obietnice zysku.
            """
        )

        if st.session_state.get("model_saved"):
            st.info("Model zapisany w folderze `models/quant_model.joblib`.")

        try:
            model = st.session_state["model"]
            if hasattr(model, "lgbm"):
                importances = model.lgbm.base_model.feature_importances_
                feat_names = model.features
            else:
                importances = model.base_model.feature_importances_
                feat_names = model.features
            imp_df = pd.DataFrame({"cecha": feat_names, "waznosc": importances}).sort_values(
                "waznosc", ascending=False
            )
            st.markdown("#### Waznosc cech (LightGBM)")
            st.bar_chart(imp_df.set_index("cecha"))
            if hasattr(model, "weights_dict"):
                st.caption(f"Wagi ensemble: {model.weights_dict()}")
        except Exception:
            pass

    with tab_data:
        st.metric("Mecze w zbiorze", st.session_state.get("raw_rows", "—"))

else:
    with tab_backtest:
        st.info("Ustaw parametry w panelu bocznym i kliknij **Uruchom pelny pipeline**.")
