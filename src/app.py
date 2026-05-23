import streamlit as st
from sklearn.metrics import roc_auc_score

from main import (
    QuantModel,
    VectorizedBacktester,
    generate_features,
    generate_full_pipeline_data,
)

st.set_page_config(page_title="QuantBet ML", layout="wide")

st.title("QuantBet: ML Betting Engine")
st.markdown(
    "System wykrywania asymetrii (Value) na rynkach e-sportowych na podstawie statystyk i wyceny tlumu."
)

st.sidebar.header("Parametry Backtestu")
init_bankroll = st.sidebar.number_input("Kapital poczatkowy (PLN)", value=10000.0, step=1000.0)
kelly_fraction = st.sidebar.slider(
    "Ulamkowy Kelly (risk management)", min_value=0.01, max_value=0.50, value=0.10, step=0.01
)
min_ev = st.sidebar.slider("Minimalny prog EV", min_value=0.00, max_value=0.10, value=0.03, step=0.01)
n_matches = st.sidebar.select_slider(
    "Ilosc meczow w symulacji", options=[2000, 5000, 10000, 20000], value=5000
)

if st.sidebar.button("Uruchom pelny pipeline", use_container_width=True):
    with st.spinner("Trwa generowanie danych i inzynieria cech..."):
        raw_data = generate_full_pipeline_data(n_matches)
        processed_data = generate_features(raw_data)

    with st.spinner("Podzial czasowy i trening modelu LightGBM..."):
        n = len(processed_data)
        train_df = processed_data.iloc[: int(n * 0.7)]
        val_df = processed_data.iloc[int(n * 0.7) : int(n * 0.85)]
        test_df = processed_data.iloc[int(n * 0.85) :].copy()

        features = ["tA_winrate", "tB_winrate", "tA_kills", "tB_kills", "form_diff", "kills_diff"]
        target = "target"

        model = QuantModel(features)
        model.train(train_df[features], train_df[target], val_df[features], val_df[target])

        preds = model.predict_proba(test_df[features])
        test_df["p_model"] = preds
        auc_score = roc_auc_score(test_df[target], preds)

    with st.spinner("Symulacja rynkowa (backtest)..."):
        backtester = VectorizedBacktester(
            initial_bankroll=init_bankroll, kelly_fraction=kelly_fraction, min_ev=min_ev
        )
        results_df, metrics = backtester.run(test_df)

    st.success("Analiza zakonczona sukcesem.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Koncowy bankroll", f"{metrics['final_bankroll']:.2f} PLN", f"{metrics['roi']:.2f}%")
    col2.metric("Zawarte zaklady", int(metrics["total_bets"]))
    col3.metric("Win rate", f"{metrics['win_rate']:.1f}%")
    col4.metric("Test ROC-AUC", f"{auc_score:.3f}")
    col5.metric("Sredni CLV", f"{metrics['avg_clv']:.2f}%")

    st.markdown("### Krzywa kapitalu (Out-of-Sample)")
    chart_data = results_df[["date", "bankroll"]].set_index("date")
    st.line_chart(chart_data)

    with st.expander("Podglad decyzji algorytmu"):
        view_df = results_df[results_df["bet_side"] != "none"][
            [
                "date",
                "team_a",
                "team_b",
                "p_model",
                "odds_home",
                "odds_away",
                "bet_side",
                "bet_ev",
                "stake_fraction",
                "return_factor",
            ]
        ]
        st.dataframe(view_df.tail(15))
