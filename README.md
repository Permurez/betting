# QuantBet: E-Sports ML Betting Engine

Wektorowy silnik do analizy i obstawiania meczow e-sportowych wykorzystujacy uczenie maszynowe (LightGBM) do wyliczania przewagi (Expected Value) nad rynkiem.

## Struktura projektu
- src/main.py - logika ML, inzynieria cech, zabezpieczenie przed lookahead bias i wektorowy backtester.
- src/app.py - interaktywny interfejs uzytkownika w Streamlit do testowania strategii.

## Uruchomienie lokalne
1. Utworz srodowisko wirtualne i zainstaluj zaleznosci:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Uruchom aplikacje webowa:

```bash
streamlit run src\app.py
```
