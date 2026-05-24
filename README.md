# QuantBet: E-Sports ML Betting Engine

Wektorowy silnik do analizy meczów e-sportowych: cechy bez wycieku czasowego (lookahead), LightGBM z kalibracją Platta, EV + fractional Kelly i backtest na zbiorze testowym (out-of-sample).

> **Uwaga:** Domyślnie pipeline ładuje dane z **API feedu** (PandaScore + Odds API + HLTV scraper) z fallbackiem do danych syntetycznych, gdy feed jest za mały/niedostępny.

## Struktura

| Plik | Opis |
|------|------|
| `src/main.py` | ETL, feature engineering, model, backtester, `run_pipeline()` |
| `src/app.py` | Dashboard Streamlit |
| `requirements.txt` | Zależności Python |
| `setup.ps1` | Instalacja pod Windows |

## Szybki start (Windows)

```powershell
cd c:\Users\Admin\Documents\Pliki_Studia\Programowanie\betting
.\setup.ps1          # venv + pip + test CLI
.\run.ps1            # dashboard Streamlit → http://localhost:8501
```

Tylko pipeline CLI (bez UI):

```powershell
.\venv\Scripts\python.exe src\main.py
```

**Python:** działa na 3.14 (u Ciebie). Stare sztywne wersje w `requirements.txt` nie budują się na 3.14 — używamy elastycznych minów; dokładne wersje po instalacji są w `requirements-lock.txt`.

## Parametry w UI

- **Kapitał początkowy** – bankroll startowy backtestu
- **Ułamek Kelly** – np. 0.1 = 10% pełnego Kelly (mniejsze wahania)
- **Min EV** – minimalna przewaga (np. 0.03 = 3%), żeby w ogóle obstawić mecz
- **Liczba meczów** – rozmiar symulacji syntetycznej

## Metryki

- **ROC-AUC** – czy model sortuje faworytów (na testie chronologicznym)
- **Log-loss / Brier** – jak dobrze model kalibruje prawdopodobieństwa (ważne przy EV)
- **CLV** (Closing Line Value) – czy kurs wejścia był lepszy niż kurs zamknięcia
- **Max drawdown** – najgłębszy spadek kapitału od szczytu
- **Sharpe** – stosunek zwrotu do zmienności na serii zakładów
- **ROI** – wynik symulacji z Kelly (na danych syntetycznych)

## Import własnych danych (CSV)

Wgraj plik w UI (zakładka **Dane**) lub użyj szablonu `data/sample_matches.csv`. Minimum **200 meczów** do podziału 70/15/15.

Po backteście: **Eksportuj zakłady do CSV** albo zapisz model (`models/quant_model.joblib`).

## Model matematyczny (domyslnie)

- **LightGBM** + **MLP (sklearn)** w **ensemble** (wagi z log-loss na walidacji)
- **Platt scaling** – kalibracja prawdopodobienstw
- **EV + fractional Kelly** – decyzja i stawka
- **Walk-forward** – wiele okien czasowych (`scripts/run_walk_forward.py`, zakladka UI)

Konfiguracja: `config/pipeline.yaml` (`models.ensemble: true`).

## Paper trading i chmura

```powershell
# Jeden cykl: fetch RSS + patch + kursy -> paper bet (SQLite)
.\venv\Scripts\python.exe scripts\run_worker.py --once

# Docker
docker compose up -d
```

`EXECUTION_MODE=PAPER` w `.env` — bez prawdziwych stawek. LIVE wylaczone w v1.

## Zrodla LoL

| Modul | API |
|-------|-----|
| `collectors/riot_lol.py` | Riot Developer (ranked, mecze) |
| `collectors/datadragon.py` | Patch + championy |
| `features/lol_features.py` | days_since_patch, rank_diff |

Klucze: `RIOT_API_KEY`, `ODDS_API_KEY` w `.env`. Summonery: `config/pipeline.yaml` → `sources.riot_lol.summoner_names`.

## Zrodla danych (wiele stron)

Zakladka **Zrodla i kursy** w aplikacji lub CLI:

```powershell
copy .env.example .env
# Uzupelnij ODDS_API_KEY (https://the-odds-api.com/)
.\venv\Scripts\python.exe scripts\fetch_sources.py
```

| Modul | Zrodlo | Opis |
|-------|--------|------|
| `collectors/news_rss.py` | RSS (HLTV, Dot Esports, …) | Aktualnosci, kontuzje, zmiany skladu |
| `collectors/odds_api.py` | The Odds API | Kursy wielu bukmacherow (EU) |
| `services/odds_comparator.py` | — | Gdzie najwyzszy kurs, arbitraz 2-way |
| `collectors/social_instagram.py` | Instagram | Opcjonalnie `instaloader` lub Graph API |
| `collectors/stats_pandascore.py` | PandaScore | Nadchodzace mecze / statystyki |

Konfiguracja URL-i i profili: `config/sources.yaml`. Cache: `data/cache/`.

**Instagram:** scraping moze naruszac ToS Meta — na produkcji uzyj oficjalnego API. **Bukmacherzy:** nie scrapujemy stron STS/Betclic (blokady, ToS) — uzywamy agregatora API.

## Storage i realizm backtestu

- `storage.backend: timescaledb` i `storage.postgres_dsn` w `config/pipeline.yaml` lub `DATABASE_URL` w `.env`.
- Kursy The Odds API są zapisywane jako snapshoty czasowe (as-of query do point-in-time).
- Normalizacja nazw drużyn: aliasy + fuzzy fallback (`config/sources.yaml` -> `team_mapping`).
- Realizm egzekucji (limity bukmachera, latency, slippage): `execution.realism` w `config/pipeline.yaml`.

## Licencja

Zobacz plik `LICENSE`.
