import httpx
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from main import run_pipeline_from_dataframe

def fetch_and_prepare_data():
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/E1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2021/E1.csv",
    ]
    
    dfs = []
    print("Pobieranie prawdziwych danych (Football-data.co.uk)...")
    for url in urls:
        print(f"Pobieram: {url}")
        df_temp = pd.read_csv(url, on_bad_lines="skip")
        dfs.append(df_temp)
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Usuwamy remisy, zeby uproscic zadanie jako binarna klasyfikacja Win/Loss
    df = df[df["FTR"] != "D"].copy()
    
    # Mapowanie do formatu projektu
    df["match_id"] = ["M_" + str(i) for i in range(len(df))]
    
    # Poprawa daty
    df["date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    
    df["team_a"] = df["HomeTeam"]
    df["team_b"] = df["AwayTeam"]
    df["team_a_kills"] = df["FTHG"] # Home goals
    df["team_b_kills"] = df["FTAG"] # Away goals
    
    # Kursy Bet365 jako otwarcie i Pinch/inne jako zamkniecie (symulacja) lub max
    df["odds_home"] = df["B365H"]
    df["odds_away"] = df["B365A"]
    
    # Realnie na tym zbiorze czesto nie ma kursow pinnacla na otwarcie/zamkniecie wyraznych
    # uzyjmy Pinnacle (PSH/PSA) jako closing odds (najostrzejszy buk) jesli dostepny, jesli nie to Max
    closing_h = df.get("PSH", df.get("MaxH", df["B365H"]))
    closing_a = df.get("PSA", df.get("MaxA", df["B365A"]))
    
    # Wypelnienie brakow z Bet365
    df["closing_odds_home"] = closing_h.fillna(df["B365H"])
    df["closing_odds_away"] = closing_a.fillna(df["B365A"])
    
    # Target: 1 jezeli wygrali gospodarze (H), 0 gdy goscie (A)
    df["target"] = (df["FTR"] == "H").astype(int)
    
    # Selekcja powyzszych kolumn
    out_cols = [
        "match_id", "date", "team_a", "team_b", 
        "team_a_kills", "team_b_kills", 
        "odds_home", "odds_away", 
        "closing_odds_home", "closing_odds_away", "target"
    ]
    df = df[out_cols].dropna()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Przygotowano zgrupowane dane: {len(df)} spotkan bez-remisowych.")
    return df

if __name__ == "__main__":
    df_real = fetch_and_prepare_data()
    
    print("\n--- Uruchomienie Pipeline na Prawdziwych Danych Rynkowych ---")
    results, _model, metrics, model_metrics = run_pipeline_from_dataframe(
        df_real, min_ev=0.03, initial_bankroll=10000.0, kelly_fraction=0.1
    )
    
    # Drukowanie tak samo jak w main.py
    print(f"ROC-AUC (test): {model_metrics.get('roc_auc', 0.0):.4f} | Log-loss: {model_metrics.get('log_loss', 0.0):.4f}")
    
    print("\n--- Wyniki Backtestu ---")
    print(f"Zaklady postawione: {int(metrics.get('total_bets', 0))}")
    print(f"ROI: {metrics.get('roi', 0.0):.2f}%")
    print(f"Poczatkowy bankroll: 10000.00 -> Koncowy: {metrics.get('final_bankroll', 0.0):.2f}")
    print(f"CLV Srednie: {metrics.get('avg_clv', 0.0):.2f}%")
    print(f"Srednia Wartosc Oczekiwana (EV) postawionych zakladow: {metrics.get('avg_bet_ev_pct', 0.0):.2f}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0.0):.1f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
    
    print("\n--- Dokladnosc Flagowania Betow ---")
    print(f"Win Rate dla postawionych zakladow: {metrics.get('bet_win_rate_pct', 0.0):.2f}%")
    print(f"Implied Win Rate z kursu: {metrics.get('implied_win_rate_pct', 0.0):.2f}%")
    print(f"Przecietne P(Win) z modelu: {metrics.get('model_p_mean_pct', 0.0):.2f}%")
    
    # Zapisz na dysku
    df_real.to_csv("data/real_market_data.csv", index=False)
    results.to_csv("real_backtest_results.csv", index=False)
    print("\nZapisano data/real_market_data.csv oraz real_backtest_results.csv")

