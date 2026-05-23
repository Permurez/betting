"""Generuje duzy plik CSV do testow importu (min. 200 wierszy)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import generate_full_pipeline_data

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    out = Path(__file__).resolve().parents[1] / "data" / "matches_synthetic.csv"
    df = generate_full_pipeline_data(n)
    df.to_csv(out, index=False)
    print(f"Zapisano {len(df)} wierszy -> {out}")
