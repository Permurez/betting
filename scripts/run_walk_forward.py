"""Walk-forward backtest CLI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest.walk_forward import run_walk_forward
from config_loader import load_pipeline_config
from main import generate_full_pipeline_data


def main() -> None:
    cfg = load_pipeline_config().get("backtest", {})
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    raw = generate_full_pipeline_data(n)
    results, summary = run_walk_forward(
        raw,
        n_folds=cfg.get("walk_forward_folds", 5),
        min_train_rows=cfg.get("min_train_rows", 500),
        use_ensemble=True,
    )
    print("Walk-forward summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
