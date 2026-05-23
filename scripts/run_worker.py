"""Worker: cykliczny fetch + paper trading."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_loader import load_pipeline_config
from scheduler.run_cycle import run_cycle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Jeden cykl i koniec")
    args = parser.parse_args()
    cfg = load_pipeline_config()
    interval = cfg.get("scheduler", {}).get("interval_minutes", 30) * 60

    while True:
        report = run_cycle()
        print(report)
        if args.once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
