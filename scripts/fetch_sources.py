"""CLI: pobierz wszystkie zrodla do data/cache/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.runner import CollectorRunner


def main() -> None:
    runner = CollectorRunner()
    results = runner.run_all(
        include_news=True,
        include_odds=True,
        include_instagram="--instagram" in sys.argv,
        include_stats="--stats" in sys.argv,
    )
    for name, res in results.items():
        status = "OK" if res.success else "FAIL"
        print(f"[{status}] {name}: {res.message}")
        if res.meta.get("cache_path"):
            print(f"       -> {res.meta['cache_path']}")


if __name__ == "__main__":
    main()
