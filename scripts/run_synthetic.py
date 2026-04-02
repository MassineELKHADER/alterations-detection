import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_synthetic_config
from src.experiments import run_synthetic_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic alteration-detection study.")
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_synthetic_config(args.config)
    table_one, table_two = run_synthetic_study(config, args.output)
    print("Experiment 1")
    print(table_one.groupby("omega")[["precision", "recall", "f1"]].mean().round(3).to_string())
    print("\nExperiment 2")
    print(table_two.groupby("shift")[["precision", "recall", "f1"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
